"""
Inference script for semantic segmentation with AerialFormer / mmsegmentation.

Processes large geospatial rasters (GeoTIFF or VRT) using a sliding window
approach with overlap, writing results directly to disk to avoid memory
limitations.

Overlap modes:
    crop   — keep only the center crop of each window (fast, possible seams)
    blend  — average softmax probabilities in overlap zones (slower, seamless)

Key features:
    - Windowed disk writes:  output is written tile-by-tile via rasterio,
      so arbitrarily large rasters (including VRT mosaics) can be processed
      without allocating the full output in memory.
    - Blend mode uses a rolling buffer that holds only window_size rows of
      probability data at a time (~2 GB for a 217,600-px wide raster with
      4 classes), not the full output.
    - Threaded pipeline (crop mode):  read → infer → write run in parallel.
"""

import argparse
import glob
import os
import threading
from itertools import groupby
from queue import Queue

import mmcv
import numpy as np
import torch
import torch.nn.functional as F
import rasterio
from rasterio.windows import Window as RioWindow
from pyproj import CRS

# Suppress GDAL/PROJ noise
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
os.environ["CPL_LOG_ERRORS"] = "ON"
os.environ["PROJ_DEBUG"] = "0"
try:
    from osgeo import gdal
    gdal.PushErrorHandler("CPLQuietErrorHandler")
except ImportError:
    pass

from mmseg.apis import init_segmentor, inference_segmentor
import aerialseg


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run semantic segmentation inference on large geospatial rasters."
    )

    # Paths
    parser.add_argument("--config", required=True, help="Path to mmseg config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained checkpoint (.pth).")
    parser.add_argument("--input", required=True,
                        help="Input path: a directory of .tif files, a single .tif, or a .vrt file.")
    parser.add_argument("--output_dir", required=True, help="Directory for output segmentation maps.")

    # Geospatial
    parser.add_argument("--crs", type=int, default=3007,
                        help="Fallback EPSG code if source CRS is missing (default: 3007).")

    # Sliding window
    parser.add_argument("--window_size", type=int, default=512,
                        help="Sliding window size in pixels (default: 512).")
    parser.add_argument("--stride", type=int, default=384,
                        help="Stride between windows (default: 384). Smaller = more overlap.")

    # Overlap handling
    parser.add_argument("--overlap_mode", type=str, default="crop",
                        choices=["crop", "blend"],
                        help=("How to handle overlapping window predictions. "
                              "'crop' keeps only the center of each window (fast). "
                              "'blend' averages softmax probabilities in overlap zones "
                              "(slower, seamless). Default: crop."))

    # Processing
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for inference (default: cuda:0).")
    parser.add_argument("--batch_queue_size", type=int, default=4,
                        help="Max windows buffered in read/write queues (default: 4).")

    # Band selection
    parser.add_argument("--bands", type=str, default=None,
                        help=("Comma-separated 1-based band indices to read from input "
                              "(e.g. '1,2,3' for RGB). Default: use all bands."))

    # Output
    parser.add_argument("--nodata", type=int, default=255,
                        help="Nodata value for unprocessed pixels in output (default: 255).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------

def create_grids(width, height, window_size, stride):
    """
    Compute sliding window positions with overlap-aware crop regions.

    Each grid entry is a dict with fields for both crop and blend modes:
        read_x, read_y   : top-left corner of the full window to read
        win_w, win_h     : size of the window (always window_size x window_size)
        crop_x, crop_y   : offset within window of the region to keep (crop mode)
        crop_w, crop_h   : size of the region to keep (crop mode)
        write_x, write_y : output position for the kept region (crop mode)
    """
    half_overlap_x = (window_size - stride) // 2
    half_overlap_y = (window_size - stride) // 2

    grids = []

    for y in range(0, height, stride):
        y_end = (y + window_size >= height)
        read_y = (height - window_size) if y_end else y

        crop_y = 0 if read_y == 0 else half_overlap_y
        crop_h = (window_size - crop_y) if y_end else (window_size - crop_y if read_y == 0 else stride)
        if read_y == 0 and y_end:
            crop_h = window_size

        for x in range(0, width, stride):
            x_end = (x + window_size >= width)
            read_x = (width - window_size) if x_end else x

            crop_x = 0 if read_x == 0 else half_overlap_x
            crop_w = (window_size - crop_x) if x_end else (window_size - crop_x if read_x == 0 else stride)
            if read_x == 0 and x_end:
                crop_w = window_size

            grids.append({
                "read_x": read_x, "read_y": read_y,
                "win_w": window_size, "win_h": window_size,
                "crop_x": crop_x, "crop_y": crop_y,
                "crop_w": crop_w, "crop_h": crop_h,
                "write_x": read_x + crop_x,
                "write_y": read_y + crop_y,
            })

            if x_end:
                break
        if y_end:
            break

    return grids


# ---------------------------------------------------------------------------
# Image preparation (shared by both modes)
# ---------------------------------------------------------------------------

def prepare_window(window_data):
    """Convert rasterio RGB data to BGR for mmseg inference."""
    if window_data.shape[2] >= 3:
        window_data = window_data[:, :, :3]
        window_data = window_data[:, :, ::-1].copy()  # RGB → BGR
    return window_data


# ---------------------------------------------------------------------------
# Softmax probability extraction (blend mode)
# ---------------------------------------------------------------------------

def get_softmax_probs(model, img_data):
    """
    Run inference and return softmax class probabilities.

    Temporarily patches model.encode_decode to capture raw logits before
    argmax.  This works regardless of decode head architecture (no hooks
    or assumptions about internal layer names).

    Returns:
        probs : (num_classes, H, W) float32 numpy array
    """
    captured = {}
    original_fn = model.encode_decode

    def capturing_encode_decode(img, img_meta):
        logit = original_fn(img, img_meta)
        captured["logit"] = logit.detach()
        return logit

    model.encode_decode = capturing_encode_decode
    try:
        result = inference_segmentor(model, img_data)
    finally:
        model.encode_decode = original_fn

    seg_map = result[0]
    target_h, target_w = seg_map.shape

    logit = captured["logit"]  # (1, C, h, w)
    if logit.shape[2] != target_h or logit.shape[3] != target_w:
        logit = F.interpolate(
            logit, size=(target_h, target_w),
            mode="bilinear", align_corners=False,
        )

    probs = torch.softmax(logit, dim=1)[0].cpu().numpy()  # (C, H, W)
    return probs.astype(np.float32)


# ---------------------------------------------------------------------------
# Rolling buffer for blend mode
# ---------------------------------------------------------------------------

class BlendAccumulator:
    """
    Rolling buffer that accumulates softmax probabilities and flushes
    averaged argmax results to disk row-by-row.

    Memory usage: num_classes * buf_h * width * 4 bytes  (float32 probs)
                + buf_h * width * 1 byte                 (uint8 counts)

    For width=217600, 4 classes, window_size=512:  ~1.9 GB total.
    """

    def __init__(self, num_classes, width, window_size, nodata=255):
        self.num_classes = num_classes
        self.width = width
        self.nodata = nodata

        self.buf_h = window_size + 1  # enough for one grid row of windows
        self.prob_buf = np.zeros((num_classes, self.buf_h, width), dtype=np.float32)
        self.count_buf = np.zeros((self.buf_h, width), dtype=np.uint8)
        self.buf_origin = 0  # image row at buffer row 0

    def accumulate(self, probs, read_x, read_y, win_w, win_h):
        """Add full-window probabilities into the buffer."""
        by = read_y - self.buf_origin
        self.prob_buf[:, by:by + win_h, read_x:read_x + win_w] += probs
        self.count_buf[by:by + win_h, read_x:read_x + win_w] += 1

    def flush_to(self, dst, flush_row):
        """Finalize rows [buf_origin, flush_row) and write to output."""
        n = flush_row - self.buf_origin
        if n <= 0:
            return

        count_slice = self.count_buf[:n, :]
        prob_slice = self.prob_buf[:, :n, :]

        # Average probabilities where we have predictions, then argmax
        mask = count_slice > 0
        count_safe = np.where(mask, count_slice, 1).astype(np.float32)
        avg = prob_slice / count_safe[np.newaxis, :, :]
        result = np.argmax(avg, axis=0).astype(np.uint8)
        result[~mask] = self.nodata

        # Write to disk
        window = RioWindow(0, self.buf_origin, self.width, n)
        dst.write(result[np.newaxis, :, :], window=window)

        # Shift buffer: move unfinished rows to the front
        remaining = self.buf_h - n
        self.prob_buf[:, :remaining, :] = self.prob_buf[:, n:n + remaining, :]
        self.prob_buf[:, remaining:, :] = 0
        self.count_buf[:remaining, :] = self.count_buf[n:n + remaining, :]
        self.count_buf[remaining:, :] = 0
        self.buf_origin = flush_row

    def flush_remaining(self, dst, total_height):
        """Flush all buffered rows up to total_height."""
        self.flush_to(dst, total_height)


# ---------------------------------------------------------------------------
# Crop mode — threaded pipeline (original approach)
# ---------------------------------------------------------------------------

SENTINEL = object()


def read_worker(src_path, grids, read_queue, band_indices=None):
    """Read windows from the source raster and push to the read queue."""
    with rasterio.open(src_path) as src:
        for grid in grids:
            window = RioWindow(
                col_off=grid["read_x"], row_off=grid["read_y"],
                width=grid["win_w"], height=grid["win_h"],
            )
            if band_indices is not None:
                data = src.read(window=window, indexes=band_indices)
            else:
                data = src.read(window=window)

            data = np.transpose(data, (1, 2, 0))  # (bands,H,W) → (H,W,bands)
            read_queue.put((grid, data))

    read_queue.put(SENTINEL)


def inference_worker_crop(model, read_queue, write_queue):
    """Infer argmax class maps for crop mode."""
    while True:
        item = read_queue.get()
        if item is SENTINEL:
            write_queue.put(SENTINEL)
            break

        grid, window_data = item
        window_data = prepare_window(window_data)

        result = inference_segmentor(model, window_data)
        seg_map = np.array(result[0], dtype=np.uint8)
        write_queue.put((grid, seg_map))


def write_worker_crop(dst, write_queue, progress_bar):
    """Write center crops to the output GeoTIFF."""
    while True:
        item = write_queue.get()
        if item is SENTINEL:
            break

        grid, seg_map = item
        cropped = seg_map[
            grid["crop_y"]:grid["crop_y"] + grid["crop_h"],
            grid["crop_x"]:grid["crop_x"] + grid["crop_w"],
        ]
        out_window = RioWindow(
            col_off=grid["write_x"], row_off=grid["write_y"],
            width=grid["crop_w"], height=grid["crop_h"],
        )
        dst.write(cropped[np.newaxis, :, :], window=out_window)
        progress_bar.update()


def process_raster_crop(model, src_path, output_path, args, out_profile):
    """Process with crop mode: threaded pipeline, center-crop overlap handling."""
    grids = create_grids(
        out_profile["width"], out_profile["height"],
        args.window_size, args.stride,
    )
    print(f"  Grid windows: {len(grids)}")
    print(f"  Overlap mode: crop (center-crop, threaded)")

    band_indices = _parse_bands(args)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        _fill_nodata(dst, out_profile["height"], out_profile["width"], args.nodata)

        read_q = Queue(maxsize=args.batch_queue_size)
        write_q = Queue(maxsize=args.batch_queue_size)
        progress_bar = mmcv.ProgressBar(len(grids))

        reader = threading.Thread(
            target=read_worker, args=(src_path, grids, read_q, band_indices))
        inferrer = threading.Thread(
            target=inference_worker_crop, args=(model, read_q, write_q))
        writer = threading.Thread(
            target=write_worker_crop, args=(dst, write_q, progress_bar))

        reader.start(); inferrer.start(); writer.start()
        reader.join();  inferrer.join();  writer.join()

    print(f"\n  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Blend mode — threaded pipeline with rolling accumulation
# ---------------------------------------------------------------------------

def inference_worker_blend(model, read_queue, write_queue):
    """Infer softmax probabilities for blend mode."""
    while True:
        item = read_queue.get()
        if item is SENTINEL:
            write_queue.put(SENTINEL)
            break

        grid, window_data = item
        window_data = prepare_window(window_data)

        probs = get_softmax_probs(model, window_data)  # (C, H, W)
        write_queue.put((grid, probs))


def write_worker_blend(dst, write_queue, progress_bar, accumulator, total_height):
    """Accumulate probabilities and flush completed rows to disk."""
    prev_read_y = -1

    while True:
        item = write_queue.get()
        if item is SENTINEL:
            break

        grid, probs = item
        read_y = grid["read_y"]

        # When we move to a new grid row, all rows before it are complete
        # because no future window can affect rows < read_y.
        if read_y > prev_read_y and prev_read_y >= 0:
            accumulator.flush_to(dst, read_y)
        prev_read_y = read_y

        # Accumulate the FULL window of probabilities (not just the crop)
        accumulator.accumulate(
            probs, grid["read_x"], grid["read_y"],
            grid["win_w"], grid["win_h"],
        )
        progress_bar.update()

    # Flush everything remaining
    accumulator.flush_remaining(dst, total_height)


def process_raster_blend(model, src_path, output_path, args, out_profile):
    """Process with blend mode: threaded pipeline, probability-averaged overlaps."""
    width = out_profile["width"]
    height = out_profile["height"]

    grids = create_grids(width, height, args.window_size, args.stride)
    print(f"  Grid windows: {len(grids)}")

    num_classes = model.decode_head.num_classes
    buf_mem_mb = (num_classes * (args.window_size + 1) * width * 4
                  + (args.window_size + 1) * width) / (1024 ** 2)
    print(f"  Overlap mode: blend (probability averaging)")
    print(f"  Classes: {num_classes},  rolling buffer: ~{buf_mem_mb:.0f} MB")

    band_indices = _parse_bands(args)
    accumulator = BlendAccumulator(num_classes, width, args.window_size, args.nodata)

    with rasterio.open(output_path, "w", **out_profile) as dst:
        read_q = Queue(maxsize=args.batch_queue_size)
        write_q = Queue(maxsize=args.batch_queue_size)
        progress_bar = mmcv.ProgressBar(len(grids))

        reader = threading.Thread(
            target=read_worker, args=(src_path, grids, read_q, band_indices))
        inferrer = threading.Thread(
            target=inference_worker_blend, args=(model, read_q, write_q))
        writer = threading.Thread(
            target=write_worker_blend,
            args=(dst, write_q, progress_bar, accumulator, height))

        reader.start(); inferrer.start(); writer.start()
        reader.join();  inferrer.join();  writer.join()

    print(f"\n  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _parse_bands(args):
    if args.bands is not None:
        return [int(b) for b in args.bands.split(",")]
    return None


def _fill_nodata(dst, height, width, nodata):
    """Fill the output raster with nodata in memory-friendly strips."""
    strip_h = 512
    nodata_strip = np.full((1, strip_h, width), nodata, dtype=np.uint8)
    for row in range(0, height, strip_h):
        h = min(strip_h, height - row)
        dst.write(nodata_strip[:, :h, :], window=RioWindow(0, row, width, h))


def _build_output_profile(src_path, args):
    """Read source metadata and build the output GeoTIFF profile."""
    with rasterio.open(src_path) as src:
        width, height = src.width, src.height
        src_crs = src.crs
        src_transform = src.transform

    print(f"  Dimensions: {width} x {height} pixels")

    if src_crs:
        out_crs = src_crs
    else:
        out_crs = CRS.from_epsg(args.crs)

    return {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "uint8",
        "crs": out_crs,
        "transform": src_transform,
        "compress": "lzw",
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "nodata": args.nodata,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def process_raster(model, src_path, output_path, args):
    """Route to the correct processing mode."""
    print(f"\nProcessing: {os.path.basename(src_path)}")
    out_profile = _build_output_profile(src_path, args)

    if args.overlap_mode == "blend":
        process_raster_blend(model, src_path, output_path, args, out_profile)
    else:
        process_raster_crop(model, src_path, output_path, args, out_profile)


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("\nInitializing model...")
    model = init_segmentor(args.config, args.checkpoint, device=args.device)

    input_path = args.input

    if input_path.lower().endswith(".vrt"):
        out_name = os.path.splitext(os.path.basename(input_path))[0] + "_segmentation.tif"
        output_path = os.path.join(args.output_dir, out_name)
        process_raster(model, input_path, output_path, args)

    elif os.path.isfile(input_path) and input_path.lower().endswith(".tif"):
        out_name = os.path.splitext(os.path.basename(input_path))[0] + "_segmentation.tif"
        output_path = os.path.join(args.output_dir, out_name)
        process_raster(model, input_path, output_path, args)

    elif os.path.isdir(input_path):
        tif_files = sorted(glob.glob(os.path.join(input_path, "*.tif")))
        print(f"Found {len(tif_files)} GeoTIFF files to process.")
        for fpath in tif_files:
            out_name = os.path.splitext(os.path.basename(fpath))[0] + "_segmentation.tif"
            output_path = os.path.join(args.output_dir, out_name)
            process_raster(model, fpath, output_path, args)

    else:
        print(f"Error: Input path not recognized: {input_path}")
        return

    print("\nAll processing complete.")


if __name__ == "__main__":
    main()