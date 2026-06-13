"""
Tile preparation script for semantic segmentation (mmsegmentation format).

Reads large geospatial TIFF images and their single-band indexed annotations
from a directory structure, clips them into smaller tiles, and organizes them
into train/val splits.

Expected input structure (one or more dataset roots):
    dataset_root/
        image_dir/
            6398_149_train_image.tif   OR   6398_149_image.tif
            ...
        annotation_dir/
            6398_149_train_annotation.tif   OR   6398_149_annotation.tif
            ...

Filename formats supported:
    {id1}_{id2}_{split}_{type}.tif   — split encoded in filename
    {id1}_{id2}_{type}.tif           — no split; use --train_ratio

Split modes (checked in this order of priority):
    1. --split_direction + --train_ratio  →  spatial (geographic) split
    2. --train_ratio (alone)              →  random whole-file split
    3. neither                            →  filename-encoded split tokens

Output structure:
    out_dir/
        img_dir/train/   val/
        ann_dir/train/   val/
"""

import argparse
import glob
import math
import os
import os.path as osp
import random
import shutil
import logging
import tempfile
from pathlib import Path
from collections import defaultdict

# --- Suppress GDAL/PROJ noise before importing rasterio ---
os.environ["GTIFF_SRS_SOURCE"] = "EPSG"
os.environ["CPL_LOG_ERRORS"] = "ON"
os.environ["PROJ_DEBUG"] = "0"
try:
    from osgeo import gdal
    gdal.PushErrorHandler("CPLQuietErrorHandler")
except ImportError:
    pass

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.transform import Affine
from pyproj import CRS
import mmcv


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Tile large geospatial images for semantic segmentation training."
    )

    # --- Paths ---
    parser.add_argument(
        "dataset_paths",
        nargs="+",
        help=(
            "One or more dataset root directories, each containing "
            "'image_dir/' and 'annotation_dir/' subdirectories."
        ),
    )
    parser.add_argument("-o", "--out_dir", required=True, help="Output root directory.")
    parser.add_argument("--tmp_dir", default=None, help="Temporary directory for intermediate files.")

    # --- Tiling ---
    parser.add_argument("--clip_size", type=int, default=512, help="Tile size in pixels (default: 512).")
    parser.add_argument(
        "--stride_size",
        type=int,
        default=256,
        help="Stride for sliding window on *training* tiles (default: 256). Validation always uses stride=clip_size.",
    )

    # --- Geospatial ---
    parser.add_argument(
        "--crs",
        type=int,
        default=3007,
        help="Fallback EPSG code when source CRS is missing/local (default: 3007 / SWEREF99 12 00).",
    )

    # --- Output format ---
    parser.add_argument(
        "--img_format",
        choices=["tif", "png"],
        default="tif",
        help="Output format for image tiles (default: tif).",
    )
    parser.add_argument(
        "--ann_format",
        choices=["tif", "png"],
        default="tif",
        help="Output format for annotation tiles (default: tif).",
    )
    parser.add_argument(
        "--bands",
        type=str,
        default="1,2,3,4",
        help='Comma-separated 1-based band indices to keep for images (default: "1,2,3,4"). Use "all" for every band.',
    )

    # --- Georef copying ---
    parser.add_argument(
        "--copy_georef",
        action="store_true",
        help="Copy georeference info from images onto annotations (useful when labels lack CRS/transform).",
    )

    # --- Splitting ---
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=None,
        help=(
            "Fraction of data for training (e.g. 0.8). "
            "Combined with --split_direction for spatial splits, "
            "or used alone for random whole-file splits."
        ),
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed for reproducible splits (default: 42)."
    )
    parser.add_argument(
        "--split_direction",
        type=str,
        default=None,
        choices=["ns", "sn", "ew", "we"],
        help=(
            "Spatial split direction. Requires --train_ratio. "
            "ns = train is North portion; "
            "sn = train is South portion; "
            "ew = train is East portion; "
            "we = train is West portion."
        ),
    )

    # --- Split name mapping ---
    parser.add_argument(
        "--val_split_names",
        type=str,
        default="test,val",
        help=(
            'Comma-separated list of split tokens in filenames that map to the '
            'validation set (default: "test,val"). Everything else maps to train.'
        ),
    )

    # --- Inference ---
    parser.add_argument(
        "--inference",
        action="store_true",
        help="Inference mode: tile images only (no annotations, no train/val split).",
    )

    args = parser.parse_args()

    # Validate
    if args.split_direction and args.train_ratio is None:
        parser.error("--split_direction requires --train_ratio.")

    if args.inference:
        pass
    else:
        for p in args.dataset_paths:
            img_dir = os.path.join(p, "image_dir")
            ann_dir = os.path.join(p, "annotation_dir")
            if not os.path.isdir(img_dir):
                parser.error(f"image_dir not found: {img_dir}")
            if not os.path.isdir(ann_dir):
                parser.error(f"annotation_dir not found: {ann_dir}")

    return args


# ---------------------------------------------------------------------------
# Filename parsing
# ---------------------------------------------------------------------------

def parse_filename(filepath):
    """
    Parse filenames in one of two formats:

        With split token:    6398_149_train_image.tif
        Without split token: 6398_149_image.tif

    Returns
    -------
    dict with keys:
        tile_id  : str          e.g. '6398_149'
        split    : str or None  e.g. 'train', 'test', or None
        role     : str          'image' or 'annotation'
        stem     : str
        ext      : str
    None if the filename doesn't match either pattern.
    """
    stem = Path(filepath).stem
    ext = Path(filepath).suffix
    parts = stem.split("_")

    if len(parts) < 3:
        return None

    id1 = parts[0]
    id2 = parts[1]

    # Last token must be the role
    if parts[-1] in ("image", "annotation"):
        role = parts[-1]
        split_token = parts[2] if len(parts) == 4 else None
    else:
        return None

    return {
        "tile_id": f"{id1}_{id2}",
        "split": split_token,
        "role": role,
        "stem": stem,
        "ext": ext,
    }


# ---------------------------------------------------------------------------
# Georef helper
# ---------------------------------------------------------------------------

class GeorefInfo:
    """Lightweight container for CRS + affine transform."""

    def __init__(self, crs, transform):
        self.crs = crs
        self.transform = transform


def store_georef_info(src_path):
    """Read CRS and transform from a raster file."""
    with rasterio.open(src_path) as src:
        return GeorefInfo(src.crs, src.transform)


# ---------------------------------------------------------------------------
# Spatial (geographic) splitting
# ---------------------------------------------------------------------------

def compute_spatial_split_windows(src, direction, train_ratio):
    """
    Compute rasterio Windows for a spatial train/val split of a raster.

    The split divides the raster along one axis at the position determined
    by *train_ratio*.  The "train" portion always corresponds to the first
    part in the named direction:

        ns  →  train = North (top rows),     val = South (bottom rows)
        sn  →  train = South (bottom rows),  val = North (top rows)
        ew  →  train = East  (right cols),   val = West  (left cols)
        we  →  train = West  (left cols),    val = East  (right cols)

    Note on orientation: raster row 0 is the top (north), column 0 is the
    left (west).  The split names describe *geographic* direction, so "ns"
    means training data comes from the northern (upper) portion.

    Parameters
    ----------
    src : rasterio.DatasetReader
        Opened raster (only .height / .width are used).
    direction : str
        One of 'ns', 'sn', 'ew', 'we'.
    train_ratio : float
        Fraction of pixels assigned to the training portion.

    Returns
    -------
    train_window, val_window : rasterio.windows.Window
    """
    h, w = src.height, src.width
    direction = direction.lower()

    if direction in ("ns", "sn"):
        # Split along rows (Y-axis)
        train_rows = round(h * train_ratio)
        val_rows = h - train_rows

        if direction == "ns":
            # North = top rows → train
            train_win = Window(col_off=0, row_off=0,
                               width=w, height=train_rows)
            val_win   = Window(col_off=0, row_off=train_rows,
                               width=w, height=val_rows)
        else:  # sn
            # South = bottom rows → train
            train_win = Window(col_off=0, row_off=h - train_rows,
                               width=w, height=train_rows)
            val_win   = Window(col_off=0, row_off=0,
                               width=w, height=val_rows)

    else:  # ew, we
        # Split along columns (X-axis)
        train_cols = round(w * train_ratio)
        val_cols = w - train_cols

        if direction == "we":
            # West = left cols → train
            train_win = Window(col_off=0, row_off=0,
                               width=train_cols, height=h)
            val_win   = Window(col_off=train_cols, row_off=0,
                               width=val_cols, height=h)
        else:  # ew
            # East = right cols → train
            train_win = Window(col_off=w - train_cols, row_off=0,
                               width=train_cols, height=h)
            val_win   = Window(col_off=0, row_off=0,
                               width=val_cols, height=h)

    return train_win, val_win


def write_windowed_raster(src_path, window, out_path, is_label=False):
    """
    Read a window from *src_path* and write it to *out_path* as a new GeoTIFF,
    preserving CRS and computing the correct sub-window transform.

    Parameters
    ----------
    src_path : str
    window : rasterio.windows.Window
    out_path : str
    is_label : bool
        If True, keep only the first band and cast to uint8.
    """
    with rasterio.open(src_path) as src:
        sub_transform = rasterio.windows.transform(window, src.transform)

        if is_label:
            data = src.read(window=window)
            data = data[0:1, :, :].astype(np.uint8)
            count = 1
            dtype = "uint8"
        else:
            data = src.read(window=window)
            count = data.shape[0]
            dtype = data.dtype

        profile = {
            "driver": "GTiff",
            "height": window.height,
            "width": window.width,
            "count": count,
            "dtype": dtype,
            "compress": "lzw",
            "crs": src.crs,
            "transform": sub_transform,
        }

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(data)


def spatial_split_pair(image_path, annotation_path, direction, train_ratio,
                       tmp_dir, tile_id):
    """
    Spatially split an image / annotation pair into train and val portions.

    Writes four temporary TIFFs into *tmp_dir* and returns their paths.

    Returns
    -------
    dict  with keys 'train' and 'val', each containing
          {'image': path, 'annotation': path}
    """
    with rasterio.open(image_path) as src:
        train_win, val_win = compute_spatial_split_windows(
            src, direction, train_ratio
        )

    result = {}
    for split_name, win in [("train", train_win), ("val", val_win)]:
        img_tmp = os.path.join(tmp_dir, f"{tile_id}_{split_name}_image.tif")
        ann_tmp = os.path.join(tmp_dir, f"{tile_id}_{split_name}_annotation.tif")

        write_windowed_raster(image_path, win, img_tmp, is_label=False)
        write_windowed_raster(annotation_path, win, ann_tmp, is_label=True)

        result[split_name] = {"image": img_tmp, "annotation": ann_tmp}

    logging.info(
        f"  \n\nSpatial split ({direction}) of {tile_id}: "
        f"\ntrain window {train_win.width}x{train_win.height}px, "
        f"\nval window {val_win.width}x{val_win.height}px"
    )
    return result


# ---------------------------------------------------------------------------
# Core tiling function
# ---------------------------------------------------------------------------

def clip_big_image(
    image_path,
    clip_save_dir,
    clip_size=512,
    stride_size=256,
    is_label=False,
    epsg_code=3007,
    output_format="tif",
    bands=None,
    georef_info=None,
    tile_prefix="tile",
):
    """
    Clip a large raster into fixed-size tiles using a sliding window.

    Parameters
    ----------
    image_path : str
        Path to source TIFF.
    clip_save_dir : str
        Directory to write tiles into.
    clip_size : int
        Tile edge length in pixels.
    stride_size : int
        Sliding window step.  When < clip_size, tiles overlap.
    is_label : bool
        If True, treat as single-band annotation raster.
    epsg_code : int
        Fallback EPSG if source CRS is missing.
    output_format : str
        'tif' or 'png'.
    bands : str or None
        Comma-separated 1-based band indices (only applied to images, not labels).
        Use 'all' or None for every band.
    georef_info : GeorefInfo or None
        Optional external georef to apply (used with --copy_georef).
    tile_prefix : str
        Prefix for output tile filenames (typically the tile_id).

    Returns
    -------
    processed_count : int
    """
    # --- Parse band selection ---
    selected_bands = None
    if bands is not None and bands.lower() != "all" and not is_label:
        try:
            selected_bands = [int(b) for b in bands.split(",")]
            if any(b <= 0 for b in selected_bands):
                raise ValueError("Band indices must be >= 1")
        except ValueError as e:
            logging.error(f"Invalid band selection: {e}")
            return 0

    processed_count = 0

    # --- Open source raster ---
    if output_format == "png":
        image = mmcv.imread(image_path, backend="tifffile")
        h, w = image.shape[:2]
        if selected_bands is not None:
            zero_based = [b - 1 for b in selected_bands]
            if image.ndim == 3:
                image = image[:, :, zero_based]
        image = mmcv.bgr2rgb(image)
    else:
        src = rasterio.open(image_path)
        h, w = src.height, src.width
        num_bands = len(selected_bands) if selected_bands else src.count
        transform = georef_info.transform if georef_info else src.transform
        original_crs = georef_info.crs if georef_info else src.crs

    # --- Compute tile grid ---
    n_rows = math.ceil((h - clip_size) / stride_size) + 1 if h > clip_size else 1
    n_cols = math.ceil((w - clip_size) / stride_size) + 1 if w > clip_size else 1

    x, y = np.meshgrid(np.arange(n_cols), np.arange(n_rows))
    xmin = (x * stride_size).ravel()
    ymin = (y * stride_size).ravel()

    # Clamp so tiles don't exceed image bounds
    xmin = np.clip(xmin, 0, max(w - clip_size, 0))
    ymin = np.clip(ymin, 0, max(h - clip_size, 0))

    # Deduplicate positions (can happen at edges)
    coords = list({(int(xi), int(yi)) for xi, yi in zip(xmin, ymin)})

    try:
        for start_x, start_y in coords:
            end_x = min(start_x + clip_size, w)
            end_y = min(start_y + clip_size, h)

            out_name = f"{tile_prefix}_{start_x}_{start_y}_{end_x}_{end_y}.{output_format}"
            output_path = osp.join(clip_save_dir, out_name)

            if output_format == "png":
                if is_label:
                    clip = image[start_y:end_y, start_x:end_x]
                    if clip.ndim == 3:
                        clip = clip[:, :, 0]
                    clip = clip.astype(np.uint8)
                else:
                    clip = image[start_y:end_y, start_x:end_x]
                mmcv.imwrite(clip.astype(np.uint8), output_path)

            else:  # tif
                window = Window(start_x, start_y, end_x - start_x, end_y - start_y)
                if selected_bands is not None:
                    data = src.read(window=window, indexes=selected_bands)
                else:
                    data = src.read(window=window)

                if is_label:
                    data = data[0:1, :, :]
                    data = data.astype(np.uint8)

                tile_transform = Affine(
                    transform.a, transform.b, transform.c + transform.a * start_x,
                    transform.d, transform.e, transform.f + transform.e * start_y,
                )

                effective_crs = (
                    original_crs
                    if original_crs and not original_crs.is_local
                    else CRS.from_epsg(epsg_code)
                )

                profile = {
                    "driver": "GTiff",
                    "height": end_y - start_y,
                    "width": end_x - start_x,
                    "count": data.shape[0],
                    "dtype": data.dtype,
                    "compress": "lzw",
                    "crs": effective_crs,
                    "transform": tile_transform,
                    "photometric": "MINISBLACK",
                }

                with rasterio.open(output_path, "w", **profile) as dst:
                    dst.write(data)

            processed_count += 1

    except Exception as e:
        logging.error(f"Error tiling {image_path}: {e}")
        return processed_count

    if output_format != "png":
        src.close()

    return processed_count


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def discover_pairs(dataset_paths, val_split_names):
    """
    Walk one or more dataset roots and pair images with annotations.

    Returns
    -------
    pairs : list of dict
        Each dict: {tile_id, split, image_path, annotation_path}
    """
    val_tokens = {s.strip().lower() for s in val_split_names.split(",")}
    pairs_by_id = defaultdict(dict)

    for root in dataset_paths:
        for subdir, role in [("image_dir", "image"), ("annotation_dir", "annotation")]:
            search_dir = os.path.join(root, subdir)
            if not os.path.isdir(search_dir):
                logging.warning(f"Directory not found, skipping: {search_dir}")
                continue
            for fpath in sorted(glob.glob(os.path.join(search_dir, "*.tif"))):
                info = parse_filename(fpath)
                if info is None:
                    logging.warning(f"Skipping file with unexpected name: {fpath}")
                    continue
                key = (info["tile_id"], info["split"])
                pairs_by_id[key][role] = fpath
                pairs_by_id[key]["tile_id"] = info["tile_id"]
                pairs_by_id[key]["split_token"] = info["split"]

    pairs = []
    for key, entry in pairs_by_id.items():
        if "image" not in entry or "annotation" not in entry:
            logging.warning(
                f"Unpaired tile {entry.get('tile_id', '?')} "
                f"(split={entry.get('split_token', '?')}), skipping."
            )
            continue

        split_token = entry["split_token"]
        if split_token is not None:
            split = "val" if split_token.lower() in val_tokens else "train"
        else:
            split = None

        pairs.append(
            {
                "tile_id": entry["tile_id"],
                "split": split,
                "image_path": entry["image"],
                "annotation_path": entry["annotation"],
            }
        )

    return pairs


# ---------------------------------------------------------------------------
# Random splitting
# ---------------------------------------------------------------------------

def random_split(pairs, train_ratio, seed):
    """Override the filename-encoded split with a random one."""
    random.seed(seed)
    shuffled = pairs.copy()
    random.shuffle(shuffled)
    n_train = int(len(shuffled) * train_ratio)
    for i, p in enumerate(shuffled):
        p["split"] = "train" if i < n_train else "val"
    return shuffled


# ---------------------------------------------------------------------------
# Summary / reporting
# ---------------------------------------------------------------------------

_DIRECTION_LABELS = {
    "ns": "North → South  (train = North, val = South)",
    "sn": "South → North  (train = South, val = North)",
    "ew": "East → West    (train = East,  val = West)",
    "we": "West → East    (train = West,  val = East)",
}


def print_parameters(args):
    """Print a human-readable block of the parameters used for this run."""

    # Determine split mode description
    if args.split_direction and args.train_ratio is not None:
        split_mode = "Spatial (geographic) split"
    elif args.train_ratio is not None:
        split_mode = "Random whole-file split"
    else:
        split_mode = "Filename-encoded split tokens"

    lines = [
        "",
        "=" * 60,
        "  PARAMETERS",
        "=" * 60,
        f"  Dataset source(s):            {', '.join(args.dataset_paths)}",
        f"  Output directory:             {args.out_dir}",
        "",
        f"  Tile size (clip_size):        {args.clip_size} px",
        f"  Training stride (overlap):    {args.stride_size} px",
        f"  Validation stride:            {args.clip_size} px  (no overlap)",
        "",
        f"  Image output format:          {args.img_format}",
        f"  Annotation output format:     {args.ann_format}",
        f"  Bands kept (images):          {args.bands}",
        f"  Fallback CRS:                 EPSG:{args.crs}",
        f"  Copy georef to annotations:   {args.copy_georef}",
        "",
        f"  Split mode:                   {split_mode}",
    ]

    if args.train_ratio is not None:
        lines.append(
            f"  Training ratio:               {args.train_ratio:.0%} train / "
            f"{1 - args.train_ratio:.0%} val"
        )
    if args.split_direction:
        lines.append(
            f"  Split direction:              {_DIRECTION_LABELS.get(args.split_direction, args.split_direction)}"
        )
    if args.train_ratio is not None and args.split_direction is None:
        lines.append(f"  Random seed:                  {args.random_seed}")
    if args.train_ratio is None:
        lines.append(
            f"  Validation filename tokens:   {args.val_split_names}"
        )

    lines.append("=" * 60)
    print("\n".join(lines))


def print_summary(
    args, pairs, use_spatial_split, use_random_split,
    train_img, train_ann, val_img, val_ann,
    total_img, total_ann,
):
    """Print a detailed results summary after processing."""

    n_source_pairs = len(pairs)
    n_train_sources = sum(1 for p in pairs if p["split"] == "train")
    n_val_sources = sum(1 for p in pairs if p["split"] == "val")

    # For spatial split every source pair contributes to both splits,
    # so "source pairs per split" doesn't apply the same way.
    if use_spatial_split:
        source_note = (
            f"  Source raster pairs:           {n_source_pairs}  "
            f"(each split spatially into train + val)"
        )
    else:
        source_note = (
            f"  Source raster pairs:           {n_source_pairs}  "
            f"({n_train_sources} train, {n_val_sources} val)"
        )

    lines = [
        "",
        "=" * 60,
        "  RESULTS",
        "=" * 60,
        source_note,
        "",
        "  Tiles produced                 Images     Annotations",
        f"    Training:                    {train_img:<10} {train_ann}",
        f"    Validation:                  {val_img:<10} {val_ann}",
        f"    Total:                       {total_img:<10} {total_ann}",
        "",
        f"  Output directory:             {args.out_dir}",
        f"    Training images:            {os.path.join(args.out_dir, 'img_dir', 'train')}",
        f"    Training annotations:       {os.path.join(args.out_dir, 'ann_dir', 'train')}",
        f"    Validation images:          {os.path.join(args.out_dir, 'img_dir', 'val')}",
        f"    Validation annotations:     {os.path.join(args.out_dir, 'ann_dir', 'val')}",
        "=" * 60,
    ]

    print("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    out_dir = args.out_dir

    # --- Logging setup ---
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, "processing.log")
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger().addHandler(console)

    # --- Suppress GDAL/PROJ terminal spam ---
    # CPL_LOG redirects GDAL's own log messages to a file
    os.environ["CPL_LOG"] = os.path.join(out_dir, "gdal_warnings.log")
    # Prevent PROJ errors from reaching stderr via GDAL's error handler
    os.environ["CPL_LOG_ERRORS"] = "ON"
    # Silence the rasterio logger (which forwards GDAL errors to Python logging)
    logging.getLogger("rasterio").setLevel(logging.ERROR)
    logging.getLogger("rasterio._env").setLevel(logging.ERROR)
    # If osgeo/gdal is available, register a custom error handler to swallow
    # PROJ version-mismatch warnings that bypass CPL_LOG
    try:
        from osgeo import gdal
        gdal.PushErrorHandler("CPLQuietErrorHandler")
    except ImportError:
        pass

    # --- Print parameter summary ---
    print_parameters(args)

    # ------------------------------------------------------------------
    # Inference mode
    # ------------------------------------------------------------------
    if args.inference:
        inference_out = os.path.join(out_dir, "inference")
        os.makedirs(inference_out, exist_ok=True)
        total = 0
        for ds_path in args.dataset_paths:
            src_files = sorted(glob.glob(os.path.join(ds_path, "*.tif")))
            logging.info(f"Inference: found {len(src_files)} images in {ds_path}")
            bar = mmcv.ProgressBar(len(src_files))
            for fpath in src_files:
                info = parse_filename(fpath)
                prefix = info["tile_id"] if info else Path(fpath).stem
                n = clip_big_image(
                    fpath,
                    inference_out,
                    clip_size=args.clip_size,
                    stride_size=args.clip_size,
                    is_label=False,
                    epsg_code=args.crs,
                    output_format=args.img_format,
                    bands=args.bands,
                    tile_prefix=prefix,
                )
                total += n
                bar.update()
        logging.info(f"Inference complete: {total} tiles produced.")
        return

    # ------------------------------------------------------------------
    # Training / validation preparation
    # ------------------------------------------------------------------

    # 1. Discover image-annotation pairs
    pairs = discover_pairs(args.dataset_paths, args.val_split_names)
    logging.info(f"Discovered {len(pairs)} image/annotation pairs.")

    if not pairs:
        logging.error("No valid pairs found. Check your dataset paths and filenames.")
        return

    # 2. Determine split strategy
    use_spatial_split = (args.split_direction is not None
                         and args.train_ratio is not None)
    use_random_split  = (args.train_ratio is not None
                         and args.split_direction is None)

    # 3. Create output directories
    for split in ("train", "val"):
        os.makedirs(os.path.join(out_dir, "img_dir", split), exist_ok=True)
        os.makedirs(os.path.join(out_dir, "ann_dir", split), exist_ok=True)

    total_img_tiles = 0
    total_ann_tiles = 0
    train_img_tiles = 0
    train_ann_tiles = 0
    val_img_tiles = 0
    val_ann_tiles = 0
    georef_cache = {}

    # ==================================================================
    # PATH A — Spatial (geographic) split
    #
    # Each raster pair is cropped into a train and val portion along a
    # geographic axis *before* tiling.  This guarantees zero spatial
    # overlap between training and validation pixels.
    # ==================================================================
    if use_spatial_split:
        logging.info(
            f"Spatial split mode: direction={args.split_direction}, "
            f"train_ratio={args.train_ratio}"
        )

        with tempfile.TemporaryDirectory(dir=args.tmp_dir) as spatial_tmp:
            bar = mmcv.ProgressBar(len(pairs))
            for pair in pairs:
                tile_id = pair["tile_id"]

                # --- Crop the full raster into two geographic halves ---
                split_result = spatial_split_pair(
                    pair["image_path"],
                    pair["annotation_path"],
                    direction=args.split_direction,
                    train_ratio=args.train_ratio,
                    tmp_dir=spatial_tmp,
                    tile_id=tile_id,
                )

                # --- Tile each half independently ---
                for split_name in ("train", "val"):
                    sub = split_result[split_name]
                    stride = (args.stride_size if split_name == "train"
                              else args.clip_size)

                    if args.copy_georef:
                        georef_cache[tile_id] = store_georef_info(sub["image"])

                    img_out = os.path.join(out_dir, "img_dir", split_name)
                    n = clip_big_image(
                        sub["image"],
                        img_out,
                        clip_size=args.clip_size,
                        stride_size=stride,
                        is_label=False,
                        epsg_code=args.crs,
                        output_format=args.img_format,
                        bands=args.bands,
                        tile_prefix=tile_id,
                    )
                    total_img_tiles += n
                    if split_name == "train":
                        train_img_tiles += n
                    else:
                        val_img_tiles += n

                    georef = (georef_cache.get(tile_id)
                              if args.copy_georef else None)
                    ann_out = os.path.join(out_dir, "ann_dir", split_name)
                    n = clip_big_image(
                        sub["annotation"],
                        ann_out,
                        clip_size=args.clip_size,
                        stride_size=stride,
                        is_label=True,
                        epsg_code=args.crs,
                        output_format=args.ann_format,
                        georef_info=georef,
                        tile_prefix=tile_id,
                    )
                    total_ann_tiles += n
                    if split_name == "train":
                        train_ann_tiles += n
                    else:
                        val_ann_tiles += n

                bar.update()

    # ==================================================================
    # PATH B — Random whole-file split
    # ==================================================================
    elif use_random_split:
        pairs = random_split(pairs, args.train_ratio, args.random_seed)
        n_train = sum(1 for p in pairs if p["split"] == "train")
        n_val = len(pairs) - n_train
        logging.info(
            f"Random split applied (ratio={args.train_ratio}, "
            f"seed={args.random_seed}): {n_train} train, {n_val} val."
        )

        bar = mmcv.ProgressBar(len(pairs))
        for pair in pairs:
            tile_id = pair["tile_id"]
            split = pair["split"]
            stride = args.stride_size if split == "train" else args.clip_size

            if args.copy_georef:
                georef_cache[tile_id] = store_georef_info(pair["image_path"])

            img_out = os.path.join(out_dir, "img_dir", split)
            n = clip_big_image(
                pair["image_path"],
                img_out,
                clip_size=args.clip_size,
                stride_size=stride,
                is_label=False,
                epsg_code=args.crs,
                output_format=args.img_format,
                bands=args.bands,
                tile_prefix=tile_id,
            )
            total_img_tiles += n
            if split == "train":
                train_img_tiles += n
            else:
                val_img_tiles += n

            georef = georef_cache.get(tile_id) if args.copy_georef else None
            ann_out = os.path.join(out_dir, "ann_dir", split)
            n = clip_big_image(
                pair["annotation_path"],
                ann_out,
                clip_size=args.clip_size,
                stride_size=stride,
                is_label=True,
                epsg_code=args.crs,
                output_format=args.ann_format,
                georef_info=georef,
                tile_prefix=tile_id,
            )
            total_ann_tiles += n
            if split == "train":
                train_ann_tiles += n
            else:
                val_ann_tiles += n
            bar.update()

    # ==================================================================
    # PATH C — Filename-encoded splits
    # ==================================================================
    else:
        unassigned = [p for p in pairs if p["split"] is None]
        if unassigned:
            ids = ", ".join(p["tile_id"] for p in unassigned[:5])
            logging.error(
                f"{len(unassigned)} tile(s) have no split token in their "
                f"filename (e.g. {ids}). Use --train_ratio to assign splits "
                f"automatically, or rename files to include a split token "
                f"(e.g. 6398_149_train_image.tif)."
            )
            return

        n_train = sum(1 for p in pairs if p["split"] == "train")
        n_val = sum(1 for p in pairs if p["split"] == "val")
        logging.info(f"Using filename-encoded splits: {n_train} train, {n_val} val.")

        bar = mmcv.ProgressBar(len(pairs))
        for pair in pairs:
            tile_id = pair["tile_id"]
            split = pair["split"]
            stride = args.stride_size if split == "train" else args.clip_size

            if args.copy_georef:
                georef_cache[tile_id] = store_georef_info(pair["image_path"])

            img_out = os.path.join(out_dir, "img_dir", split)
            n = clip_big_image(
                pair["image_path"],
                img_out,
                clip_size=args.clip_size,
                stride_size=stride,
                is_label=False,
                epsg_code=args.crs,
                output_format=args.img_format,
                bands=args.bands,
                tile_prefix=tile_id,
            )
            total_img_tiles += n
            if split == "train":
                train_img_tiles += n
            else:
                val_img_tiles += n

            georef = georef_cache.get(tile_id) if args.copy_georef else None
            ann_out = os.path.join(out_dir, "ann_dir", split)
            n = clip_big_image(
                pair["annotation_path"],
                ann_out,
                clip_size=args.clip_size,
                stride_size=stride,
                is_label=True,
                epsg_code=args.crs,
                output_format=args.ann_format,
                georef_info=georef,
                tile_prefix=tile_id,
            )
            total_ann_tiles += n
            if split == "train":
                train_ann_tiles += n
            else:
                val_ann_tiles += n
            bar.update()

    # --- Summary ---
    print_summary(
        args, pairs, use_spatial_split, use_random_split,
        train_img_tiles, train_ann_tiles,
        val_img_tiles, val_ann_tiles,
        total_img_tiles, total_ann_tiles,
    )


if __name__ == "__main__":
    main()