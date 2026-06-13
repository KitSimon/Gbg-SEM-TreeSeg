#!/usr/bin/env python
"""
Runner script for AerialFormer inference.

Edit the configuration below, then run:
    python run_inference.py
"""

import subprocess
import sys

# ===================================================================
#  CONFIGURATION — edit these to match your setup
# ===================================================================

# -------------------------------------------------------------------
#  PATHS
# -------------------------------------------------------------------

# Path to the inference script
SCRIPT_PATH = "AF_3_inference.py"

# mmseg config file (must match the architecture of the checkpoint)
CONFIG = "configs/aerialformer/aerialformer_base_512x512_gbg.py"

# Trained model checkpoint
CHECKPOINT = "Project_1/AF_2_training_runs/aerialformer_base_512x512_gbg/2026_0509_2011/iter_20000.pth"

# Input: a directory of .tif files, a single .tif file, or a .vrt mosaic
INPUT = "/mnt/f/LM_Ortofoto_2022_RGB/_mosaik.vrt"
#"Project_1/AF_3_inference_src/LM_Ortofoto_2022_RGB/_mosaik.vrt" 
#/mnt/f/LM_Ortofoto_2022_RGB/_mosaik.vrt 
#/mnt/f/LM_Ortofoto_2022_RGB/tif/0_16m/6397_144.tif
#/mnt/f/LM_Ortofoto_2022_RGB/tif/0_16m/6398_149.tif

# Output directory for segmentation maps
OUTPUT_DIR = "Project_1/AF_3_inference_results/segmentation1"

# -------------------------------------------------------------------
#  SLIDING WINDOW
# -------------------------------------------------------------------

# Window size in pixels (should match model's training crop size)
WINDOW_SIZE = 512

# Stride between windows.  Smaller = more overlap = fewer edge artifacts.
# Common choices relative to window size:
#   512  → no overlap (fastest, visible seams)
#   384  → 128px overlap (good balance)
#   256  → 256px overlap (best quality, slowest)
STRIDE = 384

# How to handle predictions in overlap zones:
#   "crop"  — keep only the center of each window (fast, possible seams)
#   "blend" — average softmax class probabilities across overlapping
#             windows (slower, seamless transitions)
OVERLAP_MODE = "blend"

# -------------------------------------------------------------------
#  BAND SELECTION
#
#  Which bands to read from the input raster, as 1-based indices.
#  Must match what the model was trained on.
#  Set to None to use all bands.
# -------------------------------------------------------------------

BANDS = "1,2,3"         # e.g. "1,2,3" for RGB, "1,2,3,4" for RGBI

# -------------------------------------------------------------------
#  GEOSPATIAL
# -------------------------------------------------------------------

# Fallback EPSG code when source CRS is missing
EPSG_CODE = 3007        # SWEREF99 12 00

# -------------------------------------------------------------------
#  PROCESSING
# -------------------------------------------------------------------

# Device for inference
DEVICE = "cuda:0"

# Queue buffer size (number of windows buffered between threads)
BATCH_QUEUE_SIZE = 4

# Nodata value for unprocessed pixels in the output raster
NODATA = 255


# ===================================================================
#  BUILD AND RUN — normally no edits needed below this line
# ===================================================================

def build_command():
    cmd = [sys.executable, SCRIPT_PATH]

    cmd.extend(["--config", CONFIG])
    cmd.extend(["--checkpoint", CHECKPOINT])
    cmd.extend(["--input", INPUT])
    cmd.extend(["--output_dir", OUTPUT_DIR])

    cmd.extend(["--window_size", str(WINDOW_SIZE)])
    cmd.extend(["--stride", str(STRIDE)])
    cmd.extend(["--overlap_mode", OVERLAP_MODE])

    if BANDS is not None:
        cmd.extend(["--bands", BANDS])

    cmd.extend(["--crs", str(EPSG_CODE)])
    cmd.extend(["--device", DEVICE])
    cmd.extend(["--batch_queue_size", str(BATCH_QUEUE_SIZE)])
    cmd.extend(["--nodata", str(NODATA)])

    return cmd


def main():
    cmd = build_command()

    overlap = WINDOW_SIZE - STRIDE
    overlap_pct = (overlap / WINDOW_SIZE) * 100 if WINDOW_SIZE > 0 else 0

    print("=" * 60)
    print("  AerialFormer Inference")
    print("=" * 60)
    print(f"  Config:               {CONFIG}")
    print(f"  Checkpoint:           {CHECKPOINT}")
    print(f"  Input:                {INPUT}")
    print(f"  Output directory:     {OUTPUT_DIR}")
    print()
    print(f"  Window size:          {WINDOW_SIZE} px")
    print(f"  Stride:               {STRIDE} px  ({overlap}px overlap, {overlap_pct:.0f}%)")
    print(f"  Overlap mode:         {OVERLAP_MODE}")
    print(f"  Bands:                {BANDS if BANDS else 'all'}")
    print(f"  Device:               {DEVICE}")
    print(f"  Fallback CRS:         EPSG:{EPSG_CODE}")
    print(f"  Nodata value:         {NODATA}")
    print()
    print("-" * 60)
    print(f"  {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()