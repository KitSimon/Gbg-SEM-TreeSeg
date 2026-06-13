#!/usr/bin/env python
"""
Runner script for prepare_tiles.py

Edit the configuration below, then run:
    python run_prepare_tiles.py
"""

import subprocess
import sys

# ===================================================================
#  CONFIGURATION — edit these to match your setup
# ===================================================================

# -------------------------------------------------------------------
#  PATHS
# -------------------------------------------------------------------

# One or more dataset root directories.
# Each must contain 'image_dir/' and 'annotation_dir/' subdirectories.
DATASET_PATHS = [
    "Project_1/AF_1_training_src",
    # "/path/to/dataset_2",       # uncomment to add more
]

# Where tiled output will be written
OUT_DIR = "Project_1/AF_1_training_data"

# (Optional) Temporary directory for intermediate files.
# Set to None to use system default.
TMP_DIR = None

# Path to prepare_tiles.py (relative or absolute)
SCRIPT_PATH = "AF_1_preprocess.py"                          #"tools/convert_datasets/AF_1_preprocess.py"

# -------------------------------------------------------------------
#  TILING PARAMETERS
# -------------------------------------------------------------------

CLIP_SIZE = 512         # Tile size in pixels
STRIDE_SIZE = 256       # Sliding-window step for training tiles
                        # (validation always uses stride = clip_size)

# -------------------------------------------------------------------
#  GEOSPATIAL
# -------------------------------------------------------------------

EPSG_CODE = 3007        # Fallback CRS (SWEREF99 12 00)

# -------------------------------------------------------------------
#  OUTPUT FORMAT
# -------------------------------------------------------------------

IMG_FORMAT = "tif"      # "tif" (preserves georef) or "png"
ANN_FORMAT = "tif"      # "tif" or "png"
BANDS = "1,2,3"         # Comma-separated 1-based band indices, or "all"

# Copy georef from images to annotations
COPY_GEOREF = True

# -------------------------------------------------------------------
#  SPLIT CONFIGURATION
#
#  Three modes (highest priority first):
#
#  1) SPATIAL SPLIT  — set both TRAIN_RATIO and SPLIT_DIRECTION
#     Each raster is geographically cropped into train/val halves,
#     then tiled.  Zero spatial overlap between splits.
#
#  2) RANDOM SPLIT   — set TRAIN_RATIO, leave SPLIT_DIRECTION as None
#     Whole files are randomly assigned to train or val.
#
#  3) FILENAME SPLIT — set both to None
#     The split token in each filename (e.g. "train" / "test") is used.
# -------------------------------------------------------------------

# Which filename tokens count as validation (comma-separated)
VAL_SPLIT_NAMES = "test,val"

# Training fraction.  Set to None to use filename-encoded splits.
TRAIN_RATIO = 0.75424       # e.g. 0.8 for 80% train / 20% val, 0.83584012241775057383320581484315 för jämn rad
RANDOM_SEED = 42

# Spatial split direction (requires TRAIN_RATIO).
# Set to None for random split; or one of: "ns", "sn", "ew", "we"
#   ns = train is the North portion,  val is South
#   sn = train is the South portion,  val is North
#   ew = train is the East  portion,  val is West
#   we = train is the West  portion,  val is East
SPLIT_DIRECTION = "sn"

# -------------------------------------------------------------------
#  MODE
# -------------------------------------------------------------------

# Inference mode: tile images only (no annotations, no train/val split).
# For inference, point DATASET_PATHS at directories with raw .tif images.
INFERENCE = False


# ===================================================================
#  BUILD AND RUN — normally no edits needed below this line
# ===================================================================

def build_command():
    cmd = [sys.executable, SCRIPT_PATH]

    # Dataset paths (positional)
    cmd.extend(DATASET_PATHS)

    # Output
    cmd.extend(["-o", OUT_DIR])

    # Optional temp dir
    if TMP_DIR is not None:
        cmd.extend(["--tmp_dir", TMP_DIR])

    # Tiling
    cmd.extend(["--clip_size", str(CLIP_SIZE)])
    cmd.extend(["--stride_size", str(STRIDE_SIZE)])

    # Geospatial
    cmd.extend(["--crs", str(EPSG_CODE)])

    # Formats
    cmd.extend(["--img_format", IMG_FORMAT])
    cmd.extend(["--ann_format", ANN_FORMAT])
    cmd.extend(["--bands", BANDS])

    # Georef copy
    if COPY_GEOREF:
        cmd.append("--copy_georef")

    # Splits
    cmd.extend(["--val_split_names", VAL_SPLIT_NAMES])

    if TRAIN_RATIO is not None:
        cmd.extend(["--train_ratio", str(TRAIN_RATIO)])
        cmd.extend(["--random_seed", str(RANDOM_SEED)])

    if SPLIT_DIRECTION is not None:
        cmd.extend(["--split_direction", SPLIT_DIRECTION])

    # Inference
    if INFERENCE:
        cmd.append("--inference")

    return cmd


def main():
    cmd = build_command()

    print("=" * 50)
    print("  prepare_tiles.py")
    print("=" * 50)
    print("Command:")
    print(f"  {' '.join(cmd)}")
    print("-" * 50)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()