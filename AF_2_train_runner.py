#!/usr/bin/env python
"""
Runner script for the AerialFormer / mmsegmentation training script.

Edit the configuration below, then run:
    python AF_2_train_runner.py
"""

import os, datetime
import subprocess
import sys


# ===================================================================
#  CONFIGURATION — edit these to match your setup
# ===================================================================

# -------------------------------------------------------------------
#  REQUIRED
# -------------------------------------------------------------------

# Path to the mmseg config file defining model, dataset, schedule, etc.
CONFIG = "configs/aerialformer/aerialformer_base_512x512_gbg.py"

# Path to the training script
SCRIPT_PATH = "tools/train.py"

BASE_WORK_DIR = "Project_1/AF_2_training_runs"  # your custom root

MAX_ITERS = 150000  # Set to None to use the config default (160000)

# -------------------------------------------------------------------
#  DATASET
#
#  These override the corresponding values in the dataset config
#  (e.g. gbg_trees_seg.py / GothenburgTreeSeg class) without needing
#  to edit those files.  Set any value to None to keep the config default.
# -------------------------------------------------------------------

# Root directory containing img_dir/ and ann_dir/ (the output of
# prepare_tiles.py).  Overrides data_root in the dataset config.
DATA_ROOT = "Project_1/AF_1_training_data"

# File extension for image tiles (e.g. ".tif" or ".png").
# Overrides img_suffix in the dataset class.
IMG_SUFFIX = ".tif"

# File extension for annotation tiles (e.g. ".tif" or ".png").
# Overrides seg_map_suffix in the dataset class.
SEG_MAP_SUFFIX = ".tif"


# -------------------------------------------------------------------
#  PATHS
# -------------------------------------------------------------------

# Working directory for logs and checkpoints.
# Set to None to use the config default (work_dirs/{config_name}/{timestamp}).
config_name = os.path.splitext(os.path.basename(CONFIG))[0]
timestamp = datetime.datetime.now().strftime("%Y_%m%d_%H%M")
WORK_DIR = os.path.join(BASE_WORK_DIR, config_name, timestamp)
#WORK_DIR = None

# Load pretrained weights from a checkpoint (no training state resumed).
# Set to None to train from scratch / use config-defined init.
LOAD_FROM = None

# Resume training from a checkpoint (restores optimizer, epoch, etc.).
# Set to None to start fresh.  See also AUTO_RESUME below.
RESUME_FROM = None

# Automatically resume from the latest checkpoint in WORK_DIR.
AUTO_RESUME = False



# -------------------------------------------------------------------
#  GPU SETTINGS
# -------------------------------------------------------------------

# Single GPU id to use (only for non-distributed training).
GPU_ID = 0

# Distributed launcher: "none", "pytorch", "slurm", or "mpi"
LAUNCHER = "none"

# -------------------------------------------------------------------
#  VALIDATION
# -------------------------------------------------------------------

# Set to True to skip validation during training.
NO_VALIDATE = False

# -------------------------------------------------------------------
#  REPRODUCIBILITY
# -------------------------------------------------------------------

# Random seed.  Set to None for auto-generated seed.
SEED = None

# Use a different seed per GPU rank in distributed training.
DIFF_SEED = False

# Force deterministic CUDNN operations (slower but reproducible).
DETERMINISTIC = False

# -------------------------------------------------------------------
#  CONFIG OVERRIDES
#
#  Override any config value without editing the config file.
#  Use a dict of key=value pairs.  Nested keys use dot notation.
#  Examples:
#      {"data.samples_per_gpu": 4, "optimizer.lr": 0.001}
#      {"model.backbone.depth": 50}
#  Set to None or {} for no overrides.
#
#  Note: DATA_ROOT, IMG_SUFFIX, and SEG_MAP_SUFFIX above are
#  automatically injected into these overrides — no need to
#  duplicate them here.
# -------------------------------------------------------------------

CFG_OPTIONS = None
# CFG_OPTIONS = {
#     "data.samples_per_gpu": 4,
#     "runner.max_iters": 80000,
#     "optimizer.lr": 0.001,
# }


# ===================================================================
#  BUILD AND RUN — normally no edits needed below this line
# ===================================================================

def build_cfg_overrides():
    """Merge the dataset convenience variables into CFG_OPTIONS."""
    overrides = dict(CFG_OPTIONS) if CFG_OPTIONS else {}

    # data_root must be set on each split (train, val, test)
    if DATA_ROOT is not None:
        for split in ("train", "val", "test"):
            overrides[f"data.{split}.data_root"] = DATA_ROOT

    # img_suffix and seg_map_suffix are passed through the dataset class
    if IMG_SUFFIX is not None:
        for split in ("train", "val", "test"):
            overrides[f"data.{split}.img_suffix"] = IMG_SUFFIX

    if SEG_MAP_SUFFIX is not None:
        for split in ("train", "val", "test"):
            overrides[f"data.{split}.seg_map_suffix"] = SEG_MAP_SUFFIX
    
    if MAX_ITERS is not None:
        overrides["runner.max_iters"] = MAX_ITERS

    return overrides


def build_command():
    cmd = [sys.executable, SCRIPT_PATH, CONFIG]

    if WORK_DIR is not None:
        cmd.extend(["--work-dir", WORK_DIR])

    if LOAD_FROM is not None:
        cmd.extend(["--load-from", LOAD_FROM])

    if RESUME_FROM is not None:
        cmd.extend(["--resume-from", RESUME_FROM])

    if AUTO_RESUME:
        cmd.append("--auto-resume")

    cmd.extend(["--gpu-id", str(GPU_ID)])

    cmd.extend(["--launcher", LAUNCHER])

    if NO_VALIDATE:
        cmd.append("--no-validate")

    if SEED is not None:
        cmd.extend(["--seed", str(SEED)])

    if DIFF_SEED:
        cmd.append("--diff_seed")

    if DETERMINISTIC:
        cmd.append("--deterministic")

    overrides = build_cfg_overrides()
    if overrides:
        cmd.append("--cfg-options")
        for key, value in overrides.items():
            cmd.append(f"{key}={value}")

    return cmd


def main():
    cmd = build_command()

    print("=" * 60)
    print("  AerialFormer Training")
    print("=" * 60)
    print(f"  Config file:          {CONFIG}")
    print(f"  GPU:                  {GPU_ID}  (launcher: {LAUNCHER})")
    if WORK_DIR:
        print(f"  Work directory:       {WORK_DIR}")
    if LOAD_FROM:
        print(f"  Load weights from:    {LOAD_FROM}")
    if RESUME_FROM:
        print(f"  Resume from:          {RESUME_FROM}")
    if AUTO_RESUME:
        print(f"  Auto-resume:          enabled")
    if SEED is not None:
        print(f"  Seed:                 {SEED}"
              f"{'  (per-rank)' if DIFF_SEED else ''}"
              f"{'  (deterministic)' if DETERMINISTIC else ''}")

    # Dataset
    if DATA_ROOT or IMG_SUFFIX or SEG_MAP_SUFFIX:
        print()
        print("  Dataset overrides:")
        if DATA_ROOT:
            print(f"    Data root:          {DATA_ROOT}")
        if IMG_SUFFIX:
            print(f"    Image suffix:       {IMG_SUFFIX}")
        if SEG_MAP_SUFFIX:
            print(f"    Annotation suffix:  {SEG_MAP_SUFFIX}")

    # Extra overrides
    if CFG_OPTIONS:
        print()
        print("  Config overrides:")
        for k, v in CFG_OPTIONS.items():
            print(f"    {k} = {v}")

    print()
    print("-" * 60)
    print(f"  {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()