# Outline & walkthrough

[← back to README](../README.md)

This document is the detailed companion to the README. It contains:

1. **[Detailed outline](#detailed-outline)** — a chart of every stage.
2. **[Repository layout](#repository-layout)** — an annotated tree of the script
   and config files.
3. **[Walkthrough](#walkthrough)** — how to prepare data, set up, and run each
   runner on a custom dataset, with the decisions and gotchas that matter.

Upstream AerialFormer usage is preserved in
[README_AerialFormer.md](../README_AerialFormer.md).

## Detailed outline

```
INPUTS
  orthophoto ......... GeoTIFF / VRT, RGB, EPSG:3007 (SWEREF99 12 00)
  vector layers ...... water (e.g. <water>.gpkg) + tree crowns (e.g. <treecrowns>.gpkg)
        │
        ▼
┌────────────────────────────────────────────────────────────────────────────────┐
│ AF_0 → AF_3   edit each runner's config block; run from the repo root          │
│                                                                                │
│   0. labels .............. AF_0_training_data.py                               │
│        • burn each vector layer onto the ortho grid in draw order              │
│        • class_value = CLASSES idx + 1  →  bg=1, water=2, tree=3  (0 reserved) │
│        ↳ label raster (uint8, aligned to the template ortho)                   │
│                                                                                │
│   1. tile + split ........ AF_1_preprocess_runner.py → tools/AF_1_preprocess.py│
│        • tile image + label into 512×512 tiles, stride 256 train / 512 val     │
│        • split mode = spatial | random | filename                              │
│        ↳ mmseg layout: img_dir/{train,val}/  ann_dir/{train,val}/              │
│                                                                                │
│   2. train ............... AF_2_train_runner.py → tools/train.py   (mmseg 0.x) │
│        • AerialFormer = Swin encoder + multi-dilated CNN decoder, num_classes=3│
│        • reduce_zero_label: raw 1/2/3 → 0/1/2 at load; raw 0 → 255 (ignored)   │
│        ↳ checkpoint .pth + a snapshot of the resolved config                   │
│                                                                                │
│   3. inference ........... AF_3_inference_runner.py → tools/AF_3_inference.py  │
│        • sliding window 512 px, stride 384 (128 px overlap)                    │
│        • overlap mode: crop (fast, seams) | blend (softmax avg, seamless)      │
│        • windowed disk writes → large VRT mosaics in bounded RAM               │
└────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
OUTPUT
  <stem>_segmentation.tif ... single-band uint8, 0-based class indices
                              (0=bg, 1=water, 2=tree); 255=nodata

DOWNSTREAM
  Gbg-INST-TreeSeg FG_2 ..... consumes the raster as the canopy prior (tree = 2)
```

## Repository layout

```
Gbg-SEM-TreeSeg/                   ← run every runner from here (the aerialseg package root)
├── AF_0_training_data.py          rasterize vector layers → a semantic label raster
├── AF_1_preprocess_runner.py      config + invoke tools/AF_1_preprocess.py (tiling / split)
├── AF_2_train_runner.py           config + invoke tools/train.py (mmseg training)
├── AF_3_inference_runner.py       config + invoke tools/AF_3_inference.py (sliding-window inference)
├── setup.py                       installs the aerialseg package (pip install -e .)
├── LICENSE                        GNU AGPL-3.0
├── README.md                      overview, install, quickstart
├── README_AerialFormer.md         upstream AerialFormer usage documentation
├── tools/
│   ├── AF_1_preprocess.py         tiling engine: clip rasters into train/val tiles, 3 split modes
│   ├── AF_3_inference.py          sliding-window inference: crop/blend overlap, threaded windowed I/O
│   ├── train.py                   mmsegmentation 0.x training entry point
│   └── test.py                    mmsegmentation 0.x evaluation entry point
├── aerialseg/                     the model + dataset package (registered via pip install -e .)
│   ├── datasets/
│   │   └── gothenburg_tree_seg.py                GothenburgTreeSeg dataset class — CLASSES, PALETTE, reduce_zero_label
│   └── models/
│       ├── backbones/swin_stem.py                SwinStemTransformer encoder
│       └── decode_heads/aerialformer_head.py     MDCDecoder (multi-dilated CNN decoder)
├── configs/
│   ├── _base_/datasets/gbg_trees_seg.py          dataset pipeline, crop_size 512, normalization
│   ├── _base_/models/aerialformer.py             base model (num_classes placeholder, overridden)
│   ├── _base_/schedules/schedule.py              max_iters, poly lr, eval/checkpoint intervals
│   ├── _base_/default_runtime.py                 logging / runtime hooks
│   └── aerialformer/
│       ├── aerialformer_tiny_512x512_gbg.py      Swin-Tiny variant, num_classes=3, single-GPU friendly
│       └── aerialformer_base_512x512_gbg.py      Swin-Base variant, num_classes=3, SyncBN / multi-GPU
└── Project_1/                                    data + results, gitignored:
                                                  AF_0_training/ · AF_1_training_{src,data}/ ·
                                                  AF_2_training_runs/ · AF_3_inference_{src,results}/
```

## Walkthrough

### Before you start

Install the environment (developed and tested with Python 3.9, PyTorch 2.0.1,
mmcv-full 1.7.1, mmsegmentation **0.30.0**):

```bash
conda create -n gbg_sem_treeseg python=3.9
conda activate gbg_sem_treeseg
pip install torch==2.0.1 torchvision
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmsegmentation==0.30.0
pip install rasterio geopandas pyproj tifffile
pip install -e .                 # registers the aerialseg model + dataset classes
```

> **mmsegmentation 0.x only.** This repo targets the **0.x** branch (IterBased
> runner, `tools/train.py` API). It is **not** compatible with mmsegmentation
> 1.x. Match the `mmcv-full` wheel index to your torch/CUDA.

The class scheme is defined once in
`aerialseg/datasets/gothenburg_tree_seg.py`:

```python
CLASSES = ('bg', 'water', 'tree')                       # 3 classes, indices 0/1/2
PALETTE = [[255, 255, 255], [0, 0, 255], [0, 255, 0]]   # white / blue / green
# reduce_zero_label=True
```

> **The +1 / reduce_zero_label rule — the single most important gotcha.**
> AF_0 burns `class_value = (index in CLASSES) + 1`, so the **label raster** holds
> `bg=1, water=2, tree=3`, and pixel value `0` is reserved for "unlabeled".
> `reduce_zero_label=True` then shifts everything down by one at load time
> (`1/2/3 → 0/1/2`, raw `0 → 255 = ignored`). The **AF_3 output** is therefore
> 0-based (`bg=0, water=1, tree=2`) — that is the `tree = 2` the downstream
> instance pipeline expects.

### A. Create the label raster (AF_0)

Edit the `class_config` and `template_path` at the bottom of
`AF_0_training_data.py`. One entry per vector layer:

```python
class_config = [
    {"file_path": "Project_1/AF_0_training/<water>.gpkg", "class_value": 2, "draw_order": 1},           # water
    {"file_path": "Project_1/AF_0_training/<treecrowns>.gpkg",     "class_value": 3, "draw_order": 3},  # tree
]
template_path = "Project_1/AF_0_training/<ortho_template>.tif"   # defines the output grid
```

Decisions and notes:

- **`class_value` = CLASSES index + 1** (water=2, tree=3); `background=1` fills
  everything unburned. Never assign `0`.
- **`draw_order`** sets the overwrite hierarchy — higher draws later and wins.
  Trees (`draw_order 3`) overwrite water where crowns overhang.
- **CRS must match.** All vector layers must share the template raster's CRS
  (EPSG:3007); there is no on-the-fly reprojection.
- **Adding a class** (e.g. `building`): give it the next free `class_value` and a
  `draw_order`, append its name/colour to `CLASSES`/`PALETTE`, bump `num_classes`
  in `configs/aerialformer/aerialformer_tiny_512x512_gbg.py`, then re-run AF_0 +
  AF_1 and retrain.

```bash
python AF_0_training_data.py
```

### B. Tile and split (AF_1)

Lay out the full rasters as `image_dir/{id1}_{id2}_image.tif` and
`annotation_dir/{id1}_{id2}_annotation.tif` under a dataset root, point
`DATASET_PATHS` in `AF_1_preprocess_runner.py` at it, and set:

| Constant | Default | Decision |
|---|---|---|
| `CLIP_SIZE` | `512` | tile size in px — match the model crop size |
| `STRIDE_SIZE` | `256` | training-tile step (50 % overlap); validation always uses stride = `CLIP_SIZE` |
| `BANDS` | `"1,2,3"` | 1-based band indices to keep, or `"all"` |
| `EPSG_CODE` | `3007` | fallback CRS when a source lacks one |
| `INFERENCE` | `False` | `True` = tile images only (no labels, no split) |

**Split mode** is chosen by which fields you set (priority order):

1. **Spatial** — set both `TRAIN_RATIO` and `SPLIT_DIRECTION` (`"ns"`/`"sn"`/`"ew"`/`"we"`). Each raster is geographically cut into train/val halves before tiling → **zero spatial overlap** between splits (the most honest evaluation).
2. **Random** — set `TRAIN_RATIO`, leave `SPLIT_DIRECTION = None`. Whole files go to train or val at random.
3. **Filename** — set both to `None`. The split token in each filename (`VAL_SPLIT_NAMES`, e.g. `"test,val"`) decides.

```bash
python AF_1_preprocess_runner.py
```

Output is the mmseg `img_dir`/`ann_dir` layout plus a `processing.log` recording
the parameters used.

### C. Train (AF_2)

Edit `AF_2_train_runner.py`:

| Constant | Note |
|---|---|
| `CONFIG` | `aerialformer_tiny_…` (single-GPU friendly) or `aerialformer_base_…` (bigger, uses **SyncBN** — intended for multi-GPU; a single GPU works but SyncBN degrades to BN) |
| `DATA_ROOT` | points mmseg at the AF_1 output without editing the config files |
| `MAX_ITERS` | overrides the config default (160 000) |
| `LOAD_FROM` / `RESUME_FROM` / `AUTO_RESUME` | load weights only / resume optimizer+iter / auto-resume latest |
| `CFG_OPTIONS` | dict of dot-path overrides, e.g. `{"data.samples_per_gpu": 4}` |

The schedule saves the best checkpoint by `IoU.tree`. Checkpoints, logs, and **a
snapshot of the resolved config** land in a timestamped directory under
`BASE_WORK_DIR` — keep that snapshot, AF_3 needs it.

```bash
python AF_2_train_runner.py
```

### D. Run inference (AF_3)

Edit `AF_3_inference_runner.py`:

| Constant | Default | Decision |
|---|---|---|
| `CHECKPOINT` | — | a trained `.pth` |
| `CONFIG` | — | the **config snapshot saved next to the checkpoint** — this guarantees the architecture/`num_classes` match the weights |
| `INPUT` | — | a single GeoTIFF, a directory of GeoTIFFs, or a `.vrt` mosaic |
| `WINDOW_SIZE` | `512` | match the training crop size |
| `STRIDE` | `384` | overlap vs speed: `512`→no overlap (seams), `384`→128 px, `256`→256 px (best, slowest) |
| `OVERLAP_MODE` | `"blend"` | `crop` keeps each window centre (fast, possible seams); `blend` averages softmax across overlaps (seamless) |
| `BANDS` | `"1,2,3"` | must match what the model was trained on |
| `DEVICE` | `"cuda:0"` | inference device |
| `NODATA` | `255` | output nodata value |

Large mosaics: AF_3 writes the output tile-by-tile (windowed disk writes), and
`blend` mode keeps only a rolling buffer of `window_size` rows of probabilities,
so a city-scale `.vrt` is processed without allocating the full output.

```bash
python AF_3_inference_runner.py
```

The output is the georeferenced single-band segmentation raster
(`0=bg, 1=water, 2=tree, 255=nodata`). Keep it pixel-aligned with the orthophoto
it will be paired with — this is the canopy prior the
[Gbg-INST-TreeSeg](https://github.com/KitSimon/Gbg-INST-TreeSeg) FG_2 stage
consumes.
