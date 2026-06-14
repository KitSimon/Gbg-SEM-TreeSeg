# Gbg-SEM-TreeSeg — Tree Semantic Segmentation

**G**othen**b**ur**g** **SEM**antic segmentation — **Tree Seg**mentation.

Semantic segmentation of trees and water in aerial orthophotos over Gothenburg,
Sweden, using the [AerialFormer](https://github.com/UARK-AICV/AerialFormer)
architecture (Swin Transformer encoder + multi-dilated CNN decoder) built on
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 0.x.

Classes: `bg` (background, 0), `water` (1), `tree` (2); 255 = nodata. Input data
is the Lantmäteriet orthophoto 2022 (RGB, 0.16 m) in SWEREF99 12 00 (EPSG:3007).
The imagery and training data are **not** included in this repository.

## Installation

Developed and tested with Python 3.9, PyTorch 2.0.1, mmcv-full 1.7.1 and
mmsegmentation 0.30.0 (the 0.x branch — **not** compatible with mmsegmentation
1.x):

```bash
conda create -n aerialformer python=3.9
conda activate aerialformer

# PyTorch — pick the CUDA build matching your driver, see pytorch.org
pip install torch==2.0.1 torchvision

# OpenMMLab stack (mmcv-full needs the wheel index matching your torch/CUDA;
# see https://mmcv.readthedocs.io/en/v1.7.1/get_started/installation.html)
pip install mmcv-full==1.7.1 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.0/index.html
pip install mmsegmentation==0.30.0

# Geospatial stack
pip install rasterio geopandas pyproj tifffile

# Register the aerialseg package (model + dataset classes) in editable mode
pip install -e .
```

## Pipeline

```
 Orthophoto ─► AF_0 labels ─► AF_1 tiles ─► AF_2 train ─► AF_3 inference ─► segmentation.tif
   (VRT/TIF)                                                                 (uint8 class indices)
```

The workflow is driven by four numbered stages. Each `*_runner.py` script at the
repository root holds its configuration at the top and invokes the underlying
scripts in `tools/` — edit and run the runners from the repository root, not the
underlying scripts.

| Stage | Script | Purpose |
|---|---|---|
| AF_0 | `AF_0_training_data.py` | Rasterize vector layers (GeoPackage/Shapefile) into a label raster matching an orthophoto grid |
| AF_1 | `AF_1_preprocess_runner.py` | Tile large image/label rasters into 512×512 training tiles with spatial / random / filename-based train-val split |
| AF_2 | `AF_2_train_runner.py` | Train via mmsegmentation with the configs in `configs/aerialformer/*_gbg.py` |
| AF_3 | `AF_3_inference_runner.py` | Sliding-window inference on arbitrarily large GeoTIFF/VRT mosaics with seamless overlap blending → georeferenced segmentation rasters |

Project-specific code added on top of upstream AerialFormer (the pipeline scripts,
the `gothenburg_tree_seg` dataset class, and the `gbg` configs) is summarized in
[docs/walkthrough.md](docs/walkthrough.md). `_archive/` holds superseded iterations and the
upstream Potsdam support, kept for reference; upstream usage docs are preserved in
[README_AerialFormer.md](README_AerialFormer.md).

## Quickstart

Run each stage's runner from the repository root, editing the constants at the top
first. On a custom dataset, work through AF_0 → AF_3 in order; the full walkthrough
(class definitions, tiling/split modes, per-stage knobs) is in
[docs/walkthrough.md](docs/walkthrough.md).

```bash
# AF_0 — rasterize vector layers into a label raster on the orthophoto grid
python AF_0_training_data.py

# AF_1 — tile image+label rasters into 512×512 train/val tiles
python AF_1_preprocess_runner.py

# AF_2 — train AerialFormer (checkpoints + config snapshot land under BASE_WORK_DIR)
python AF_2_train_runner.py

# AF_3 — sliding-window inference on a GeoTIFF, a directory of GeoTIFFs, or a VRT
python AF_3_inference_runner.py
```

AF_3 writes a georeferenced single-band GeoTIFF of class indices (0-based,
matching `CLASSES` order; 255 = nodata) — this is the semantic raster the
downstream instance pipeline consumes.

## Documentation

- [docs/walkthrough.md](docs/walkthrough.md) — detailed pipeline diagram,
  annotated repository layout, and a stage-by-stage (AF_0 → AF_3) walkthrough of
  data prep, setup, and running each runner, with the per-stage configuration
  decisions and gotchas.
- [README_AerialFormer.md](README_AerialFormer.md) — upstream AerialFormer usage
  documentation.

## Related repositories

This repo is the **semantic-segmentation stage** of a two-stage tree-crown
workflow. Its downstream sibling,
**[Gbg-INST-TreeSeg](https://github.com/KitSimon/Gbg-INST-TreeSeg)**, turns these
outputs into individual tree-crown polygons with Cellpose-SAM. Two artifacts cross
the boundary:

| Gbg-SEM-TreeSeg artifact | Used by Gbg-INST-TreeSeg | As |
|---|---|---|
| Per-tree crown polygons (digitised training labels) | FG_0 | Cellpose training labels |
| AF_3 `<stem>_segmentation.tif` | FG_2 | canopy prior (tree class = 2) |

## Attribution

This repository is a derivative work of
**[AerialFormer](https://github.com/UARK-AICV/AerialFormer)** by Yamazaki et al.
(UARK-AICV). The model implementation (`aerialseg/`), base configs, and tooling
originate from that project, with modifications noted in the files and summarized
in [docs/walkthrough.md](docs/walkthrough.md). If you use this work, please cite
both this repository (see [CITATION.cff](CITATION.cff)) and the upstream
AerialFormer paper:

```bibtex
@article{yamazaki2023aerialformer,
  title={AerialFormer: Multi-resolution Transformer for Aerial Image Segmentation},
  author={Yamazaki, Kashu and Hanyu, Taisei and Tran, Minh and Garcia, Adrian and Tran, Anh and McCann, Roy and Liao, Haitao and Rainwater, Chase and Adkins, Meredith and Molthan, Andrew and others},
  journal={arXiv preprint arXiv:2306.06842},
  year={2023}
}
```

AerialFormer is in turn built on
**[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)** (OpenMMLab),
licensed under Apache License 2.0. Files carrying the `Copyright (c) OpenMMLab.`
header derive from that project.

## License

Original code in this repository — the `AF_0`–`AF_3` pipeline scripts and runners,
the gbg dataset class and configs, and this documentation — is licensed under the
**GNU Affero General Public License v3.0** (see [LICENSE](LICENSE)). In short: you
may use, modify, and redistribute it, but derivative works must be released under
the same license, including when the software is offered as a network service.

Inherited portions are **not** covered by that grant and retain their own status:

- The upstream AerialFormer code (`aerialseg/` model implementation, base configs)
  is published without a license; copyright remains with its authors and it is
  reproduced here for research purposes with attribution.
- Files carrying the `Copyright (c) OpenMMLab.` header derive from mmsegmentation
  and remain under the Apache License 2.0.
