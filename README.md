# Gbg-SEM-TreeSeg

**G**othen**b**ur**g** **SEM**antic segmentation — **Tree Seg**mentation.

Semantic segmentation of trees and water in aerial orthophotos over Gothenburg,
Sweden, using the [AerialFormer](https://github.com/UARK-AICV/AerialFormer)
architecture (Swin Transformer encoder + multi-dilated CNN decoder) built on
[mmsegmentation](https://github.com/open-mmlab/mmsegmentation) 0.x.

Classes: `bg` (background), `water`, `tree`. Input data is Lantmäteriet
orthophoto 2022 (RGB, 0.16 m) in SWEREF99 12 00 (EPSG:3007). The imagery and
training data are **not** included in this repository.

## Pipeline

The workflow is driven by four numbered script stages at the repository root.
Each stage has a `*_runner.py` with the configuration at the top — edit and run
the runner, not the underlying script.

| Stage | Script | Purpose |
|---|---|---|
| AF_0 | `AF_0_training_data.py` | Rasterize vector layers (GeoPackage/Shapefile) into a label raster matching an orthophoto grid |
| AF_1 | `AF_1_preprocess.py` (+ runner) | Tile large image/label rasters into 512×512 training tiles with spatial/random/filename-based train-val splitting |
| AF_2 | `AF_2_train_runner.py` | Train via mmsegmentation (`tools/train.py`) with the configs in `configs/aerialformer/*_gbg.py` |
| AF_3 | `AF_3_inference.py` (+ runner) | Sliding-window inference on arbitrarily large GeoTIFF/VRT mosaics with seamless overlap blending, writing georeferenced segmentation rasters |

Project-specific code added on top of upstream AerialFormer:

- `AF_0`–`AF_3` pipeline scripts and runners
- `aerialseg/datasets/gothenburg_tree_seg.py` (dataset class)
- `configs/_base_/datasets/gbg_trees_seg.py`,
  `configs/aerialformer/aerialformer_{tiny,base}_512x512_gbg.py`
- Assorted edits to base configs (logging, evaluation, schedule)

`_archive/` contains superseded development iterations and the upstream
Potsdam dataset support, kept for reference. Upstream usage documentation is
preserved in [README_AerialFormer.md](README_AerialFormer.md).

## Installation

Developed and tested with Python 3.9, PyTorch 2.0.1, mmcv-full 1.7.1 and
mmsegmentation 0.30.0 (the 0.x branch — **not** compatible with
mmsegmentation 1.x):

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

## Getting started on a custom dataset

1. **Create a label raster (AF_0).** Edit `class_config` and `template_path`
   at the bottom of `AF_0_training_data.py`: one entry per vector layer, with
   `class_value` = the class's index in `CLASSES` + 1 (pixel value 0 is
   reserved for "unlabeled" and must not be used — see the comments in the
   script). The template raster defines the output grid; vector layers must
   share its CRS.
2. **Define your classes.** Set `CLASSES`/`PALETTE` in
   `aerialseg/datasets/gothenburg_tree_seg.py` (or a copy registered in
   `aerialseg/datasets/__init__.py`) and `num_classes` in
   `configs/aerialformer/aerialformer_tiny_512x512_gbg.py`. All three must
   agree in length; the comments in those files walk through adding a class.
3. **Tile and split (AF_1).** Place rasters as
   `image_dir/{id1}_{id2}_image.tif` and
   `annotation_dir/{id1}_{id2}_annotation.tif` under a dataset root, point
   `DATASET_PATHS` in `AF_1_preprocess_runner.py` at it, choose tile/stride
   sizes and a split mode (spatial, random, or filename-encoded), and run the
   runner. Output is the mmsegmentation `img_dir`/`ann_dir` layout plus a
   `processing.log` recording the parameters used.
4. **Train (AF_2).** Set `CONFIG`, `DATA_ROOT`, and `MAX_ITERS` in
   `AF_2_train_runner.py` and run it. Checkpoints, logs, and a snapshot of the
   resolved config land in a timestamped directory under `BASE_WORK_DIR`.
5. **Run inference (AF_3).** In `AF_3_inference_runner.py`, point `CHECKPOINT`
   at a trained `.pth` and `CONFIG` at the config snapshot saved next to it
   (guarantees the architecture matches), set `INPUT` to a GeoTIFF, a
   directory of GeoTIFFs, or a VRT mosaic, and run. The output is a
   georeferenced single-band GeoTIFF of class indices (0-based, matching
   `CLASSES` order; 255 = nodata).

## Attribution

This repository is a derivative work of
**[AerialFormer](https://github.com/UARK-AICV/AerialFormer)** by Yamazaki et
al. (UARK-AICV). The model implementation (`aerialseg/`), base configs, and
tooling originate from that project, with modifications noted in the files and
summarized above. If you use this work, please cite their paper:

```bibtex
@article{yamazaki2023aerialformer,
  title={AerialFormer: Multi-resolution Transformer for Aerial Image Segmentation},
  author={Yamazaki, Kashu and Hanyu, Taisei and Tran, Minh and Garcia, Adrian and Tran, Anh and McCann, Roy and Liao, Haitao and Rainwater, Chase and Adkins, Meredith and Molthan, Andrew and others},
  journal={arXiv preprint arXiv:2306.06842},
  year={2023}
}
```

AerialFormer is in turn built on
**[mmsegmentation](https://github.com/open-mmlab/mmsegmentation)**
(OpenMMLab), licensed under Apache License 2.0. Files carrying the
`Copyright (c) OpenMMLab.` header derive from that project.

## License

Original code in this repository — the `AF_0`–`AF_3` pipeline scripts and
runners, the gbg dataset class and configs, and this documentation — is
licensed under the **GNU Affero General Public License v3.0** (see
[LICENSE](LICENSE)). In short: you may use, modify, and redistribute it, but
derivative works must be released under the same license, including when the
software is offered as a network service.

Inherited portions are **not** covered by that grant and retain their own
status:

- The upstream AerialFormer code (`aerialseg/` model implementation, base
  configs) is published without a license; copyright remains with its authors
  and it is reproduced here for research purposes with attribution.
- Files carrying the `Copyright (c) OpenMMLab.` header derive from
  mmsegmentation and remain under the Apache License 2.0.
