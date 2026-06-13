"""
Rasterize vector layers into a semantic segmentation label raster.

Reads one or more vector files (e.g. .shp, .gpkg), assigns each a class
value, and burns them into a raster whose grid matches a given template
(e.g. an orthophoto). Layers are drawn in a user-specified order so that
higher-priority classes overwrite lower ones where they overlap.

Dependencies:
    pip install rasterio geopandas numpy

Usage:
    1. Edit the ``class_config`` list and ``template_path`` at the bottom.
    2. Run:  python AF_0_training_data.py
"""

from __future__ import annotations

import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def check_files(file_paths: list[str]) -> None:
    """Raise FileNotFoundError if any path does not exist."""
    for path in file_paths:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"File does not exist: {path}")


def remove_invalid_geometries(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop rows whose geometry is None or invalid (e.g. NaN coordinates)."""
    before = len(gdf)
    gdf = gdf[gdf.geometry.notna()]
    gdf = gdf[gdf.is_valid]
    after = len(gdf)
    if before != after:
        print(f"  Removed {before - after} invalid geometries out of {before}")
    return gdf


# ---------------------------------------------------------------------------
# Core rasterization
# ---------------------------------------------------------------------------

def rasterize_semantic(
    class_config: list[dict],
    template_path: str,
    output_path: str | None = None,
    background: int = 0,
    dtype: str = "uint8",
) -> np.ndarray:
    """
    Burn vector layers into a raster that matches *template_path*.

    Parameters
    ----------
    class_config : list of dict
        Each dict must contain:
            - ``file_path``   : str – path to a vector file (.shp, .gpkg, …)
            - ``class_value`` : int – pixel value for this class
            - ``draw_order``  : int – layers are drawn low → high; higher
              values overwrite lower ones where they overlap.
    template_path : str
        Path to a raster whose extent, resolution and CRS define the
        output grid (e.g. an orthophoto GeoTIFF).
    output_path : str, optional
        If given, the label raster is written here as a single-band GeoTIFF.
    background : int
        Value assigned to pixels that are not covered by any vector layer.
    dtype : str
        NumPy-compatible data-type string for the output raster
        (default ``"uint8"`` – suitable for up to 255 classes).

    Returns
    -------
    label : numpy.ndarray
        2-D array with shape (height, width) containing class labels.
    """

    # --- Validate inputs ---------------------------------------------------
    file_paths = [entry["file_path"] for entry in class_config]
    check_files([template_path] + file_paths)

    # --- Read template metadata --------------------------------------------
    with rasterio.open(template_path) as src:
        profile = src.profile.copy()
        transform = src.transform
        height = src.height
        width = src.width

    # --- Initialise output with background value --------------------------
    label = np.full((height, width), fill_value=background, dtype=dtype)

    # --- Sort by draw_order ------------------------------------------------
    sorted_config = sorted(class_config, key=lambda x: x["draw_order"])

    # --- Burn each layer ----------------------------------------------------
    for entry in sorted_config:
        file_path = entry["file_path"]
        class_value = entry["class_value"]
        print(f"Processing class {class_value}: {os.path.basename(file_path)}")

        # Read vector data
        # NOTE: CRS reprojection is skipped because of a PROJ database
        # conflict in this conda environment.  All input layers are assumed
        # to share the same CRS as the template raster (SWEREF99 12 00).
        gdf = gpd.read_file(file_path)

        # Clean geometries
        gdf = remove_invalid_geometries(gdf)
        if gdf.empty:
            print("  No valid geometries – skipping layer.")
            continue

        # Build (geometry, value) pairs for rasterio.features.rasterize
        shapes = [(geom, class_value) for geom in gdf.geometry]

        # Burn into a temporary array
        burned = rasterize(
            shapes,
            out_shape=(height, width),
            transform=transform,
            fill=0,            # 0 = "no data" in the temp layer
            all_touched=True,  # matches R's touches = TRUE
            dtype=dtype,
        )

        # Overwrite label pixels where the current layer has values
        mask = burned != 0
        label[mask] = burned[mask]

    # --- Write output -------------------------------------------------------
    if output_path is not None:
        out_profile = profile.copy()
        out_profile.update(
            count=1,
            dtype=dtype,
            compress="lzw",
            driver="GTiff",
        )
        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.write(label, 1)
        print(f"Label raster saved to: {output_path}")

    return label


# ---------------------------------------------------------------------------
# Example usage – edit paths and class definitions to suit your project
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    # Each entry: vector file, the class value to assign, and the draw order.
    # Higher draw_order layers overwrite lower ones where they overlap.
    #
    # Class values must stay in sync with the rest of the pipeline:
    #   class_value = index in CLASSES + 1, where CLASSES is defined in
    #   aerialseg/datasets/gothenburg_tree_seg.py (currently bg, water, tree
    #   → bg=1 via `background` below, water=2, tree=3).
    #   The +1 offset exists because reduce_zero_label=True shifts all labels
    #   down by one at training time; 0 is reserved for "unlabeled"/ignore
    #   and must never be assigned to a real class.
    #
    # To add a class (e.g. buildings, see commented template below):
    #   1. Add an entry here with the next free class_value and a draw_order
    #      placing it correctly in the overwrite hierarchy.
    #   2. Append the class name to CLASSES and a color to PALETTE in
    #      aerialseg/datasets/gothenburg_tree_seg.py.
    #   3. Bump num_classes in configs/aerialformer/aerialformer_tiny_512x512_gbg.py.
    #   4. Re-run AF_0 + AF_1 to regenerate labels/tiles, then retrain (AF_2).
    #
    # Suggested order (from the R script comments):
    #   Water → Impervious → Low vegetation → Buildings → Trees
    class_config = [
        {
            "file_path": r"Project_1/AF_0_training/vattendrag.gpkg",
            "class_value": 2,
            "draw_order": 1,
        },
        # Template for adding a class (here: buildings as class 4).
        # Trees (draw_order 3) would still overwrite buildings where crowns
        # overhang roofs.
        #{
        #    "file_path": r"Project_1/AF_0_training/byggnader_krontäckning.gpkg",
        #    "class_value": 4,
        #    "draw_order": 2,
        #},
        {
            "file_path": r"Project_1/AF_0_training/hull30.gpkg",
            "class_value": 3,
            "draw_order": 3,
        },
    ]

    template_path = r"Project_1/AF_0_training/6398_149.tif"

    output_name = os.path.splitext(os.path.basename(template_path))[0]
    output_path = os.path.join(
        r"Project_1/AF_0_training",
        f"gbg_trees_{output_name}_label_2.tif",
    )

    label = rasterize_semantic(
        class_config=class_config,
        template_path=template_path,
        output_path=output_path,
        background=1,   # matches Script 1's default
        dtype="uint8",
    )

    print(f"Done – unique class values: {np.unique(label)}")