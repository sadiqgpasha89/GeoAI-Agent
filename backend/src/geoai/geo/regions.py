"""Mask clean-up and polygonisation (pixel space)."""

from __future__ import annotations

import numpy as np
from rasterio import features
from scipy import ndimage as ndi
from shapely.geometry import Polygon, shape
from shapely.geometry.base import BaseGeometry


def clean_mask(mask: np.ndarray, min_pixels: int = 40, close: int = 2) -> np.ndarray:
    """Morphological closing + removal of tiny components."""
    m = mask.astype(bool)
    if close:
        m = ndi.binary_closing(m, structure=np.ones((3, 3)), iterations=close)
    m = ndi.binary_opening(m, structure=np.ones((3, 3)))
    labels, n = ndi.label(m)
    if n == 0:
        return m
    sizes = ndi.sum(m, labels, index=np.arange(1, n + 1))
    keep = np.isin(labels, np.arange(1, n + 1)[sizes >= min_pixels])
    return keep


def mask_to_polygons(mask: np.ndarray, min_pixels: int = 40, simplify_px: float = 1.5) -> list[Polygon]:
    """Vectorise a boolean mask into pixel-space polygons (largest first)."""
    out: list[Polygon] = []
    for geom, val in features.shapes(mask.astype(np.uint8), mask=mask.astype(bool)):
        if val != 1:
            continue
        poly = shape(geom)
        if poly.area < min_pixels:
            continue
        poly = poly.simplify(simplify_px, preserve_topology=True)
        if poly.is_empty:
            continue
        out.extend(list(poly.geoms) if poly.geom_type == "MultiPolygon" else [poly])
    out.sort(key=lambda p: p.area, reverse=True)
    return out


def rasterize(geoms: list[BaseGeometry], shape_hw: tuple[int, int]) -> np.ndarray:
    if not geoms:
        return np.zeros(shape_hw, dtype=bool)
    return features.rasterize(
        [(g, 1) for g in geoms if not g.is_empty], out_shape=shape_hw, fill=0, dtype="uint8", all_touched=False
    ).astype(bool)


def fraction_inside(mask: np.ndarray, geom: BaseGeometry) -> float:
    """Fraction of a geometry's pixels that are True in ``mask``."""
    region = rasterize([geom], mask.shape)
    n = int(region.sum())
    return float(mask[region].sum() / n) if n else 0.0
