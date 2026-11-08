"""Deterministic spectral / colour indices and masks.

These are classical, explainable measurements. The LLM/VLM never computes areas or masks; it only
interprets the numbers these functions produce.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .raster import Raster

EPS = 1e-6


def _hsv(r: Raster) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rgb = r.rgb
    mx, mn = rgb.max(-1), rgb.min(-1)
    sat = (mx - mn) / (mx + EPS)
    return mx, sat, mx - mn


def ndvi(r: Raster) -> np.ndarray:
    if not r.has_nir:
        raise ValueError("NDVI needs a NIR band")
    red, nir = r.bands["red"], r.bands["nir"]
    return ((nir - red) / (nir + red + EPS)).astype(np.float32)


def greenness(r: Raster) -> np.ndarray:
    """NDVI when NIR exists, otherwise the visible-band normalised excess-green index."""
    if r.has_nir:
        return ndvi(r)
    red, g, b = r.bands["red"], r.bands["green"], r.bands["blue"]
    return ((2 * g - red - b) / (g + red + b + EPS)).astype(np.float32)


def vegetation_mask(r: Raster) -> np.ndarray:
    gi = greenness(r)
    return gi > (0.3 if r.has_nir else 0.12)


def water_index(r: Raster) -> np.ndarray:
    if r.has_nir:  # NDWI (McFeeters)
        g, nir = r.bands["green"], r.bands["nir"]
        return ((g - nir) / (g + nir + EPS)).astype(np.float32)
    red, b = r.bands["red"], r.bands["blue"]  # visible-only: blue dominance
    return ((b - red) / (b + red + EPS)).astype(np.float32)


def water_mask(r: Raster) -> np.ndarray:
    thr = 0.1 if r.has_nir else 0.07
    m = water_index(r) > thr
    if not r.has_nir:
        m &= ~vegetation_mask(r)
    return m


def burn_like_mask(r: Raster) -> np.ndarray:
    """Dark, desaturated surfaces (char / ash)."""
    mx, sat, _ = _hsv(r)
    return (mx < 0.27) & (sat < 0.2)


def fire_mask(r: Raster) -> np.ndarray:
    """Saturated orange/red emitters."""
    red, g, b = r.bands["red"], r.bands["green"], r.bands["blue"]
    return (red > 0.78) & (red - g > 0.25) & (g > b) & (red - b > 0.5)


def smoke_mask(r: Raster) -> np.ndarray:
    """Bright, low-saturation haze (only meaningful in wildfire scenes)."""
    mx, sat, _ = _hsv(r)
    lum = r.rgb.mean(-1)
    smooth = ndi.uniform_filter(lum, 9)
    return (mx > 0.55) & (sat < 0.1) & (smooth > 0.55)


def bare_soil_mask(r: Raster) -> np.ndarray:
    red, g, b = r.bands["red"], r.bands["green"], r.bands["blue"]
    mx, sat, _ = _hsv(r)
    return (red - b > 0.2) & (red - g < 0.15) & (g - b > 0.08) & (mx > 0.5) & (sat > 0.2) & ~vegetation_mask(r)


def local_std(arr: np.ndarray, size: int = 3) -> np.ndarray:
    m = ndi.uniform_filter(arr, size)
    m2 = ndi.uniform_filter(arr * arr, size)
    return np.sqrt(np.clip(m2 - m * m, 0, None))


def rubble_mask(r: Raster) -> np.ndarray:
    """High local texture on non-vegetated, non-water surfaces (debris / collapsed structures)."""
    lum = r.rgb.mean(-1)
    tex = ndi.uniform_filter(local_std(lum, 3), 3)
    return (tex > 0.055) & ~vegetation_mask(r) & ~water_mask(r)


def builtup_mask(r: Raster) -> np.ndarray:
    """Impervious / built surfaces = everything that is not vegetation, water or bare soil."""
    return ~(vegetation_mask(r) | water_mask(r) | bare_soil_mask(r))
