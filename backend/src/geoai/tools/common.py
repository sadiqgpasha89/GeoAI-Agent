"""Helpers shared by the domain tools."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString, Point, box
from shapely.geometry.base import BaseGeometry

from geoai.geo.indices import water_index
from geoai.geo.regions import clean_mask, fraction_inside
from geoai.geo.vector import VFeature
from geoai.tools.base import ToolContext


def where(bbox: list[float], w: int, h: int) -> str:
    """Human-readable position of a bounding box inside the image."""
    cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
    col = "left" if cx < 0.34 else "right" if cx > 0.66 else "centre"
    row = "upper" if cy < 0.34 else "lower" if cy > 0.66 else "middle"
    if col == "centre" and row == "middle":
        return "centre"
    if col == "centre":
        return f"{row} centre"
    if row == "middle":
        return f"{col} side"
    return f"{row}-{col}"


def in_scene(bbox: list[float], w: int, h: int) -> str:
    """Phrase like 'in the upper-left of the scene' / 'on the left side of the scene'."""
    p = where(bbox, w, h)
    if p == "centre":
        return "in the centre of the scene"
    if p.endswith("side"):
        return f"on the {p} of the scene"
    return f"in the {p} of the scene"


def plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


def vlm_features(ctx: ToolContext, query: str, prefix: str) -> list[VFeature]:
    """Ask the VLM for boxes (live mode only); results are cached per query."""
    key = f"vlm:{query}"

    def fetch() -> list[VFeature]:
        boxes = ctx.gateway.locate(ctx.latest.raster.to_png_bytes(), query, ctx.latest.raster.width, ctx.latest.raster.height)
        return [VFeature(id=f"{prefix}{i + 1:02d}", props={"confidence": b.confidence, "label": b.label},
                         geom=box(*b.bbox_px), source="vlm") for i, b in enumerate(boxes)]

    return ctx.cached(key, fetch)


def get_objects(ctx: ToolContext, layer: str, vlm_query: str, prefix: str) -> tuple[list[VFeature], str]:
    """Reference-layer features if available, else VLM proposals (live), else nothing."""
    if ctx.layers and ctx.layers.has(layer):
        return ctx.layers.get(layer), "layer"
    if ctx.gateway.mode == "live":
        return vlm_features(ctx, vlm_query, prefix), "vlm"
    return [], "none"


def water_masks(ctx: ToolContext):
    from geoai.geo.indices import water_mask

    now = ctx.cached("water_now", lambda: clean_mask(water_mask(ctx.latest.raster), 60))
    before = None
    if ctx.before is not None:
        before = ctx.cached("water_before", lambda: clean_mask(water_mask(ctx.before.raster), 60))
    return now, before


def water_confidence(ctx: ToolContext, mask: np.ndarray) -> float:
    if not mask.any():
        return 0.5
    thr = 0.1 if ctx.latest.raster.has_nir else 0.07
    margin = float(np.mean(water_index(ctx.latest.raster)[mask]) - thr)
    return float(np.clip(0.62 + 1.4 * margin, 0.55, 0.95))


def line_samples(line: LineString, step_px: float = 2.0) -> list[tuple[Point, tuple[float, float]]]:
    """Points along a line with unit normals."""
    out = []
    n = max(2, int(line.length / step_px))
    for i in range(n + 1):
        d = line.length * i / n
        p, q = line.interpolate(max(d - 1, 0)), line.interpolate(min(d + 1, line.length))
        dx, dy = q.x - p.x, q.y - p.y
        norm = float(np.hypot(dx, dy)) or 1.0
        out.append((line.interpolate(d), (-dy / norm, dx / norm)))
    return out


def runs_along(line: LineString, mask: np.ndarray, width_px: float, min_share: float = 0.6, step_px: float = 2.0):
    """Contiguous stretches of ``line`` where >= ``min_share`` of the cross-section is True in ``mask``.

    Returns (runs, total_samples, hit_samples) with runs = [(start_d, end_d)] in pixel distance.
    """
    h, w = mask.shape
    offsets = np.linspace(-0.4, 0.4, 5) * width_px
    samples = line_samples(line, step_px)
    flags = []
    for p, (nx, ny) in samples:
        hits = 0
        for o in offsets:
            x, y = int(round(p.x + nx * o)), int(round(p.y + ny * o))
            if 0 <= x < w and 0 <= y < h and mask[y, x]:
                hits += 1
        flags.append(hits / len(offsets) >= min_share)
    runs, start = [], None
    n = len(samples)
    for i, f in enumerate(flags):
        d = line.length * i / max(n - 1, 1)
        if f and start is None:
            start = d
        if not f and start is not None:
            runs.append((start, d))
            start = None
    if start is not None:
        runs.append((start, line.length))
    return runs, n, sum(flags)


def severity_rank(s: str | None) -> int:
    return {"critical": 4, "high": 3, "moderate": 2, "low": 1, "info": 0}.get(s or "", 0)


def frac_inside(mask: np.ndarray, geom: BaseGeometry) -> float:
    return fraction_inside(mask, geom)
