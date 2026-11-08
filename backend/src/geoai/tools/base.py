"""Tool framework: typed arguments, structured outputs and shared geometry helpers."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry

from geoai.config import Settings
from geoai.geo.georef import GeoRef
from geoai.geo.vector import VectorLayers
from geoai.schemas import Domain, Evidence, Kind, Measurement, Observation, Severity
from geoai.storage.assets import AssetBundle


class Finding(BaseModel):
    """A deterministic, human-readable statement backed by measurements."""

    text: str
    topics: list[str] = Field(default_factory=list)
    kind: Kind = "observed"


class ToolOutput(BaseModel):
    observations: list[Observation] = Field(default_factory=list)
    metrics: dict[str, Measurement] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    review_reasons: list[str] = Field(default_factory=list)
    next_actions: list[str] = Field(default_factory=list)
    text: str | None = None  # free text from a VLM tool
    confidence: float | None = None

    def merge(self, other: ToolOutput) -> None:
        self.observations += other.observations
        self.metrics.update(other.metrics)
        self.findings += other.findings
        for name in ("assumptions", "limitations", "review_reasons", "next_actions"):
            for item in getattr(other, name):
                if item not in getattr(self, name):
                    getattr(self, name).append(item)


class ToolError(RuntimeError):
    """A recoverable tool failure; surfaced to the user as a limitation instead of crashing the run."""


@dataclass
class ToolContext:
    latest: AssetBundle
    before: AssetBundle | None
    gateway: Any  # llm.gateway.ModelGateway
    settings: Settings
    question: str
    domain: Domain
    memo: dict[str, Any] = field(default_factory=dict)
    _ids: dict[str, int] = field(default_factory=dict)

    @property
    def georef(self) -> GeoRef | None:
        return self.latest.raster.georef

    @property
    def layers(self) -> VectorLayers | None:
        return self.latest.layers

    @property
    def meta(self) -> dict:
        return self.latest.scene_meta or {}

    @property
    def shape(self) -> tuple[int, int]:
        return self.latest.raster.height, self.latest.raster.width

    def cached(self, key: str, fn: Callable[[], Any]) -> Any:
        if key not in self.memo:
            self.memo[key] = fn()
        return self.memo[key]

    def next_id(self, prefix: str) -> str:
        self._ids[prefix] = self._ids.get(prefix, 0) + 1
        return f"{prefix}-{self._ids[prefix]:02d}"

    # ---- unit helpers --------------------------------------------------
    def px_to_m2(self, px: float) -> float | None:
        a = self.latest.raster.pixel_area_m2
        return None if a is None else px * a

    def px_to_m(self, px: float) -> float | None:
        s = self.latest.raster.gsd_m
        return None if s is None else px * s

    def m_to_px(self, m: float) -> float | None:
        s = self.latest.raster.gsd_m
        return None if not s else m / s

    def area_metric(self, px: float, method: str) -> Measurement:
        if self.georef and self.latest.raster.pixel_area_m2:
            return Measurement(value=round(px * self.latest.raster.pixel_area_m2 / 10_000, 3), unit="ha", method=method)
        return Measurement(value=int(round(px)), unit="px²", method=method)

    def length_metric(self, px: float, method: str) -> Measurement:
        m = self.px_to_m(px)
        return Measurement(value=round(m, 1), unit="m", method=method) if m is not None else Measurement(value=round(px, 1), unit="px", method=method)

    def not_georeferenced_note(self) -> str:
        return "The image is not georeferenced: measurements are in pixels and no map coordinates are available."


def clamp_box(b: list[float], w: int, h: int) -> list[float]:
    x0, y0, x1, y1 = b
    x0, x1 = sorted((min(max(x0, 0), w), min(max(x1, 0), w)))
    y0, y1 = sorted((min(max(y0, 0), h), min(max(y1, 0), h)))
    return [round(x0, 1), round(y0, 1), round(x1, 1), round(y1, 1)]


def make_obs(
    ctx: ToolContext, tool: str, prefix: str, label: str, geom: BaseGeometry, confidence: float, *,
    kind: Kind = "observed", severity: Severity | None = None, attributes: dict | None = None, note: str | None = None,
    asset: str = "latest",
) -> Observation:
    h, w = ctx.shape
    geom = geom.intersection(box(0, 0, w, h)) if not geom.is_empty else geom
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        parts = [g for g in getattr(geom, "geoms", []) if g.geom_type == "Polygon"]
        geom = max(parts, key=lambda g: g.area) if parts else geom
    x0, y0, x1, y1 = clamp_box(list(geom.bounds), w, h) if not geom.is_empty else (0, 0, 0, 0)
    poly = None
    gj = None
    area_px = None
    area_m2 = None
    if geom.geom_type == "Polygon" and not geom.is_empty:
        g = geom.simplify(0.75) if len(geom.exterior.coords) > 160 else geom
        poly = [[round(x, 1), round(y, 1)] for x, y in g.exterior.coords]
        area_px = float(geom.area)
        area_m2 = ctx.px_to_m2(area_px)
        if ctx.georef:
            gj = ctx.georef.geom_px_to_wgs84(g)
    elif geom.geom_type == "LineString" and ctx.georef:
        gj = ctx.georef.geom_px_to_wgs84(geom)
    bbox = [x0, y0, x1, y1]
    return Observation(
        id=ctx.next_id(prefix), label=label, kind=kind, severity=severity, confidence=round(float(np.clip(confidence, 0, 1)), 3),
        bbox_px=bbox, pixel_polygon=poly, geometry=gj, area_px=None if area_px is None else round(area_px, 1),
        area_m2=None if area_m2 is None else round(area_m2, 1), attributes=attributes or {},
        evidence=[Evidence(asset_id=getattr(ctx.latest if asset == "latest" else ctx.before, "asset_id", ""),
                           region_px=bbox, source_tool=tool, note=note)],
    )


def pct(x: float) -> float:
    return round(100 * float(x), 1)


# ------------------------------------------------------------------ registry
@dataclass
class ToolSpec:
    name: str
    description: str
    args_model: type[BaseModel]
    fn: Callable[[ToolContext, Any], ToolOutput]
    domains: tuple[Domain, ...] = ()


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        self._tools[spec.name] = spec

    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> ToolSpec:
        return self._tools[name]

    def describe(self) -> list[dict[str, Any]]:
        return [{"name": t.name, "description": t.description, "args": t.args_model.model_json_schema().get("properties", {})}
                for t in self._tools.values()]

    def validate_call(self, name: str, args: dict[str, Any]) -> BaseModel:
        if name not in self._tools:
            raise ToolError(f"Unknown tool: {name}")
        try:
            return self._tools[name].args_model.model_validate(args)
        except ValidationError as exc:
            raise ToolError(f"Invalid arguments for {name}: {exc.errors()[0]['msg']}") from exc

    def run(self, name: str, args: dict[str, Any], ctx: ToolContext) -> tuple[ToolOutput, int]:
        parsed = self.validate_call(name, args)
        t0 = time.perf_counter()
        out = self._tools[name].fn(ctx, parsed)
        return out, int((time.perf_counter() - t0) * 1000)


class NoArgs(BaseModel):
    pass


def polygon_from_box(b: list[float]) -> Polygon:
    return box(*b)
