"""Vector reference layers (buildings, roads, parks, ...), converted to the raster's pixel space."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry

from .georef import GeoRef


@dataclass
class VFeature:
    id: str
    props: dict
    geom: BaseGeometry  # pixel space (x right, y down)
    source: str = "layer"  # "layer" (authoritative-style vector data) or "vlm" (model proposal)

    @property
    def polygon(self) -> BaseGeometry:
        return self.geom


@dataclass
class VectorLayers:
    georef: GeoRef
    layers: dict[str, list[VFeature]] = field(default_factory=dict)

    def get(self, name: str) -> list[VFeature]:
        return self.layers.get(name, [])

    def has(self, name: str) -> bool:
        return bool(self.layers.get(name))

    def road_area(self, feat: VFeature) -> BaseGeometry:
        """Roads are stored as centre-lines with ``width_m``; return the carriageway polygon in pixels."""
        if feat.geom.geom_type in ("LineString", "MultiLineString"):
            gsd = self.georef.pixel_size_m or 1.0
            width_px = float(feat.props.get("width_m", 8.0)) / gsd
            return feat.geom.buffer(width_px / 2, cap_style="flat")
        return feat.geom


def load_layers(path: Path, georef: GeoRef) -> VectorLayers:
    data = json.loads(path.read_text())
    out: dict[str, list[VFeature]] = {}
    for f in data.get("features", []):
        props = dict(f.get("properties") or {})
        layer = props.pop("layer", "features")
        geom = georef.geom_wgs84_to_px(shape(f["geometry"]))
        out.setdefault(layer, []).append(VFeature(id=str(props.get("id", f"{layer}-{len(out.get(layer, []))}")), props=props, geom=geom))
    return VectorLayers(georef=georef, layers=out)
