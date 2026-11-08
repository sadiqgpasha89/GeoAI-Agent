"""CRS-aware pixel <-> world conversions. Pixel space and geographic space stay distinct."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer
from shapely.geometry import mapping
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as shp_transform

WGS84 = "EPSG:4326"


@dataclass(frozen=True)
class GeoRef:
    """Affine transform (pixel -> CRS coordinates) plus the CRS of the raster."""

    transform: Affine
    crs: str
    width: int
    height: int

    @cached_property
    def _crs(self) -> CRS:
        return CRS.from_user_input(self.crs)

    @cached_property
    def _to_wgs84(self) -> Transformer:
        return Transformer.from_crs(self._crs, WGS84, always_xy=True)

    @cached_property
    def _from_wgs84(self) -> Transformer:
        return Transformer.from_crs(WGS84, self._crs, always_xy=True)

    # -- pixel <-> CRS ----------------------------------------------------
    def px_to_crs(self, x: float, y: float) -> tuple[float, float]:
        return self.transform @ (x, y)

    def crs_to_px(self, x: float, y: float) -> tuple[float, float]:
        return (~self.transform) @ (x, y)

    def px_to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        cx, cy = self.px_to_crs(x, y)
        lon, lat = self._to_wgs84.transform(cx, cy)
        return float(lon), float(lat)

    def lonlat_to_px(self, lon: float, lat: float) -> tuple[float, float]:
        cx, cy = self._from_wgs84.transform(lon, lat)
        return self.crs_to_px(cx, cy)

    # -- metric scale -----------------------------------------------------
    @cached_property
    def pixel_size_m(self) -> float | None:
        """Ground sampling distance in metres (geometric mean of x/y), if the CRS is metric."""
        a, e = abs(self.transform.a), abs(self.transform.e)
        if self._crs.is_projected:
            factor = self._crs.axis_info[0].unit_conversion_factor or 1.0
            return float(np.sqrt(a * e) * factor)
        # geographic CRS: approximate at the scene centre
        lat = self.px_to_lonlat(self.width / 2, self.height / 2)[1]
        m_per_deg_lat = 111_320.0
        m_per_deg_lon = 111_320.0 * float(np.cos(np.radians(lat)))
        return float(np.sqrt(a * m_per_deg_lon * e * m_per_deg_lat))

    @property
    def pixel_area_m2(self) -> float | None:
        s = self.pixel_size_m
        return None if s is None else s * s

    # -- geometry ---------------------------------------------------------
    def geom_px_to_wgs84(self, geom: BaseGeometry) -> dict[str, Any]:
        def fn(x, y, z=None):
            xs = np.asarray(x, dtype=float)
            ys = np.asarray(y, dtype=float)
            cx = self.transform.c + self.transform.a * xs + self.transform.b * ys
            cy = self.transform.f + self.transform.d * xs + self.transform.e * ys
            lon, lat = self._to_wgs84.transform(cx, cy)
            return lon, lat

        return mapping(shp_transform(fn, geom))

    def geom_wgs84_to_px(self, geom: BaseGeometry) -> BaseGeometry:
        inv = ~self.transform

        def fn(x, y, z=None):
            cx, cy = self._from_wgs84.transform(np.asarray(x), np.asarray(y))
            px = inv.a * cx + inv.b * cy + inv.c
            py = inv.d * cx + inv.e * cy + inv.f
            return px, py

        return shp_transform(fn, geom)

    def bounds_wgs84(self) -> list[float]:
        corners = [(0, 0), (self.width, 0), (self.width, self.height), (0, self.height)]
        pts = [self.px_to_lonlat(x, y) for x, y in corners]
        lons, lats = zip(*pts, strict=True)
        return [min(lons), min(lats), max(lons), max(lats)]

    def to_dict(self) -> dict[str, Any]:
        return {"crs": self.crs, "transform": list(self.transform)[:6], "width": self.width, "height": self.height}
