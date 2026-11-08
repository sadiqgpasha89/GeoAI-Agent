"""Raster loading and normalisation.

Everything downstream works on :class:`Raster`: named float32 bands in [0, 1] plus an optional
:class:`GeoRef`. Plain JPG/PNG images are supported (no georeferencing); GeoTIFFs keep their CRS.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .georef import GeoRef

SUPPORTED_RASTER_MIME = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
    "image/tiff": ".tif",
}


class RasterError(ValueError):
    """Raised for unreadable or unsupported imagery (surfaced to the user as a 422)."""


@dataclass
class Raster:
    bands: dict[str, np.ndarray]
    georef: GeoRef | None = None
    captured_at: datetime | None = None
    sensor: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def height(self) -> int:
        return next(iter(self.bands.values())).shape[0]

    @property
    def width(self) -> int:
        return next(iter(self.bands.values())).shape[1]

    @property
    def has_nir(self) -> bool:
        return "nir" in self.bands

    @property
    def rgb(self) -> np.ndarray:
        return np.stack([self.bands["red"], self.bands["green"], self.bands["blue"]], axis=-1)

    @property
    def gsd_m(self) -> float | None:
        return self.georef.pixel_size_m if self.georef else None

    @property
    def pixel_area_m2(self) -> float | None:
        return self.georef.pixel_area_m2 if self.georef else None

    def to_png_bytes(self, max_side: int | None = None, fmt: str = "PNG") -> bytes:
        rgb = self.rgb
        if self.has_nir:  # multispectral reflectance is dark in visible bands: apply a display stretch
            rgb = np.clip(rgb * 2.8, 0, 1) ** 0.85
        img = Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8), "RGB")
        if max_side and max(img.size) > max_side:
            img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {}))
        return buf.getvalue()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _scale(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    if np.issubdtype(arr.dtype, np.integer):
        mx = float(arr.max()) if arr.size else 1.0
        denom = 10000.0 if mx <= 12000 else 65535.0  # surface reflectance vs. full 16-bit
        return np.clip(arr.astype(np.float32) / denom, 0, 1)
    return np.clip(arr.astype(np.float32), 0, 1)


def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ", "%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            dt = datetime.strptime(value.strip(), fmt)
            return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _load_tiff(path: Path) -> Raster:
    import rasterio
    from rasterio.errors import RasterioIOError

    try:
        with rasterio.open(path) as src:
            data = src.read()
            descs = [(d or "").lower() for d in src.descriptions]
            tags = src.tags()
            georef = None
            if src.crs is not None and not src.transform.is_identity:
                crs = src.crs.to_string()
                if crs:
                    georef = GeoRef(src.transform, crs, src.width, src.height)
    except RasterioIOError as exc:
        raise RasterError(f"Cannot read GeoTIFF: {exc}") from exc

    n = data.shape[0]
    if n == 1:
        g = _scale(data[0])
        bands = {"red": g, "green": g, "blue": g}
    elif n >= 3:
        names = ["red", "green", "blue", "nir"]
        if n >= 4 and descs[:4] and all(d in {"red", "green", "blue", "nir"} for d in descs[:4]):
            names = descs[:4]
        bands = {name: _scale(data[i]) for i, name in enumerate(names[: min(n, 4)])}
        if set(bands) < {"red", "green", "blue"}:
            raise RasterError("GeoTIFF band descriptions must include red, green and blue")
    else:
        raise RasterError(f"Unsupported band count: {n}")
    return Raster(
        bands=bands,
        georef=georef,
        captured_at=_parse_dt(tags.get("ACQUISITION_DATETIME") or tags.get("TIFFTAG_DATETIME")),
        sensor=tags.get("SENSOR"),
    )


def _load_pil(path: Path) -> Raster:
    try:
        with Image.open(path) as im:
            im.load()
            exif_dt = None
            try:
                exif_dt = im.getexif().get(306)
            except Exception:  # noqa: BLE001 - EXIF is best effort
                pass
            rgb = np.asarray(im.convert("RGB"), dtype=np.uint8)
    except (OSError, Image.DecompressionBombError) as exc:
        raise RasterError(f"Cannot read image: {exc}") from exc
    f = rgb.astype(np.float32) / 255.0
    return Raster(
        bands={"red": f[..., 0], "green": f[..., 1], "blue": f[..., 2]},
        captured_at=_parse_dt(exif_dt),
    )


def load_raster(path: Path) -> Raster:
    head = path.read_bytes()[:4]
    is_tiff = head[:2] in (b"II", b"MM")
    return _load_tiff(path) if is_tiff else _load_pil(path)
