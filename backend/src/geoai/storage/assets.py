"""Asset resolution: bundled sample scenes + validated user uploads."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from geoai.config import Settings
from geoai.geo.raster import Raster, RasterError, load_raster, sha256_file
from geoai.geo.vector import VectorLayers, load_layers
from geoai.schemas import AssetInfo

_MAGIC = {
    "image/png": lambda b: b.startswith(b"\x89PNG\r\n\x1a\n"),
    "image/jpeg": lambda b: b.startswith(b"\xff\xd8\xff"),
    "image/webp": lambda b: b[:4] == b"RIFF" and b[8:12] == b"WEBP",
    "image/tiff": lambda b: b[:4] in (b"II*\x00", b"MM\x00*"),
}
_EXT = {"image/png": ".png", "image/jpeg": ".jpg", "image/webp": ".webp", "image/tiff": ".tif"}


class UploadError(ValueError):
    """Invalid upload (maps to HTTP 400/413/422)."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class AssetNotFound(LookupError):
    pass


@dataclass
class AssetBundle:
    asset_id: str
    raster: Raster
    info: AssetInfo
    path: Path
    layers: VectorLayers | None = None
    scene_meta: dict | None = None


def sniff_mime(data: bytes) -> str | None:
    head = data[:16]
    return next((m for m, check in _MAGIC.items() if check(head)), None)


def clamav_scan(data: bytes, host: str, port: int) -> None:
    """Stream bytes to clamd (INSTREAM). Raises UploadError when a signature matches or scanning fails."""
    try:
        with socket.create_connection((host, port), timeout=10) as s:
            s.sendall(b"zINSTREAM\0")
            for i in range(0, len(data), 1 << 16):
                chunk = data[i : i + (1 << 16)]
                s.sendall(struct.pack("!I", len(chunk)) + chunk)
            s.sendall(struct.pack("!I", 0))
            reply = s.recv(4096).decode(errors="replace")
    except OSError as exc:  # fail closed
        raise UploadError("Malware scanner unavailable; upload rejected", 503) from exc
    if "FOUND" in reply:
        raise UploadError("Upload rejected by malware scanner", 422)


class AssetService:
    def __init__(self, settings: Settings, db):  # noqa: ANN001 - Database (avoid import cycle in typing)
        self.settings = settings
        self.db = db
        settings.upload_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[tuple[str, float], AssetBundle] = {}

    # ------------------------------------------------------------- samples
    def sample_ids(self) -> list[str]:
        d = self.settings.sample_dir
        return sorted(p.parent.name for p in d.glob("*/scene.json")) if d.exists() else []

    def sample_meta(self, scene_id: str) -> dict:
        p = self.settings.sample_dir / scene_id / "scene.json"
        if not p.exists():
            raise AssetNotFound(f"Unknown sample scene: {scene_id}")
        return json.loads(p.read_text())

    def _sample_bundle(self, asset_id: str) -> AssetBundle:
        try:
            _, scene_id, which = asset_id.split(":")
        except ValueError as exc:
            raise AssetNotFound(f"Malformed sample asset id: {asset_id}") from exc
        if which not in ("before", "after") or "/" in scene_id or ".." in scene_id:
            raise AssetNotFound(f"Unknown sample asset: {asset_id}")
        meta = self.sample_meta(scene_id)
        path = self.settings.sample_dir / scene_id / f"{which}.tif"
        if not path.exists():
            raise AssetNotFound(f"Unknown sample asset: {asset_id}")
        key = (str(path), path.stat().st_mtime)
        if key in self._cache:
            return self._cache[key]
        raster = load_raster(path)
        info = self._info(asset_id, path, raster, "image/tiff", f"{scene_id}-{which}.tif", sample=True)
        layers = None
        lp = path.parent / "layers.geojson"
        if raster.georef and lp.exists():
            layers = load_layers(lp, raster.georef)
        bundle = AssetBundle(asset_id, raster, info, path, layers, meta)
        self._cache[key] = bundle
        return bundle

    def sample_asset_id(self, scene_id: str, which: str = "after") -> str:
        return f"sample:{scene_id}:{which}"

    # ------------------------------------------------------------- uploads
    def _info(self, asset_id: str, path: Path, raster: Raster, mime: str, filename: str | None, sample: bool = False) -> AssetInfo:
        g = raster.georef
        return AssetInfo(
            id=asset_id, filename=filename, content_type=mime, width=raster.width, height=raster.height,
            bands=list(raster.bands), georeferenced=g is not None, crs=g.crs if g else None,
            gsd_m=round(g.pixel_size_m, 4) if g and g.pixel_size_m else None,
            captured_at=raster.captured_at, bounds_wgs84=g.bounds_wgs84() if g else None,
            sha256=sha256_file(path), size_bytes=path.stat().st_size, sample=sample,
        )

    def register_upload(self, data: bytes, filename: str | None, tenant: str) -> AssetInfo:
        s = self.settings
        if not data:
            raise UploadError("Empty file", 400)
        if len(data) > s.max_upload_mb * 1024 * 1024:
            raise UploadError(f"File exceeds the {s.max_upload_mb} MB limit", 413)
        mime = sniff_mime(data)
        if mime is None:
            raise UploadError("Unsupported file type. Upload PNG, JPEG, WebP or GeoTIFF imagery.", 415)
        if s.clamav_host:
            clamav_scan(data, s.clamav_host, s.clamav_port)

        asset_id = f"a_{uuid.uuid4().hex[:16]}"
        key = f"{asset_id}{_EXT[mime]}"
        path = s.upload_dir / key
        path.write_bytes(data)
        try:
            self._check_dimensions(path, mime)
            raster = load_raster(path)
            info = self._info(asset_id, path, raster, mime, _safe_name(filename))
        except (RasterError, Image.DecompressionBombError, OSError) as exc:
            path.unlink(missing_ok=True)
            raise UploadError(f"Invalid image: {exc}", 422) from exc
        except UploadError:
            path.unlink(missing_ok=True)
            raise
        self.db.add_asset(info, tenant, key)
        return info

    def _check_dimensions(self, path: Path, mime: str) -> None:
        if mime == "image/tiff":
            import rasterio

            try:
                with rasterio.open(path) as src:
                    w, h = src.width, src.height
            except Exception as exc:  # noqa: BLE001
                raise RasterError(f"Cannot read GeoTIFF: {exc}") from exc
        else:
            with Image.open(path) as im:
                w, h = im.size
        if w < 32 or h < 32:
            raise UploadError("Image is too small (minimum 32x32 pixels)")
        if w * h > self.settings.max_image_pixels:
            raise UploadError(f"Image exceeds the {self.settings.max_image_pixels:,}-pixel limit")

    # ------------------------------------------------------------- resolve
    def resolve(self, asset_id: str, tenant: str) -> AssetBundle:
        if asset_id.startswith("sample:"):
            return self._sample_bundle(asset_id)
        found = self.db.get_asset(asset_id, tenant)
        if found is None:
            raise AssetNotFound(f"Asset not found: {asset_id}")
        info, key = found
        path = self.settings.upload_dir / key
        ck = (str(path), path.stat().st_mtime)
        if ck in self._cache:
            return self._cache[ck]
        bundle = AssetBundle(asset_id, load_raster(path), info, path)
        if len(self._cache) > 32:
            self._cache.clear()
        self._cache[ck] = bundle
        return bundle


def _safe_name(name: str | None) -> str | None:
    if not name:
        return None
    return "".join(ch for ch in Path(name).name if ch.isalnum() or ch in "._- ")[:120] or None


def hash_payload(obj: object) -> str:
    import json as _json

    return hashlib.sha256(_json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()
