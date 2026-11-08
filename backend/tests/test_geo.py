import numpy as np
import pytest
from affine import Affine
from shapely.geometry import box

from geoai.geo.georef import GeoRef
from geoai.geo.indices import ndvi, water_mask
from geoai.geo.raster import RasterError, load_raster
from geoai.geo.regions import clean_mask, mask_to_polygons


def test_georef_roundtrip_and_scale():
    ref = GeoRef(Affine(0.5, 0, 612000, 0, -0.5, 4180000), "EPSG:32610", 800, 600)
    assert ref.pixel_size_m == pytest.approx(0.5)
    assert ref.pixel_area_m2 == pytest.approx(0.25)
    lon, lat = ref.px_to_lonlat(400, 300)
    x, y = ref.lonlat_to_px(lon, lat)
    assert (x, y) == pytest.approx((400, 300), abs=1e-3)
    assert -123 < lon < -120 and 37 < lat < 38.5


def test_geometry_conversion_is_distinct_from_pixels():
    ref = GeoRef(Affine(1, 0, 500000, 0, -1, 4000000), "EPSG:32610", 100, 100)
    gj = ref.geom_px_to_wgs84(box(10, 10, 20, 20))
    assert gj["type"] == "Polygon"
    lon, lat = gj["coordinates"][0][0]
    assert abs(lon) < 180 and abs(lat) < 90 and lon != 10  # lon/lat, not pixel coordinates
    back = ref.geom_wgs84_to_px(__import__("shapely").geometry.shape(gj))
    assert back.bounds == pytest.approx((10, 10, 20, 20), abs=1e-3)


def test_geographic_crs_pixel_size_is_metric():
    ref = GeoRef(Affine(1e-5, 0, 10.0, 0, -1e-5, 50.0), "EPSG:4326", 100, 100)
    assert 0.5 < ref.pixel_size_m < 1.5


def test_load_geotiff_and_png(data_dir, tmp_path):
    r = load_raster(data_dir / "samples/agri-valley/after.tif")
    assert r.has_nir and r.georef and r.captured_at.year == 2026
    nd = ndvi(r)
    assert nd.min() >= -1 and nd.max() <= 1
    p = load_raster(data_dir / "samples/flood-riverbend/after.png")
    assert p.georef is None and not p.has_nir


def test_unreadable_image_raises(tmp_path):
    f = tmp_path / "bad.png"
    f.write_bytes(b"\x89PNG\r\n\x1a\n" + b"junk")
    with pytest.raises(RasterError):
        load_raster(f)


def test_water_mask_finds_flood_not_before(data_dir):
    after = load_raster(data_dir / "samples/flood-riverbend/after.tif")
    before = load_raster(data_dir / "samples/flood-riverbend/before.tif")
    a, b = water_mask(after).mean(), water_mask(before).mean()
    assert a > 0.25 and b < 0.05


def test_mask_cleanup_and_polygons():
    m = np.zeros((100, 100), bool)
    m[10:40, 10:40] = True
    m[80, 80] = True  # speck
    polys = mask_to_polygons(clean_mask(m, 50), 50)
    assert len(polys) == 1 and polys[0].area == pytest.approx(900, rel=0.1)
