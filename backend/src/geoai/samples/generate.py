"""Deterministic synthetic sample scenes (one per domain, each with a before/after pair).

The scenes are *synthetic* by design: no third-party imagery licences are involved, ground truth is
known, and the demo pipeline runs offline. Each scene ships:

* ``before.tif`` / ``after.tif``  - georeferenced rasters (UTM, with acquisition-time tags)
* ``before.png`` / ``after.png``  - display previews
* ``layers.geojson``              - authoritative-style vector layers in EPSG:4326
* ``scene.json``                  - metadata, suggested questions, context fixtures

Run: ``geoai-generate-samples --out data/samples``
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy import ndimage as ndi
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import unary_union

from geoai.geo.georef import GeoRef
from geoai.geo.raster import Raster
from geoai.geo.regions import rasterize

W, H = 800, 600
Color = tuple[int, int, int]


# --------------------------------------------------------------------------- helpers
def fbm(rng: np.random.Generator, scales=(60, 22, 7), weights=(0.6, 0.3, 0.1)) -> np.ndarray:
    out = np.zeros((H, W))
    for s, wt in zip(scales, weights, strict=True):
        n = ndi.gaussian_filter(rng.standard_normal((H, W)), s)
        out += wt * (n - n.mean()) / (n.std() + 1e-9)
    return (out - out.min()) / (out.max() - out.min())


class Canvas:
    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        self.img = np.zeros((H, W, 3), np.float32)

    def copy(self) -> Canvas:
        c = Canvas(self.rng)
        c.img = self.img.copy()
        return c

    def fill(self, color: Color, amp: float = 0.02, field: np.ndarray | None = None, field_amp: float = 0.0):
        base = np.array(color, np.float32) / 255
        self.img[:] = base
        if field is not None:
            self.img += ((field - 0.5) * field_amp)[..., None]
        self.img += self.rng.normal(0, amp, (H, W, 1)).astype(np.float32)
        np.clip(self.img, 0, 1, out=self.img)

    def paint(self, mask: np.ndarray, color: Color | np.ndarray, amp: float = 0.012, alpha: float | np.ndarray = 1.0):
        if not mask.any():
            return
        col = np.asarray(color, np.float32)
        if col.ndim == 3:
            col = col[mask]
        col = col / 255
        noise = self.rng.normal(0, amp, (int(mask.sum()), 1)).astype(np.float32)
        a = alpha[mask][:, None] if isinstance(alpha, np.ndarray) else alpha
        self.img[mask] = (1 - a) * self.img[mask] + a * np.clip(col + noise, 0, 1)

    def geom(self, g, color: Color, amp: float = 0.012, alpha: float = 1.0):
        self.paint(rasterize([g], (H, W)), color, amp, alpha)


def blob(rng, cx, cy, r, jitter=0.25, n=28) -> Polygon:
    ang = np.linspace(0, 2 * math.pi, n, endpoint=False)
    rad = r * (1 + jitter * (rng.random(n) - 0.5) * 2)
    return Polygon(np.c_[cx + rad * np.cos(ang), cy + rad * np.sin(ang)]).buffer(r * 0.15).buffer(-r * 0.1)


def dashes(line: LineString, every=44, length=20, width=1.6) -> list[Polygon]:
    out = []
    d = 6.0
    while d + length < line.length:
        out.append(LineString([line.interpolate(d), line.interpolate(d + length)]).buffer(width))
        d += every
    return out


def dist_to(geoms) -> np.ndarray:
    return ndi.distance_transform_edt(~rasterize(geoms, (H, W)))


def utm_ref(crs: str, e0: float, n0: float, gsd: float) -> GeoRef:
    return GeoRef(from_origin(e0, n0, gsd, gsd), crs, W, H)


def to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)


def write_tif(path: Path, data: np.ndarray, ref: GeoRef, names: list[str], tags: dict):
    with rasterio.open(
        path, "w", driver="GTiff", height=H, width=W, count=data.shape[0], dtype=data.dtype,
        crs=ref.crs, transform=ref.transform, compress="deflate", predictor=2,
    ) as dst:
        dst.write(data)
        for i, n in enumerate(names, start=1):
            dst.set_band_description(i, n)
        dst.update_tags(**tags)


def write_scene(out: Path, meta: dict, ref: GeoRef, before, after, layers: dict[str, list[tuple[dict, object]]]):
    out.mkdir(parents=True, exist_ok=True)
    for name, arr, when in (("before", before, meta["before_at"]), ("after", after, meta["after_at"])):
        tags = {"ACQUISITION_DATETIME": when, "SENSOR": meta["sensor"]}
        if isinstance(arr, Canvas):
            write_tif(out / f"{name}.tif", to_uint8(arr.img).transpose(2, 0, 1), ref, ["red", "green", "blue"], tags)
            bands = {"red": arr.img[..., 0], "green": arr.img[..., 1], "blue": arr.img[..., 2]}
        else:  # multispectral dict of reflectance bands
            stack = np.stack([arr[k] for k in ("red", "green", "blue", "nir")])
            write_tif(out / f"{name}.tif", np.round(stack * 10000).astype(np.uint16), ref,
                      ["red", "green", "blue", "nir"], tags)
            bands = {k: v.astype(np.float32) for k, v in arr.items()}
        (out / f"{name}.png").write_bytes(Raster(bands=bands).to_png_bytes())
    feats = []
    for layer, items in layers.items():
        for props, geom in items:
            feats.append({
                "type": "Feature",
                "properties": {"layer": layer, **props},
                "geometry": ref.geom_px_to_wgs84(geom),
            })
    (out / "layers.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))
    meta = {**meta, "width": W, "height": H, "crs": ref.crs, "gsd_m": ref.pixel_size_m,
            "bounds_wgs84": ref.bounds_wgs84()}
    (out / "scene.json").write_text(json.dumps(meta, indent=2))


def road_layer(roads: list[tuple[str, LineString, float]], gsd: float):
    return [({"id": f"r{i + 1:02d}", "name": n, "width_m": round(w * gsd, 1)}, ln) for i, (n, ln, w) in enumerate(roads)]


def draw_roads(c: Canvas, roads, color=(84, 84, 88), mark=True):
    for _, ln, w in roads:
        c.geom(ln.buffer(w / 2, cap_style="flat"), color, 0.012)
    if mark:
        for _, ln, w in roads:
            if w >= 22:
                for d in dashes(ln):
                    c.geom(d, (222, 200, 90), 0.01)


ROOFS: list[Color] = [(170, 92, 72), (112, 112, 118), (190, 172, 150), (214, 214, 208), (122, 88, 68), (96, 100, 108)]


# --------------------------------------------------------------------------- 1. flood
def scene_flood(out: Path):
    rng = np.random.default_rng(11)
    gsd = 0.6
    ref = utm_ref("EPSG:32610", 612000, 4180000, gsd)
    terrain = fbm(rng)
    river = LineString([(830, 30), (700, 175), (555, 300), (480, 430), (425, 640)])
    trib = LineString([(555, 300), (440, 245), (340, 205), (262, 112), (240, 40)])
    roads = [
        ("Riverbend Road", LineString([(0, 337), (250, 330), (520, 338), (800, 325)]), 26),
        ("Oak Street", LineString([(200, 0), (205, 140), (198, 335)]), 18),
        ("Mill Lane", LineString([(198, 335), (215, 450), (210, 600)]), 18),
    ]
    sites = [(97, 70), (97, 150), (97, 235), (300, 65), (300, 140), (300, 225), (97, 405), (300, 420), (120, 520), (310, 520)]
    uses = ["residential"] * 10
    uses[4], uses[5] = "clinic", "school"
    blds = []
    for i, (cx, cy) in enumerate(sites):
        w, h = rng.uniform(46, 66), rng.uniform(36, 52)
        blds.append(({"id": f"b{i + 1:02d}", "use": uses[i], "name": f"Building {i + 1}"}, box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    colors = [ROOFS[i % len(ROOFS)] for i in range(len(blds))]
    elev = np.minimum(dist_to([river]) / 150, dist_to([trib]) / 70 + 0.1) + (terrain - 0.5) * 0.16

    def compose(level: float, overlay: bool) -> Canvas:
        c = Canvas(np.random.default_rng(5))
        c.fill((92, 126, 62), 0.018, terrain, 0.14)
        trees = unary_union([blob(rng, rng.uniform(0, W), rng.uniform(0, H), rng.uniform(14, 30)) for _ in range(46)])
        c.geom(trees.difference(unary_union([ln.buffer(w) for _, ln, w in roads] + [river.buffer(60)])), (52, 92, 46), 0.03)
        water = elev < level
        wcol = np.zeros((H, W, 3)) + np.array([82, 110, 135])
        wcol -= (np.clip(level - elev, 0, 0.6) * 55)[..., None]
        if not overlay:
            c.paint(water, wcol, 0.012)
        draw_roads(c, roads)
        for (_, g), col in zip(blds, colors, strict=True):
            c.geom(g.buffer(2), (60, 60, 62), 0.01)
            c.geom(g, col, 0.012)
        if overlay:
            c.paint(water, wcol, 0.012, alpha=0.9)
        return c

    layers = {"buildings": blds, "roads": road_layer(roads, gsd)}
    meta = {
        "id": "flood-riverbend", "domain": "flood", "title": "Riverbend flood extent",
        "location": "Riverbend (synthetic scene)", "sensor": "Synthetic aerial RGB (0.6 m)",
        "before_at": "2026-10-28T16:05:00Z", "after_at": "2026-11-02T10:42:00Z",
        "description": "Riverside neighbourhood before and after a river flood. Synthetic imagery with known ground truth.",
        "suggested_questions": [
            "Describe this scene in detail",
            "How many buildings are flooded?",
            "Is the whole road flooded?",
            "Summarize flood impact and prioritise response",
            "Locate the flooded buildings",
        ],
    }
    write_scene(out, meta, ref, compose(0.06, False), compose(0.74, True), layers)


# --------------------------------------------------------------------------- 2. wildfire
def scene_wildfire(out: Path):
    rng = np.random.default_rng(23)
    gsd = 4.0
    ref = utm_ref("EPSG:32610", 702000, 4310000, gsd)
    terrain = fbm(rng)
    forest_mask_noise = fbm(rng, (80, 30, 10))
    lake = blob(rng, 120, 480, 62, 0.2)
    roads = [
        ("Highway 9", LineString([(0, 470), (250, 455), (520, 460), (800, 430)]), 14),
        ("Ridge Road", LineString([(610, 455), (590, 330), (540, 180), (500, 0)]), 8),
    ]
    town = []
    for i in range(46):
        x = 540 + rng.uniform(0, 200)
        y = 400 + rng.uniform(0, 140)
        if abs(y - 460) < 14:
            continue
        w, h = rng.uniform(6, 10), rng.uniform(5, 9)
        town.append(({"id": f"h{len(town) + 1:02d}", "use": "residential"}, box(x, y, x + w, y + h)))
    evac = unary_union([g for _, g in town]).convex_hull.buffer(95)

    yy, xx = np.mgrid[0:H, 0:W]
    ang = math.radians(28)
    dx, dy = xx - 430, yy - 250
    u, v = dx * math.cos(ang) + dy * math.sin(ang), -dx * math.sin(ang) + dy * math.cos(ang)
    score = 1 - (u / 235) ** 2 - (v / 118) ** 2 + (fbm(rng, (40, 14, 5)) - 0.5) * 0.45
    burn = score > 0
    edge = burn & ~ndi.binary_erosion(burn, iterations=7)
    front = edge & (u > 40) & (fbm(rng, (10, 4, 2)) > 0.42)  # leading (south-east) edge is still burning
    wind = np.array([math.cos(math.radians(-40)), math.sin(math.radians(-40))])  # smoke drifts north-east
    base = np.array([float(xx[front].mean()), float(yy[front].mean())])
    rel = np.stack([xx - base[0], yy - base[1]], -1)
    along, across = rel @ wind, rel @ np.array([-wind[1], wind[0]])
    plume = np.clip(1 - np.abs(across) / (40 + 0.42 * np.clip(along, 0, None)), 0, 1) ** 0.7 * (along > -30) * np.exp(-np.clip(along, 0, None) / 260)

    def compose(after: bool) -> Canvas:
        c = Canvas(np.random.default_rng(6))
        c.fill((40, 84, 46), 0.02, terrain, 0.12)
        c.paint(forest_mask_noise > 0.64, (98, 132, 72), 0.025, 0.9)  # meadows
        c.geom(lake, (58, 98, 132), 0.012)
        draw_roads(c, roads, (122, 117, 112), mark=False)
        for i, (_, g) in enumerate(town):
            c.geom(g, [(152, 146, 140), (172, 122, 102), (190, 184, 176)][i % 3], 0.01)
        if after:
            ash = (fbm(np.random.default_rng(9), (6, 2, 1)) - 0.5) * 30
            c.paint(burn, np.clip(np.array([38, 36, 35]) + ash[..., None], 0, 255), 0.01)
            glow = ndi.gaussian_filter(front.astype(float), 4) * 1.5
            c.paint(glow > 0.25, (190, 80, 36), 0.02, alpha=np.clip(glow, 0, 0.8))
            c.paint(front, (255, 128, 34), 0.025)
            c.paint(plume > 0.03, (212, 212, 216), 0.006, alpha=np.clip(plume * 0.85, 0, 0.7))
        return c

    layers = {"buildings": town, "roads": road_layer(roads, gsd), "evacuation_zones": [({"id": "ez1", "name": "Pine Ridge evacuation zone"}, evac)]}
    meta = {
        "id": "wildfire-pineridge", "domain": "wildfire", "title": "Pine Ridge fire perimeter",
        "location": "Pine Ridge (synthetic scene)", "sensor": "Synthetic multi-sensor RGB (4 m)",
        "before_at": "2026-10-20T18:30:00Z", "after_at": "2026-11-05T18:12:00Z",
        "weather": {"source": "demo fixture", "wind_from_deg": 225, "wind_speed_kmh": 28, "gust_kmh": 46,
                    "relative_humidity_pct": 14, "temperature_c": 36},
        "description": "Forested foothills with a small town to the south-east; a fast-moving fire front and smoke plume.",
        "suggested_questions": [
            "Describe this scene in detail",
            "Where are the burn scars and active fire fronts?",
            "How much area has burned?",
            "Which buildings are at risk?",
            "Locate the smoke plume",
        ],
    }
    write_scene(out, meta, ref, compose(False), compose(True), layers)


# --------------------------------------------------------------------------- 3. earthquake
def scene_earthquake(out: Path):
    rng = np.random.default_rng(37)
    gsd = 0.3
    ref = utm_ref("EPSG:32611", 380000, 3770000, gsd)
    roads = [
        ("Main Street", LineString([(0, 200), (800, 200)]), 40),
        ("Harbor Avenue", LineString([(0, 420), (800, 420)]), 40),
        ("1st Avenue", LineString([(275, 0), (275, 600)]), 36),
        ("2nd Avenue", LineString([(545, 0), (545, 600)]), 36),
    ]
    xs = [(25, 245), (305, 515), (575, 780)]
    ys = [(25, 170), (230, 390), (450, 580)]
    blds = []
    for xi, (x0, x1) in enumerate(xs):
        for yi, (y0, y1) in enumerate(ys):
            if (xi, yi) == (2, 2):
                continue
            n = 2 if (x1 - x0) > 200 else 1
            w = (x1 - x0 - 20 * (n + 1)) / n
            for k in range(n):
                bx = x0 + 20 + k * (w + 20)
                by0, by1 = y0 + 14, y1 - 14
                blds.append(({"id": f"b{len(blds) + 1:02d}", "use": "commercial" if yi == 1 else "residential"}, box(bx, by0, bx + w, by1)))
    damage = {"b03": "collapsed", "b04": "collapsed", "b07": "collapsed", "b09": "damaged", "b10": "damaged", "b12": "damaged", "b02": "damaged", "b14": "damaged"}
    colors = [ROOFS[(i * 5 + 1) % len(ROOFS)] for i in range(len(blds))]
    trees = [Point(x, 200 + s * 30).buffer(7) for x in range(30, 800, 52) for s in (-1, 1)]
    cols = [(150, 140, 125), (120, 110, 100), (176, 160, 140), (96, 90, 84), (205, 198, 186), (140, 100, 80)]

    def rubble(c: Canvas, geom: Polygon):
        m = rasterize([geom], (H, W))
        idx = np.random.default_rng(int(geom.area)).integers(0, len(cols), (H // 3 + 1, W // 3 + 1))
        up = np.kron(idx, np.ones((3, 3), int))[:H, :W]
        c.paint(m, np.array(cols, float)[up], 0.045)

    def compose(after: bool) -> Canvas:
        c = Canvas(np.random.default_rng(7))
        c.fill((118, 124, 98), 0.02)
        draw_roads(c, roads, (88, 88, 92), mark=True)
        for t in trees:
            c.geom(t, (60, 100, 52), 0.03)
        for (p, g), col in zip(blds, colors, strict=True):
            st = damage.get(p["id"]) if after else None
            c.geom(g.buffer(1.5), (66, 66, 68), 0.01)
            c.geom(g, col, 0.01)
            if st == "collapsed":
                rubble(c, g.buffer(rng.uniform(4, 10)).buffer(-2))
            elif st == "damaged":
                x0, y0, x1, y1 = g.bounds
                rubble(c, box(x0, y0, x0 + (x1 - x0) * 0.42, y1).intersection(g.buffer(5)))
        if after:  # debris spilling onto the roads
            rubble(c, box(305, 182, 380, 218))  # fully blocks Main Street lane width
            rubble(c, box(238, 405, 282, 436).buffer(3))  # partial block on Harbor Avenue / 1st Avenue
            rubble(c, box(600, 188, 650, 205))
        return c

    layers = {"buildings": blds, "roads": road_layer(roads, gsd)}
    meta = {
        "id": "quake-northpoint", "domain": "earthquake", "title": "Northpoint damage survey",
        "location": "Northpoint (synthetic scene)", "sensor": "Synthetic UAS RGB (0.3 m)",
        "before_at": "2026-09-14T17:00:00Z", "after_at": "2026-11-03T15:20:00Z",
        "description": "Commercial/residential blocks before and after a strong earthquake.",
        "suggested_questions": [
            "Describe this scene in detail",
            "Which buildings are damaged or collapsed?",
            "Are any roads blocked by debris?",
            "Rank buildings for field inspection",
            "Locate the collapsed buildings",
        ],
    }
    write_scene(out, meta, ref, compose(False), compose(True), layers)


# --------------------------------------------------------------------------- 4. urban
def scene_urban(out: Path):
    rng = np.random.default_rng(41)
    gsd = 2.0
    ref = utm_ref("EPSG:32610", 560000, 4180000, gsd)
    vx, hy = [0, 160, 320, 480, 640, 800], [0, 150, 300, 450, 600]
    rw = 14
    roads = [(f"Avenue {i}", LineString([(x, 0), (x, H)]), rw) for i, x in enumerate(vx[1:-1], start=1)]
    roads += [(f"Street {i}", LineString([(0, y), (W, y)]), rw) for i, y in enumerate(hy[1:-1], start=1)]
    new_road = ("New Link Road", LineString([(640, 375), (800, 375)]), rw)
    blocks = {}
    kinds_before = {
        (0, 0): "dense", (1, 0): "dense", (2, 0): "park", (3, 0): "dense", (4, 0): "dense",
        (0, 1): "dense", (1, 1): "dense", (2, 1): "dense", (3, 1): "park", (4, 1): "bare",
        (0, 2): "dense", (1, 2): "park", (2, 2): "dense", (3, 2): "pond", (4, 2): "bare",
        (0, 3): "dense", (1, 3): "dense", (2, 3): "dense", (3, 3): "bare", (4, 3): "dense",
    }
    changes = {(4, 1): "dense", (4, 2): "dense", (3, 3): "bare", (3, 1): "bare"}  # after-state differences
    zoning = {k: ("park" if v == "park" else "mixed-use" if v in ("dense", "bare") else "water") for k, v in kinds_before.items()}
    zoning[(4, 1)] = zoning[(4, 2)] = "mixed-use"
    zoning[(3, 3)] = "reserved"
    zoning[(3, 1)] = "park"  # a protected park block that is stripped in the after image
    for (i, j), k in kinds_before.items():
        blocks[(i, j)] = box(vx[i] + rw / 2, hy[j] + rw / 2, vx[i + 1] - rw / 2, hy[j + 1] - rw / 2)

    def fill_block(c: Canvas, kind: str, g: Polygon, bl: list):
        x0, y0, x1, y1 = g.bounds
        if kind == "dense":
            c.geom(g, (150, 148, 144), 0.012)
            y = y0 + 6
            while y < y1 - 16:
                x = x0 + 6
                hgt = rng.uniform(13, 22)
                while x < x1 - 16:
                    wd = rng.uniform(12, 24)
                    b = box(x, y, min(x + wd, x1 - 5), min(y + hgt, y1 - 5))
                    bl.append(b)
                    c.geom(b, ROOFS[int(rng.integers(0, len(ROOFS)))], 0.012)
                    x += wd + rng.uniform(4, 8)
                y += hgt + rng.uniform(6, 9)
        elif kind == "park":
            c.geom(g, (88, 138, 72), 0.02)
            for _ in range(26):
                c.geom(Point(rng.uniform(x0 + 6, x1 - 6), rng.uniform(y0 + 6, y1 - 6)).buffer(rng.uniform(3, 6)), (46, 96, 48), 0.03)
        elif kind == "bare":
            c.geom(g, (176, 150, 105), 0.025)
        elif kind == "pond":
            c.geom(g, (88, 138, 72), 0.02)
            c.geom(blob(rng, (x0 + x1) / 2, (y0 + y1) / 2, 55, 0.2), (60, 100, 135), 0.012)

    def compose(after: bool) -> tuple[Canvas, list]:
        c = Canvas(np.random.default_rng(8))
        c.fill((140, 138, 135), 0.01)
        for _, ln, w in roads + ([new_road] if after else []):
            c.geom(ln.buffer(w / 2, cap_style="flat"), (92, 92, 96), 0.012)
        bl: list = []
        for key, g in blocks.items():
            kind = changes.get(key, kinds_before[key]) if after else kinds_before[key]
            fill_block(c, kind, g, bl)
        return c, bl

    before, _ = compose(False)
    after, after_bld = compose(True)
    bld_items = [({"id": f"b{i + 1:03d}", "use": "residential" if i % 3 else "commercial"}, g) for i, g in enumerate(after_bld)]
    parks = [({"id": f"p{i + 1}", "name": f"Park {i + 1}"}, blocks[k]) for i, k in enumerate(sorted(k for k, v in zoning.items() if v == "park"))]
    zones = [({"id": f"z{i + 1:02d}", "zone": z}, blocks[k]) for i, (k, z) in enumerate(sorted(zoning.items()))]
    layers = {"buildings": bld_items, "roads": road_layer(roads + [new_road], gsd), "parks": parks, "zoning": zones}
    meta = {
        "id": "urban-eastdistrict", "domain": "urban", "title": "East District growth study",
        "location": "East District (synthetic scene)", "sensor": "Synthetic satellite RGB (2 m)",
        "before_at": "2024-10-05T18:40:00Z", "after_at": "2026-10-09T18:41:00Z",
        "description": "Planned district two years apart: new development on vacant lots, one protected park block cleared.",
        "suggested_questions": [
            "Describe this scene in detail",
            "How has land use changed?",
            "Is there construction that conflicts with zoning?",
            "How many buildings are within 400 m of a park?",
            "Locate the parks",
        ],
    }
    write_scene(out, meta, ref, before, after, layers)


# --------------------------------------------------------------------------- 5. agriculture
def scene_agriculture(out: Path):
    rng = np.random.default_rng(53)
    gsd = 3.0
    ref = utm_ref("EPSG:32611", 700000, 4000000, gsd)
    gap = 14
    cols, rows = 4, 3
    cw, rh = (W - gap * (cols + 1)) / cols, (H - gap * (rows + 1)) / rows
    crops = ["corn", "soybean", "corn", "wheat", "alfalfa", "soybean", "corn", "wheat", "corn", "fallow", "soybean", "alfalfa"]
    base_ndvi = {"corn": 0.82, "soybean": 0.78, "wheat": 0.74, "alfalfa": 0.85, "fallow": 0.1}
    fields = []
    for r in range(rows):
        for c_ in range(cols):
            i = r * cols + c_
            x0, y0 = gap + c_ * (cw + gap), gap + r * (rh + gap)
            fields.append(({"id": f"f{i + 1:02d}", "name": f"Field {i + 1:02d}", "crop": crops[i]}, box(x0, y0, x0 + cw, y0 + rh)))
    yy, xx = np.mgrid[0:H, 0:W]
    smooth = fbm(rng, (50, 18, 6))
    stripes = np.sin(xx * 1.9) * 0.012

    def ellipse(cx, cy, rx, ry):
        return np.clip(1 - np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2), 0, 1) > 0

    f = {p["id"]: g for p, g in fields}
    bx = lambda k: f[k].bounds  # noqa: E731
    f03, f07 = bx("f03"), bx("f07")
    stress_after = [
        (ellipse(f03[0] + 0.7 * (f03[2] - f03[0]), f03[1] + 0.35 * (f03[3] - f03[1]), 85, 60), 0.34, "f03"),
        (ellipse(f07[0] + 0.3 * (f07[2] - f07[0]), f07[1] + 0.7 * (f07[3] - f07[1]), 50, 38), 0.47, "f07"),
    ]
    stress_before = [(ellipse(f03[0] + 0.7 * (f03[2] - f03[0]), f03[1] + 0.35 * (f03[3] - f03[1]), 35, 25), 0.6, "f03")]
    f06 = f["f06"]

    def make(after: bool) -> dict[str, np.ndarray]:
        nd = np.full((H, W), 0.05, np.float32)  # tracks / farmyards / margins
        for p, g in fields:
            m = rasterize([g], (H, W))
            nd[m] = base_ndvi[p["crop"]] + stripes[m] + (smooth[m] - 0.5) * 0.08
        for m, val, fid in (stress_after if after else stress_before):
            fm = rasterize([f[fid]], (H, W))
            nd[m & fm] = val + (smooth[m & fm] - 0.5) * 0.06
        f6m = rasterize([f06], (H, W))
        nd[f6m] = (0.58 if after else 0.74) + stripes[f6m] + (smooth[f6m] - 0.5) * 0.05  # mild nutrient deficiency
        canal = LineString([(0, 6), (W, 6)]).buffer(4)
        cm = rasterize([canal], (H, W))
        red = 0.2 * (1 - nd) + 0.03 + rng.normal(0, 0.004, (H, W))
        nir = np.minimum(red * (1 + nd) / (1 - nd + 1e-3), 0.95)
        green = red * (0.78 + 1.25 * nd)
        blue = red * (0.6 - 0.2 * nd)
        red, green, blue, nir = (np.where(cm, v, a) for v, a in zip((0.04, 0.06, 0.09, 0.02), (red, green, blue, nir), strict=True))
        return {"red": red.astype(np.float32), "green": green.astype(np.float32), "blue": blue.astype(np.float32), "nir": nir.astype(np.float32)}

    roads = [("County Road 12", LineString([(0, 300), (800, 300)]), 6)]
    layers = {"fields": fields, "roads": road_layer(roads, gsd)}
    meta = {
        "id": "agri-valley", "domain": "agriculture", "title": "Valley farm district",
        "location": "Valley farm district (synthetic scene)", "sensor": "Synthetic multispectral R,G,B,NIR (3 m)",
        "before_at": "2026-10-04T10:30:00Z", "after_at": "2026-10-25T10:28:00Z",
        "description": "Twelve fields with irrigation failure, a disease streak and mild nutrient deficiency; one fallow field.",
        "suggested_questions": [
            "Describe this scene in detail",
            "Which fields show crop stress?",
            "Estimate area of stressed crops",
            "How did crop health change since the earlier pass?",
            "Rank fields for scouting",
        ],
    }
    write_scene(out, meta, ref, make(False), make(True), layers)


SCENES = {
    "flood-riverbend": scene_flood,
    "wildfire-pineridge": scene_wildfire,
    "quake-northpoint": scene_earthquake,
    "urban-eastdistrict": scene_urban,
    "agri-valley": scene_agriculture,
}


def generate_all(out_dir: Path) -> None:
    for sid, fn in SCENES.items():
        fn(out_dir / sid)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic GeoAI sample scenes")
    ap.add_argument("--out", type=Path, default=Path("data/samples"))
    args = ap.parse_args()
    generate_all(args.out)
    print(f"Generated {len(SCENES)} scenes in {args.out}")


if __name__ == "__main__":
    main()
