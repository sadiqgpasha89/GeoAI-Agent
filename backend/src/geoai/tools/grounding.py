"""Open-vocabulary-style grounding, deterministic edition (demo engine) + region filters."""

from __future__ import annotations

import re
from collections.abc import Callable

from geoai.schemas import Observation
from geoai.tools.base import ToolContext, make_obs

REGION_WORDS = {"top": "upper", "upper": "upper", "bottom": "lower", "lower": "lower", "left": "left", "right": "right",
                "center": "centre", "centre": "centre", "middle": "centre"}


def region_filter(query: str, w: int, h: int) -> Callable[[list[float]], bool] | None:
    words = {REGION_WORDS[x] for x in re.findall(r"\b(top|upper|bottom|lower|left|right|center|centre|middle)\b", query.lower())}
    if not words:
        return None

    def ok(b: list[float]) -> bool:
        cx, cy = (b[0] + b[2]) / 2 / w, (b[1] + b[3]) / 2 / h
        if "upper" in words and cy > 0.5:
            return False
        if "lower" in words and cy < 0.5:
            return False
        if "left" in words and cx > 0.5:
            return False
        if "right" in words and cx < 0.5:
            return False
        if "centre" in words and not (0.25 <= cx <= 0.75 and 0.25 <= cy <= 0.75):
            return False
        return True

    return ok


# (regex, tool, labels, human label)
RULES: list[tuple[str, str, set[str], str]] = [
    (r"flooded (building|house|home|structure)|(building|house|home)s? .*flood", "assess_flood", {"flooded_building", "partially_flooded_building"}, "flooded buildings"),
    (r"(flooded|inundated|submerged) (road|street|highway)|road.*(flooded|inundated)", "assess_flood", {"flooded_road_segment"}, "flooded road sections"),
    (r"(collapsed|destroyed|pancaked)", "assess_earthquake", {"collapsed_building"}, "collapsed buildings"),
    (r"damaged (building|house|structure)|damage", "assess_earthquake", {"collapsed_building", "damaged_building"}, "damaged buildings"),
    (r"(blocked|debris|obstruct).*(road|street)|(road|street).*(blocked|debris)", "assess_earthquake", {"blocked_road_segment"}, "blocked road sections"),
    (r"rubble|debris", "assess_earthquake", {"debris_area", "collapsed_building"}, "debris"),
    (r"burn(ed|t)?[ -]?(scar|area)|burnt|charred", "assess_wildfire", {"burn_scar"}, "burn scars"),
    (r"smoke|plume", "assess_wildfire", {"smoke_plume"}, "smoke plume"),
    (r"fire|flame|blaze", "assess_wildfire", {"active_fire_front"}, "active fire fronts"),
    (r"at[- ]risk|threatened|endangered", "assess_wildfire", {"building_high_risk", "building_watch_risk", "building_in_burn_scar"}, "buildings at risk"),
    (r"evacuation", "assess_wildfire", {"evacuation_zone"}, "evacuation zone"),
    (r"(stress|unhealthy|dry|dead|poor)|anomal", "assess_crop_health", {"field_severe", "field_stressed", "stress_anomaly"}, "stressed crop areas"),
    (r"park|green ?space|open space", "assess_urban", {"park"}, "parks"),
    (r"zoning|conflict|violation", "assess_urban", {"zoning_conflict"}, "zoning conflicts"),
    (r"construction|new (development|building)|developed", "assess_urban", {"construction_site", "new_development"}, "new development / construction"),
    (r"water|flood(ed)? (area|extent|zone)|river|lake|inundat|flood", "assess_flood", {"flood_extent"}, "water / flood extent"),
]
LAYER_RULES: list[tuple[str, str, str]] = [
    (r"building|house|home|structure|roof|school|clinic", "buildings", "buildings"),
    (r"road|street|highway|avenue|bridge", "roads", "roads"),
    (r"field|crop|farm|parcel", "fields", "fields"),
]


def demo_ground(ctx: ToolContext, query: str, run_tool: Callable[[str], list[Observation]]) -> tuple[list[Observation], str | None]:
    q = query.lower()
    for pat, tool, labels, human in RULES:
        if re.search(pat, q):
            obs = [o for o in run_tool(tool) if o.label in labels]
            return obs, human
    for pat, layer, human in LAYER_RULES:
        if re.search(pat, q) and ctx.layers and ctx.layers.has(layer):
            obs = []
            feats = ctx.layers.get(layer)
            uses = {str(f.props["use"]).lower() for f in feats if f.props.get("use")}
            wanted = {u for u in uses if re.search(rf"\b{re.escape(u)}\b", q)}
            if wanted:
                feats = [f for f in feats if str(f.props.get("use", "")).lower() in wanted]
                human = f"{'/'.join(sorted(wanted))} {human}"
            for f in feats:
                geom = ctx.layers.road_area(f) if layer == "roads" else f.geom
                obs.append(make_obs(ctx, "ground_object", layer[:3], layer.rstrip("s"), geom, 0.9, attributes={"feature_id": f.id, **{k: v for k, v in f.props.items() if k in ("name", "use", "crop")}}))
            return obs, human
    return [], None
