"""Wildfire assessment: burn scars, active fire fronts, smoke, and exposure of buildings / evacuation zones."""

from __future__ import annotations

import math

import numpy as np
from shapely.ops import unary_union

from geoai.geo.indices import burn_like_mask, fire_mask, smoke_mask
from geoai.geo.regions import clean_mask, mask_to_polygons
from geoai.schemas import Domain, Measurement
from geoai.tools.base import Finding, NoArgs, ToolContext, ToolOutput, make_obs, pct
from geoai.tools.common import get_objects, in_scene, plural
from geoai.tools.weather import compass, get_weather

TOOL = "assess_wildfire"


def assess_wildfire(ctx: ToolContext, _args: NoArgs) -> ToolOutput:
    out = ToolOutput()
    r = ctx.latest.raster
    h, w = ctx.shape
    total = float(h * w)
    burn = burn_like_mask(r)
    if ctx.before is not None and ctx.before.raster.bands["red"].shape == burn.shape:
        burn &= ~burn_like_mask(ctx.before.raster)
        out.assumptions.append("Burn scar = newly darkened surface relative to the earlier image (pre-fire dark areas excluded).")
    else:
        out.assumptions.append("No earlier image: burn scar is estimated from dark, desaturated surfaces alone and may include shadows.")
        out.next_actions.append("Provide a pre-fire image to separate burn scars from naturally dark surfaces.")
    burn = clean_mask(burn, 150, close=3)
    fire = clean_mask(fire_mask(r), 6, close=1)
    smoke = clean_mask(smoke_mask(r), 400, close=2)
    if not ctx.georef:
        out.assumptions.append(ctx.not_georeferenced_note())

    burn_px, fire_px, smoke_px = float(burn.sum()), float(fire.sum()), float(smoke.sum())
    out.metrics["burned_area"] = ctx.area_metric(burn_px, "dark/desaturated surface change")
    out.metrics["burned_cover_pct"] = Measurement(value=pct(burn_px / total), unit="%", method="share of image pixels")
    out.metrics["active_fire_area"] = ctx.area_metric(fire_px, "saturated orange/red emitter pixels")
    out.metrics["smoke_area"] = ctx.area_metric(smoke_px, "bright low-saturation haze")

    for p in mask_to_polygons(burn, 150)[:4]:
        out.observations.append(make_obs(ctx, TOOL, "burn", "burn_scar", p, 0.86, severity="high", attributes={"area_ha": ctx.area_metric(p.area, "").value}))
    fire_regions = mask_to_polygons(clean_mask(fire | _dilate(fire, 3), 10, close=2), 10)
    for p in fire_regions[:8]:
        out.observations.append(make_obs(ctx, TOOL, "fire", "active_fire_front", p, 0.82, severity="critical"))
    for p in mask_to_polygons(smoke, 400)[:2]:
        out.observations.append(make_obs(ctx, TOOL, "smoke", "smoke_plume", p, 0.7, severity="moderate", kind="observed"))

    if burn_px:
        ha = out.metrics["burned_area"]
        out.findings.append(Finding(
            text=f"A burn scar covers {pct(burn_px / total)}% of the scene ({ha.value:,} {ha.unit}), located {in_scene(_bounds(burn), w, h)}.",
            topics=["burn", "area", "summary"]))
    else:
        out.findings.append(Finding(text="No burn scar was detected.", topics=["burn", "area", "summary"]))
    if fire_px:
        out.findings.append(Finding(text=f"{plural(len(fire_regions), 'active fire front segment')} detected {in_scene(_bounds(fire), w, h)}, along the edge of the scar.", topics=["fire", "summary"]))
        out.review_reasons.append("Active fire detected: high-consequence finding that requires analyst confirmation before alerts are issued.")
    else:
        out.findings.append(Finding(text="No active flame pixels were detected.", topics=["fire"]))
    if smoke_px:
        out.findings.append(Finding(text=f"A smoke plume ({ctx.area_metric(smoke_px, '').value:,} {ctx.area_metric(smoke_px, '').unit}) is visible {in_scene(_bounds(smoke), w, h)}; areas under smoke are not assessable.", topics=["smoke", "summary"]))
        out.limitations.append("Surface beneath the smoke plume cannot be assessed in this image.")

    # ---- weather (inference) -------------------------------------------
    wx = ctx.cached("weather", lambda: get_weather(ctx))
    downwind = None
    if wx:
        downwind = (float(wx["wind_from_deg"]) + 180) % 360
        out.metrics["wind_from"] = Measurement(value=round(float(wx["wind_from_deg"])), unit="°", method=wx["source"])
        out.findings.append(Finding(text=f"Wind is from the {compass(wx['wind_from_deg'])} at {wx['wind_speed_kmh']:.0f} km/h with {wx['relative_humidity_pct']:.0f}% humidity — spread is expected towards the {compass(downwind)}.", topics=["weather", "wind", "risk"], kind="inferred"))
        if wx["source"] == "demo fixture":
            out.assumptions.append("Weather comes from the sample scene fixture, not a live feed.")

    # ---- exposure --------------------------------------------------------
    blds, src = get_objects(ctx, "buildings", "building", "bld")
    front_geom = unary_union(fire_regions) if fire_regions else None
    burn_geom = unary_union(mask_to_polygons(burn, 150)) if burn_px else None
    if blds and (front_geom is not None or burn_geom is not None):
        ref = front_geom if front_geom is not None else burn_geom
        fire_c = (ref.centroid.x, ref.centroid.y)
        counts = {"burned": 0, "high": 0, "watch": 0}
        nearest = None
        for f in blds:
            d_fire = f.geom.distance(front_geom) if front_geom is not None else float("inf")
            d_scar = f.geom.distance(burn_geom) if burn_geom is not None else float("inf")
            d_m = ctx.px_to_m(min(d_fire, d_scar))
            if d_m is None:
                continue
            nearest = d_m if nearest is None else min(nearest, d_m)
            bearing = (math.degrees(math.atan2(f.geom.centroid.x - fire_c[0], -(f.geom.centroid.y - fire_c[1]))) + 360) % 360
            is_downwind = downwind is not None and abs((bearing - downwind + 180) % 360 - 180) <= 60
            if d_scar == 0:
                tier, sev = "burned", "critical"
            elif d_m <= 500:
                tier, sev = "high", "critical" if is_downwind else "high"
            elif d_m <= 1000:
                tier, sev = "watch", "moderate"
            else:
                continue
            counts[tier] += 1
            out.observations.append(make_obs(ctx, TOOL, "bld", f"building_{tier}_risk" if tier != "burned" else "building_in_burn_scar", f.geom.buffer(2), 0.8, severity=sev, kind="inferred" if tier != "burned" else "observed",
                                             attributes={"feature_id": f.id, "distance_to_fire_m": round(d_m), "downwind": bool(is_downwind)}))
        out.metrics["buildings_total"] = Measurement(value=len(blds), unit="count", method=f"{src} layer")
        out.metrics["buildings_in_scar"] = Measurement(value=counts["burned"], unit="count")
        out.metrics["buildings_within_500m"] = Measurement(value=counts["high"], unit="count")
        out.metrics["buildings_within_1km"] = Measurement(value=counts["watch"], unit="count")
        if nearest is not None:
            out.metrics["nearest_building_to_fire"] = Measurement(value=round(nearest), unit="m", method="edge-to-edge distance to fire front or scar")
        out.findings.append(Finding(
            text=f"Of {len(blds)} mapped buildings: {counts['burned']} lie inside the burn scar, {counts['high']} are within 500 m of the fire and {counts['watch']} within 1 km"
                 + (f"; the nearest is {nearest:,.0f} m away." if nearest is not None else "."),
            topics=["building", "risk", "count", "summary"]))
    elif not blds:
        out.limitations.append("Asset exposure is not determinable: no building reference layer is available.")

    evac = ctx.layers.get("evacuation_zones") if ctx.layers else []
    if evac and (burn_geom is not None or front_geom is not None):
        for ez in evac:
            threat = unary_union([g for g in (burn_geom, front_geom) if g is not None])
            d = ctx.px_to_m(ez.geom.distance(threat))
            inter = ez.geom.intersection(threat).area
            out.metrics["evacuation_zone_distance_to_fire"] = Measurement(value=round(d or 0), unit="m", method="edge-to-edge")
            out.observations.append(make_obs(ctx, TOOL, "evac", "evacuation_zone", ez.geom, 0.9, severity="high" if (d or 0) < 800 else "moderate",
                                             attributes={"name": ez.props.get("name"), "distance_to_fire_m": round(d or 0), "overlap_ha": ctx.area_metric(inter, "").value}))
            out.findings.append(Finding(text=f"The {ez.props.get('name', 'evacuation zone')} boundary is {d:,.0f} m from the nearest fire/scar edge"
                                             + (f"; {ctx.area_metric(inter, '').value:,} {ctx.area_metric(inter, '').unit} of it is already burned." if inter else "."),
                                        topics=["risk", "evacuation", "summary"]))
        out.next_actions.append("Confirm evacuation-zone status with the incident commander before issuing any public alert.")

    out.confidence = float(np.mean([o.confidence for o in out.observations])) if out.observations else 0.6
    out.limitations.append("Image-derived estimate; not an official fire perimeter. Thermal sensors or ground reports should confirm active fire.")
    return out


def _dilate(m: np.ndarray, it: int) -> np.ndarray:
    from scipy import ndimage as ndi

    return ndi.binary_dilation(m, iterations=it)


def _bounds(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())] if len(xs) else [0, 0, 0, 0]


SPEC = dict(name=TOOL, description="Detect burn scars, active fire fronts and smoke; estimate burned area; assess building and evacuation-zone exposure.",
            args_model=NoArgs, fn=assess_wildfire, domains=(Domain.wildfire,))
