"""Urban planning: land cover, change, zoning conflicts and park accessibility."""

from __future__ import annotations

import numpy as np
from shapely.ops import unary_union

from geoai.geo.indices import bare_soil_mask, builtup_mask, vegetation_mask, water_mask
from geoai.geo.regions import clean_mask, mask_to_polygons, rasterize
from geoai.schemas import Domain, Measurement
from geoai.tools.base import Finding, NoArgs, ToolContext, ToolOutput, make_obs, pct
from geoai.tools.common import get_objects, plural

TOOL = "assess_urban"
WALK_M = 400.0


def landcover(raster) -> dict[str, np.ndarray]:
    return {"vegetation": vegetation_mask(raster), "water": water_mask(raster), "bare": bare_soil_mask(raster), "built": builtup_mask(raster)}


def assess_urban(ctx: ToolContext, _args: NoArgs) -> ToolOutput:
    out = ToolOutput()
    h, w = ctx.shape
    total = float(h * w)
    lc = ctx.cached("lc_now", lambda: landcover(ctx.latest.raster))
    if not ctx.georef:
        out.assumptions.append(ctx.not_georeferenced_note())
    for k, m in lc.items():
        out.metrics[f"{k}_share"] = Measurement(value=pct(m.sum() / total), unit="%", method="rule-based land-cover classification")
    out.findings.append(Finding(
        text=f"Land cover: {pct(lc['built'].sum() / total)}% built/impervious, {pct(lc['vegetation'].sum() / total)}% vegetation, "
             f"{pct(lc['bare'].sum() / total)}% bare ground and {pct(lc['water'].sum() / total)}% water.", topics=["landuse", "summary"]))

    new_built = veg_loss = construction = None
    if ctx.before is not None and ctx.before.raster.rgb.shape == ctx.latest.raster.rgb.shape:
        lb = ctx.cached("lc_before", lambda: landcover(ctx.before.raster))
        new_built = clean_mask(lc["built"] & ~lb["built"] & ~lb["water"], 150, close=1)
        veg_loss = clean_mask(lb["vegetation"] & ~lc["vegetation"], 150, close=5)
        construction = clean_mask(lc["bare"] & ~lb["bare"], 150, close=1)
        for name, m, label, sev in (("new_built", new_built, "new_development", "info"), ("veg_loss", veg_loss, "green_space_loss", "moderate"),
                                    ("constr", construction, "construction_site", "info")):
            polys = mask_to_polygons(m, 150)
            out.metrics[f"{label}_area"] = ctx.area_metric(float(m.sum()), f"{label} = class change vs earlier image")
            for p in polys[:6]:
                out.observations.append(make_obs(ctx, TOOL, name, label, p, 0.82, severity=sev))
        bn = ctx.area_metric(float(new_built.sum()), "")
        out.findings.append(Finding(
            text=f"Since the earlier image, {bn.value:,} {bn.unit} of new built-up area appeared ({plural(len(mask_to_polygons(new_built, 150)), 'patch', 'patches')}), "
                 f"{ctx.area_metric(float(veg_loss.sum()), '').value:,} {bn.unit} of vegetation was lost and "
                 f"{ctx.area_metric(float(construction.sum()), '').value:,} {bn.unit} of new bare ground (likely construction) appeared.",
            topics=["change", "landuse", "area", "summary"]))
    else:
        out.next_actions.append("Provide an earlier image to quantify land-use change.")

    # ---- zoning conflicts ------------------------------------------------
    zones = ctx.layers.get("zoning") if ctx.layers else []
    if zones and new_built is not None:
        change = new_built | veg_loss | construction
        conflicts = 0
        for z in zones:
            if z.props.get("zone") not in ("park", "reserved", "water"):
                continue
            m = rasterize([z.geom], ctx.shape)
            share = float((change & m).sum() / max(m.sum(), 1))
            if share >= 0.1:
                conflicts += 1
                out.observations.append(make_obs(ctx, TOOL, "zone", "zoning_conflict", z.geom, 0.85, severity="high", kind="inferred",
                                                 attributes={"zone": z.props["zone"], "changed_share_pct": pct(share), "zone_id": z.id}))
        out.metrics["zoning_conflicts"] = Measurement(value=conflicts, unit="count", method="≥10% of a park/reserved/water zone changed")
        out.findings.append(Finding(
            text=(f"{plural(conflicts, 'protected zone')} {'shows' if conflicts == 1 else 'show'} conversion to built or bare land, which conflicts with the zoning layer: "
                  + ", ".join(o.attributes["zone_id"] + f" ({o.attributes['zone']}, {o.attributes['changed_share_pct']}% changed)" for o in out.observations if o.label == "zoning_conflict") + ".")
            if conflicts else "No change was detected inside park, reserved or water zones.",
            topics=["zoning", "conflict", "summary"], kind="inferred"))
        if conflicts:
            out.review_reasons.append("Possible zoning violations should be verified against permits before enforcement.")
            out.next_actions.append("Cross-check flagged parcels against building permits and the current zoning register.")
    elif not zones:
        out.limitations.append("Zoning conflicts are not determinable: no zoning layer is available.")

    # ---- accessibility ---------------------------------------------------
    blds, src = get_objects(ctx, "buildings", "building", "bld")
    parks = ctx.layers.get("parks") if ctx.layers else []
    park_geoms = [p.geom for p in parks]
    park_src = "parks layer"
    if not park_geoms:
        park_geoms = mask_to_polygons(clean_mask(lc["vegetation"], 400, close=3), 1500)
        park_src = "large vegetated patches (inferred open space)"
    if blds and park_geoms and ctx.georef:
        merged = unary_union(park_geoms)
        thr = ctx.m_to_px(WALK_M) or 0
        near = [f for f in blds if f.geom.distance(merged) <= thr]
        out.metrics["buildings_within_400m_of_park"] = Measurement(value=len(near), unit="count", method=f"straight-line distance, {park_src}")
        out.metrics["park_access_pct"] = Measurement(value=pct(len(near) / len(blds)), unit="%")
        out.findings.append(Finding(
            text=f"{len(near)} of {len(blds)} buildings ({pct(len(near) / len(blds))}%) are within {WALK_M:.0f} m (straight-line) of a park or open space [{park_src}].",
            topics=["park", "accessibility", "count", "summary"]))
        if park_src.startswith("large"):
            out.limitations.append("Parks were inferred from vegetation patches; supply a parks layer for authoritative accessibility.")
    for p in parks[:12]:
        out.observations.append(make_obs(ctx, TOOL, "park", "park", p.geom, 0.9, attributes={"name": p.props.get("name")}))
    out.limitations.append("Straight-line distances ignore street networks; use network routing for formal walkability studies.")
    out.confidence = float(np.mean([o.confidence for o in out.observations])) if out.observations else 0.7
    return out


SPEC = dict(name=TOOL, description="Classify land cover, quantify change vs an earlier image, flag zoning conflicts and measure park accessibility.",
            args_model=NoArgs, fn=assess_urban, domains=(Domain.urban,))
