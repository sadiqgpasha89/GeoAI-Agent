"""Flood assessment: water extent, change vs. earlier image, and exposure of buildings / roads."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from geoai.geo.regions import mask_to_polygons
from geoai.schemas import Domain, Measurement
from geoai.tools.base import Finding, NoArgs, ToolContext, ToolOutput, make_obs, pct
from geoai.tools.common import (
    frac_inside,
    get_objects,
    in_scene,
    plural,
    runs_along,
    severity_rank,
    water_confidence,
    water_masks,
)

TOOL = "assess_flood"
CRITICAL_USES = {"clinic", "hospital", "school", "shelter", "fire_station"}


def assess_flood(ctx: ToolContext, _args: NoArgs) -> ToolOutput:
    out = ToolOutput()
    h, w = ctx.shape
    water, water_before = water_masks(ctx)
    total_px = float(h * w)
    px = float(water.sum())
    conf_water = water_confidence(ctx, water)

    out.metrics["flood_area"] = ctx.area_metric(px, "water index threshold (visible blue dominance / NDWI)")
    out.metrics["flood_cover_pct"] = Measurement(value=pct(px / total_px), unit="%", method="share of image pixels")
    if not ctx.georef:
        out.assumptions.append(ctx.not_georeferenced_note())
    if ctx.latest.raster.captured_at is None:
        out.assumptions.append("No acquisition timestamp was provided; the image is treated as the latest state.")

    # ---- extent ----------------------------------------------------------
    polys = mask_to_polygons(water, 120)
    for p in polys[:6]:
        out.observations.append(make_obs(ctx, TOOL, "water", "flood_extent", p, conf_water, severity="moderate"))
    if px == 0:
        out.findings.append(Finding(text="No standing or flood water was detected in the image.", topics=["flood", "extent", "summary"]))
    else:
        out.findings.append(Finding(
            text=f"Water covers {pct(px / total_px)}% of the scene ({_fmt_area(ctx, px)}), in {plural(len(polys), 'connected body', 'connected bodies')}.",
            topics=["flood", "extent", "area", "summary"]))

    # ---- change ----------------------------------------------------------
    if water_before is not None and water_before.shape == water.shape:
        new = water & ~water_before
        bpx, npx = float(water_before.sum()), float(new.sum())
        out.metrics["water_area_before"] = ctx.area_metric(bpx, "same method on the earlier image")
        out.metrics["new_flood_area"] = ctx.area_metric(npx, "water now AND NOT water before")
        growth = (px - bpx) / bpx * 100 if bpx else None
        if growth is not None:
            out.metrics["water_growth_pct"] = Measurement(value=round(growth, 1), unit="%", method="(after-before)/before")
        out.findings.append(Finding(
            text=f"Compared with the earlier image, water extent grew from {_fmt_area(ctx, bpx)} to {_fmt_area(ctx, px)}"
                 + (f" (+{growth:.0f}%)" if growth is not None else "") + f"; {_fmt_area(ctx, npx)} is newly inundated.",
            topics=["change", "flood", "area", "summary"]))
        ctx.memo["new_flood"] = new
    elif ctx.before is not None:
        out.limitations.append("The earlier image has different dimensions, so change detection was skipped.")
    else:
        out.next_actions.append("Provide a pre-flood image for before/after change analysis.")

    # ---- buildings -------------------------------------------------------
    blds, src = get_objects(ctx, "buildings", "building", "bld")
    crit_flooded = 0
    flooded_ids: list[str] = []
    if blds:
        n_flood = n_part = 0
        for f in blds:
            frac = frac_inside(water, f.geom)
            prior = frac_inside(water_before, f.geom) if water_before is not None else None
            if frac < 0.1:
                continue
            status = "flooded" if frac >= 0.5 else "partially_flooded"
            use = str(f.props.get("use", "building"))
            crit = use in CRITICAL_USES and status == "flooded"
            sev = "critical" if crit else ("high" if status == "flooded" else "moderate")
            conf = 0.55 + 0.4 * abs(2 * frac - 1)
            if f.source == "vlm":
                conf = min(conf, float(f.props.get("confidence", 0.6)))
            out.observations.append(make_obs(
                ctx, TOOL, "bld", f"{status}_building", f.geom, conf, severity=sev,
                attributes={"feature_id": f.id, "use": use, "water_overlap_pct": pct(frac), "source": f.source,
                            "flooded_before": None if prior is None else prior >= 0.1}))
            n_flood += status == "flooded"
            n_part += status == "partially_flooded"
            crit_flooded += crit
            if status == "flooded":
                flooded_ids.append(f.id)
        out.metrics["buildings_total"] = Measurement(value=len(blds), unit="count", method=f"{src} layer")
        out.metrics["buildings_flooded"] = Measurement(value=n_flood, unit="count", method="≥50% of footprint under water")
        out.metrics["buildings_partially_flooded"] = Measurement(value=n_part, unit="count", method="10–50% of footprint under water")
        where_txt = ""
        flooded_obs = [o for o in out.observations if o.label == "flooded_building"]
        if flooded_obs:
            xs = [(o.bbox_px[0] + o.bbox_px[2]) / 2 for o in flooded_obs]
            ys = [(o.bbox_px[1] + o.bbox_px[3]) / 2 for o in flooded_obs]
            where_txt = f", concentrated {in_scene([min(xs), min(ys), max(xs), max(ys)], w, h)}"
        out.findings.append(Finding(
            text=f"{plural(n_flood, 'building')} {'is' if n_flood == 1 else 'are'} flooded (≥50% of the footprint under water) and "
                 f"{n_part} more {'is' if n_part == 1 else 'are'} partially flooded, out of {len(blds)} mapped{where_txt}.",
            topics=["building", "count", "flood", "summary", "impact"]))
        if src == "vlm":
            out.limitations.append("Building footprints come from VLM localisation (no reference layer); overlap is measured, but footprints are approximate.")
            out.review_reasons.append("Building footprints were proposed by a vision model rather than an authoritative layer.")
        if crit_flooded:
            out.findings.append(Finding(text=f"{plural(crit_flooded, 'critical facility', 'critical facilities')} (clinic/school/shelter) {'is' if crit_flooded == 1 else 'are'} flooded.",
                                        topics=["building", "priority", "impact", "summary"]))
    else:
        out.limitations.append("Buildings are not determinable: no building reference layer is available and no vision model is configured.")
        out.findings.append(Finding(text="Building exposure could not be determined (no building footprints available).", kind="not_determinable", topics=["building", "count"]))
    out.metrics["critical_facilities_flooded"] = Measurement(value=crit_flooded, unit="count")

    # ---- roads -----------------------------------------------------------
    cut = 0
    roads = ctx.layers.get("roads") if ctx.layers else []
    if roads:
        total_len = flooded_len = 0.0
        road_msgs = []
        for rf in roads:
            if not isinstance(rf.geom, LineString):
                continue
            width_px = float(rf.props.get("width_m", 8)) / (ctx.georef.pixel_size_m or 1) if ctx.georef else 10
            runs, n, hits = runs_along(rf.geom, water, width_px)
            length = rf.geom.length
            fl = sum(b - a for a, b in runs)
            total_len += length
            flooded_len += fl
            for a, b in runs:
                seg = LineString([rf.geom.interpolate(a), *[rf.geom.interpolate(d) for d in np.linspace(a, b, max(2, int((b - a) / 6)))][1:]])
                run_m = ctx.px_to_m(b - a)
                impassable = (run_m or 0) >= 8 if ctx.georef else (b - a) > width_px
                cut += impassable
                out.observations.append(make_obs(
                    ctx, TOOL, "road", "flooded_road_segment", seg.buffer(width_px / 2, cap_style="flat"), 0.55 + 0.4 * (hits / max(n, 1) > 0.05),
                    severity="critical" if impassable else "high",
                    attributes={"road": rf.props.get("name"), "length_m": None if run_m is None else round(run_m, 1), "impassable": bool(impassable)}))
            if fl:
                lm = ctx.px_to_m(fl)
                road_msgs.append(f"{rf.props.get('name', rf.id)}: {100 * fl / length:.0f}% of its length ({lm:.0f} m) is inundated" if lm is not None
                                 else f"{rf.props.get('name', rf.id)}: {100 * fl / length:.0f}% inundated")
        out.metrics["road_total_length"] = ctx.length_metric(total_len, "sum of mapped centre-lines")
        out.metrics["road_flooded_length"] = ctx.length_metric(flooded_len, "centre-line samples with ≥60% of the carriageway under water")
        out.metrics["road_flooded_pct"] = Measurement(value=pct(flooded_len / total_len) if total_len else 0, unit="%")
        out.metrics["road_sections_cut"] = Measurement(value=cut, unit="count", method="continuous flooded run ≥ 8 m across the full carriageway")
        if road_msgs:
            out.findings.append(Finding(
                text="; ".join(road_msgs) + f". {plural(cut, 'continuous section')} would block traffic" + (" — a field check is recommended before reopening." if cut else "."),
                topics=["road", "impact", "summary", "access"]))
        else:
            out.findings.append(Finding(text="No mapped road segment is inundated.", topics=["road", "impact"]))
    else:
        out.limitations.append("Roads are not determinable: no road reference layer is available.")

    # ---- priorities & review --------------------------------------------
    crit = sorted((o for o in out.observations if o.severity in ("critical", "high")), key=lambda o: -severity_rank(o.severity))
    if crit:
        top = ", ".join(f"{o.attributes.get('road') or o.attributes.get('use') or o.label} ({o.attributes.get('feature_id') or o.id})" for o in crit[:4])
        out.findings.append(Finding(text=f"Suggested response order: {top}.", topics=["priority", "summary"], kind="inferred"))
        out.review_reasons.append("High-consequence findings (critical facilities or impassable roads) require analyst confirmation.")
        out.next_actions.append("Dispatch a field check to the highest-priority locations before reopening access.")
    out.confidence = float(np.mean([o.confidence for o in out.observations])) if out.observations else conf_water
    out.limitations.append("Image-derived estimate; not an official emergency measurement. Water under tree canopy or shadow may be missed.")
    return out


def _fmt_area(ctx: ToolContext, px: float) -> str:
    m = ctx.area_metric(px, "")
    return f"{m.value:,} {m.unit}"


SPEC = dict(name=TOOL, description="Measure flood extent from the image (water mask), compare with the earlier image, and intersect with buildings and roads.",
            args_model=NoArgs, fn=assess_flood, domains=(Domain.flood,))
