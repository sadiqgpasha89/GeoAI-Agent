"""Earthquake impact: per-building damage classes, blocked roads and inspection triage."""

from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from geoai.geo.indices import rubble_mask
from geoai.geo.regions import clean_mask, mask_to_polygons, rasterize
from geoai.schemas import Domain, Measurement
from geoai.tools.base import Finding, NoArgs, ToolContext, ToolOutput, make_obs, pct
from geoai.tools.common import frac_inside, get_objects, plural, runs_along

TOOL = "assess_earthquake"
WEIGHT = {"collapsed": 3, "damaged": 2, "intact": 0}


def classify_buildings(ctx: ToolContext, rub: np.ndarray, blds) -> list[dict]:
    rows = []
    for f in blds:
        core = f.geom.buffer(-2) if f.geom.buffer(-2).area > 20 else f.geom
        frac = frac_inside(rub, core)
        delta = None
        if ctx.before is not None and ctx.before.raster.rgb.shape == ctx.latest.raster.rgb.shape:
            m = rasterize([core], ctx.shape)
            if m.any():
                delta = float(np.abs(ctx.latest.raster.rgb[m] - ctx.before.raster.rgb[m]).mean())
        cls = "collapsed" if frac >= 0.7 else "damaged" if frac >= 0.12 else "intact"
        conf = 0.55 + 0.4 * min(1.0, abs(frac - (0.7 if cls == "collapsed" else 0.12)) * 3 + (0.3 if cls == "intact" and frac < 0.04 else 0))
        rows.append({"feat": f, "cls": cls, "rubble_frac": frac, "delta": delta, "conf": float(np.clip(conf, 0.5, 0.95))})
    return rows


def assess_earthquake(ctx: ToolContext, _args: NoArgs) -> ToolOutput:
    out = ToolOutput()
    h, w = ctx.shape
    rub = ctx.cached("rubble", lambda: clean_mask(rubble_mask(ctx.latest.raster), 30, close=1))
    if not ctx.georef:
        out.assumptions.append(ctx.not_georeferenced_note())
    if ctx.before is None:
        out.limitations.append("No pre-event image: damage is inferred from surface texture (debris) only, not from change.")
        out.next_actions.append("Provide a pre-event image for per-building change analysis.")
    out.metrics["debris_area"] = ctx.area_metric(float(rub.sum()), "high local texture on non-vegetated surface")

    blds, src = get_objects(ctx, "buildings", "collapsed or damaged building", "bld")
    if blds:
        rows = classify_buildings(ctx, rub, blds)
        counts = {k: sum(r["cls"] == k for r in rows) for k in WEIGHT}
        for r in rows:
            if r["cls"] == "intact":
                continue
            f = r["feat"]
            sev = "critical" if r["cls"] == "collapsed" else "high"
            out.observations.append(make_obs(ctx, TOOL, "bld", f"{r['cls']}_building", f.geom, r["conf"], severity=sev, attributes={
                "feature_id": f.id, "use": f.props.get("use"), "rubble_pct": pct(r["rubble_frac"]),
                "change_vs_before": None if r["delta"] is None else round(r["delta"], 3), "source": f.source}))
        out.metrics["buildings_total"] = Measurement(value=len(rows), unit="count", method=f"{src} layer")
        for k, v in counts.items():
            out.metrics[f"buildings_{k}"] = Measurement(value=v, unit="count")
        dmg_pct = (counts["collapsed"] + counts["damaged"]) / max(len(rows), 1)
        out.findings.append(Finding(
            text=f"Of {len(rows)} mapped buildings, {counts['collapsed']} appear collapsed, {counts['damaged']} damaged and {counts['intact']} intact "
                 f"({100 * dmg_pct:.0f}% with visible damage).", topics=["building", "count", "summary", "damage"]))
        bad = [r for r in rows if r["cls"] != "intact"]
        if bad:
            out.findings.append(Finding(text="Collapsed: " + (", ".join(r["feat"].id for r in rows if r["cls"] == "collapsed") or "none")
                                             + "; damaged: " + (", ".join(r["feat"].id for r in rows if r["cls"] == "damaged") or "none") + ".",
                                        topics=["building", "damage"]))
        # triage: collapsed/commercial first, then by rubble share
        ranked = sorted(bad, key=lambda r: (-WEIGHT[r["cls"]], -(r["feat"].props.get("use") == "commercial"), -r["rubble_frac"]))
        if ranked:
            order = ", ".join(f"{r['feat'].id} ({r['cls']})" for r in ranked[:6])
            out.findings.append(Finding(text=f"Suggested field-inspection order: {order}.", topics=["priority", "triage", "summary"], kind="inferred"))
            out.next_actions.append("Send structural engineers to the collapsed buildings first; collapsed structures may have trapped occupants.")
            out.review_reasons.append("Collapsed-structure findings are safety-critical and require analyst confirmation.")
        if src == "vlm":
            out.limitations.append("Building footprints come from VLM proposals; rubble share was verified deterministically but footprints are approximate.")
    else:
        for p in mask_to_polygons(rub, 150)[:6]:
            out.observations.append(make_obs(ctx, TOOL, "debris", "debris_area", p, 0.7, severity="high"))
        out.limitations.append("Per-building damage is not determinable: no building footprint layer is available.")
        out.findings.append(Finding(text=f"{plural(len(mask_to_polygons(rub, 150)), 'debris-like region')} detected; per-building damage could not be determined without footprints.",
                                    kind="not_determinable", topics=["building", "damage", "summary"]))

    # ---- roads -----------------------------------------------------------
    roads = ctx.layers.get("roads") if ctx.layers else []
    blocked_total = 0
    msgs = []
    for rf in roads:
        if not isinstance(rf.geom, LineString):
            continue
        width_px = float(rf.props.get("width_m", 8)) / (ctx.georef.pixel_size_m or 1) if ctx.georef else 10
        runs, n, hits = runs_along(rf.geom, rub, width_px, min_share=0.5)
        for a, b in runs:
            run_m = ctx.px_to_m(b - a)
            if (run_m or 0) < 2.5 and ctx.georef:
                continue
            seg = LineString([rf.geom.interpolate(d) for d in np.linspace(a, b, max(2, int((b - a) / 4)))])
            blocked_total += 1
            out.observations.append(make_obs(ctx, TOOL, "road", "blocked_road_segment", seg.buffer(width_px / 2, cap_style="flat"), 0.8, severity="critical",
                                             attributes={"road": rf.props.get("name"), "length_m": None if run_m is None else round(run_m, 1)}))
            msgs.append(f"{rf.props.get('name', rf.id)} (~{run_m:.0f} m)" if run_m is not None else rf.props.get("name", rf.id))
    if roads:
        out.metrics["roads_blocked_sections"] = Measurement(value=blocked_total, unit="count", method="debris covering ≥50% of the carriageway over ≥ 2.5 m")
        out.findings.append(Finding(
            text=(f"{plural(blocked_total, 'road section')} blocked by debris: " + "; ".join(msgs) + ".") if blocked_total else "No road section is blocked by debris.",
            topics=["road", "access", "summary"]))
        if blocked_total:
            out.next_actions.append("Route emergency vehicles around blocked sections and dispatch clearance crews.")
            out.review_reasons.append("Road blockages affect emergency routing and require analyst confirmation.")
    else:
        out.limitations.append("Road blockages are not determinable: no road reference layer is available.")

    out.confidence = float(np.mean([o.confidence for o in out.observations])) if out.observations else 0.7
    out.limitations.append("Image-based triage only; structural safety must be confirmed by qualified engineers on site.")
    return out


SPEC = dict(name=TOOL, description="Classify building damage (intact/damaged/collapsed) and detect road blockages from debris; rank buildings for field inspection.",
            args_model=NoArgs, fn=assess_earthquake, domains=(Domain.earthquake,))
