"""Crop-health assessment: NDVI per field, stress anomalies, trend and scouting priorities."""

from __future__ import annotations

import numpy as np

from geoai.geo.indices import greenness, vegetation_mask
from geoai.geo.regions import clean_mask, mask_to_polygons, rasterize
from geoai.schemas import Domain, Measurement
from geoai.tools.base import Finding, NoArgs, ToolContext, ToolOutput, make_obs, pct
from geoai.tools.common import plural

TOOL = "assess_crop_health"
STRESS_NDVI = 0.55
FALLOW_NDVI = 0.25
HEALTH_RANK = {"severe": 3, "stressed": 2, "watch": 1, "healthy": 0, "fallow": -1}


def classify_field(mean: float, stressed_frac: float, median: float) -> str:
    if median < FALLOW_NDVI:
        return "fallow"
    if stressed_frac >= 0.25 or mean < 0.45:
        return "severe"
    if stressed_frac >= 0.08 or mean < 0.62:
        return "stressed"
    if stressed_frac >= 0.02 or mean < 0.7:
        return "watch"
    return "healthy"


def assess_crop_health(ctx: ToolContext, _args: NoArgs) -> ToolOutput:
    out = ToolOutput()
    r = ctx.latest.raster
    h, w = ctx.shape
    gi = greenness(r)
    has_nir = r.has_nir
    index = "NDVI" if has_nir else "excess-green (visible-only proxy)"
    if not has_nir:
        out.limitations.append("No NIR band: vegetation vigour is estimated from visible bands (excess-green), which is far less sensitive than NDVI.")
        out.next_actions.append("Provide multispectral imagery (red + NIR) for reliable NDVI analysis.")
    if not ctx.georef:
        out.assumptions.append(ctx.not_georeferenced_note())
    before_gi = None
    if ctx.before is not None and ctx.before.raster.rgb.shape == r.rgb.shape and ctx.before.raster.has_nir == has_nir:
        before_gi = greenness(ctx.before.raster)
    elif ctx.before is not None:
        out.limitations.append("The earlier image is incompatible (size or bands), so trend analysis was skipped.")
    else:
        out.next_actions.append("Provide an earlier acquisition for time-series trend analysis.")

    fields = ctx.layers.get("fields") if ctx.layers else []
    thr = STRESS_NDVI if has_nir else None
    if not has_nir:
        veg = vegetation_mask(r)
        thr = float(np.percentile(gi[veg], 15)) if veg.any() else 0.0

    if not fields:
        veg = clean_mask(vegetation_mask(r), 400, close=3)
        low = veg & (gi < thr) if not has_nir else veg & (gi < STRESS_NDVI)
        low = clean_mask(low, 150, close=2)
        out.metrics["vegetated_share"] = Measurement(value=pct(veg.sum() / (h * w)), unit="%", method=index)
        out.metrics["stressed_area"] = ctx.area_metric(float(low.sum()), f"{index} below stress threshold within vegetation")
        for p in mask_to_polygons(low, 150)[:8]:
            out.observations.append(make_obs(ctx, TOOL, "anom", "stress_anomaly", p, 0.55 if not has_nir else 0.75, severity="moderate", kind="inferred"))
        out.limitations.append("No field-boundary layer: results are scene-level anomalies, not per-field summaries.")
        out.findings.append(Finding(text=f"{pct(veg.sum() / (h * w))}% of the scene is vegetated; {plural(len(mask_to_polygons(low, 150)), 'relatively stressed patch', 'relatively stressed patches')} found ({ctx.area_metric(float(low.sum()), '').value:,} {ctx.area_metric(float(low.sum()), '').unit}).",
                                    topics=["crop", "stress", "area", "summary"], kind="inferred"))
        out.confidence = 0.6
        return out

    rows = []
    stressed_total = veg_total = 0.0
    for f in fields:
        m = rasterize([f.geom.buffer(-3)], ctx.shape)
        vals = gi[m]
        if vals.size == 0:
            continue
        mean, median = float(vals.mean()), float(np.median(vals))
        stress_mask = m & (gi < (STRESS_NDVI if has_nir else thr)) & (median >= FALLOW_NDVI)
        sfrac = float(stress_mask.sum() / m.sum()) if median >= FALLOW_NDVI else 0.0
        cls = classify_field(mean, sfrac, median) if has_nir else ("stressed" if sfrac >= 0.15 else "watch" if sfrac >= 0.05 else "healthy")
        d_mean = None
        if before_gi is not None:
            d_mean = mean - float(before_gi[m].mean())
        area_ha = ctx.area_metric(float(m.sum()), "").value
        rows.append({"f": f, "mean": mean, "median": median, "sfrac": sfrac, "cls": cls, "d": d_mean, "stress_mask": stress_mask, "px": float(m.sum())})
        if cls != "fallow":
            veg_total += float(m.sum())
            stressed_total += float(stress_mask.sum())
        out.observations.append(make_obs(ctx, TOOL, "field", f"field_{cls}", f.geom, 0.9 if has_nir else 0.6,
                                         severity={"severe": "high", "stressed": "moderate", "watch": "low"}.get(cls, "info"),
                                         kind="observed" if has_nir else "inferred",
                                         attributes={"feature_id": f.id, "name": f.props.get("name"), "crop": f.props.get("crop"), "mean_ndvi" if has_nir else "mean_index": round(mean, 3),
                                                     "stressed_pct": pct(sfrac), "trend_ndvi": None if d_mean is None else round(d_mean, 3), "area_ha": area_ha}))
        if sfrac >= 0.02:
            for p in mask_to_polygons(clean_mask(stress_mask, 80, close=2), 80)[:2]:
                out.observations.append(make_obs(ctx, TOOL, "anom", "stress_anomaly", p, 0.85 if has_nir else 0.55, severity="high" if cls == "severe" else "moderate",
                                                 attributes={"feature_id": f.id, "crop": f.props.get("crop")}))
    cnt = {k: sum(r_["cls"] == k for r_ in rows) for k in HEALTH_RANK}
    out.metrics["fields_total"] = Measurement(value=len(rows), unit="count")
    for k in ("severe", "stressed", "watch", "healthy", "fallow"):
        out.metrics[f"fields_{k}"] = Measurement(value=cnt[k], unit="count")
    out.metrics["stressed_area"] = ctx.area_metric(stressed_total, f"{index} < {STRESS_NDVI if has_nir else round(thr, 2)} inside cropped fields")
    out.metrics["stressed_share_pct"] = Measurement(value=pct(stressed_total / veg_total) if veg_total else 0, unit="%", method="of cropped area")
    out.findings.append(Finding(
        text=f"{len(rows)} fields analysed with {index}: {cnt['healthy']} healthy, {cnt['watch']} on watch, {cnt['stressed']} stressed, {cnt['severe']} severely stressed and {cnt['fallow']} fallow. "
             f"Stressed crop covers {ctx.area_metric(stressed_total, '').value:,} {ctx.area_metric(stressed_total, '').unit} ({pct(stressed_total / veg_total) if veg_total else 0}% of cropped area).",
        topics=["crop", "stress", "area", "count", "summary"]))
    bad = sorted((r_ for r_ in rows if HEALTH_RANK[r_["cls"]] >= 1), key=lambda r_: (-HEALTH_RANK[r_["cls"]], -r_["sfrac"]))
    for r_ in bad[:4]:
        bits = [f"{r_['f'].props.get('name')} ({r_['f'].props.get('crop')}): {r_['cls']}, mean {'NDVI' if has_nir else 'index'} {r_['mean']:.2f}, {pct(r_['sfrac'])}% of the field below threshold"]
        if r_["d"] is not None:
            bits.append(f"change {r_['d']:+.2f} since the earlier pass")
        out.findings.append(Finding(text=", ".join(bits) + ".", topics=["crop", "stress", "field", "priority"], kind="observed" if has_nir else "inferred"))
    if before_gi is not None:
        declines = [r_ for r_ in rows if r_["d"] is not None and r_["d"] < -0.05 and r_["cls"] != "fallow"]
        worst = min((r_ for r_ in rows if r_["d"] is not None), key=lambda r_: r_["d"], default=None)
        out.findings.append(Finding(
            text=f"{plural(len(declines), 'field')} declined by more than 0.05 {('NDVI' if has_nir else 'index')} since the earlier pass"
                 + (f"; the steepest decline is {worst['f'].props.get('name')} ({worst['d']:+.2f})." if worst and worst["d"] < -0.05 else "."),
            topics=["change", "trend", "crop", "summary"]))
        out.metrics["fields_declining"] = Measurement(value=len(declines), unit="count", method="Δ mean NDVI < -0.05")
    if bad:
        order = ", ".join(r_["f"].props.get("name", r_["f"].id) for r_ in bad[:5])
        out.findings.append(Finding(text=f"Suggested scouting order: {order}. The spatial pattern is consistent with water, nutrient or disease problems — causes must be confirmed on the ground.",
                                    topics=["priority", "scouting", "summary"], kind="inferred"))
        out.next_actions.append("Scout the highest-priority fields and sample soil moisture/leaf tissue to confirm the cause.")
    if any(r_["cls"] == "severe" for r_ in rows):
        out.review_reasons.append("Severely stressed fields may trigger agronomic interventions; agronomist confirmation recommended.")
    out.limitations.append("Single-date indices cannot distinguish crop type, phenology or cause of stress without ground truth.")
    out.confidence = float(np.mean([o.confidence for o in out.observations])) if out.observations else 0.7
    return out


SPEC = dict(name=TOOL, description="Compute NDVI per field, detect stress anomalies, compare with an earlier pass and rank fields for scouting.",
            args_model=NoArgs, fn=assess_crop_health, domains=(Domain.agriculture,))
