"""Machine-readable and human-readable exports of an analysis result."""

from __future__ import annotations

import csv
import html
import io
import json

from geoai.schemas import AnalysisResult


def to_geojson(res: AnalysisResult) -> dict:
    feats = []
    for o in res.observations:
        props = {"id": o.id, "label": o.label, "kind": o.kind, "severity": o.severity, "confidence": o.confidence,
                 "area_m2": o.area_m2, "bbox_px": o.bbox_px, **{f"attr_{k}": v for k, v in o.attributes.items()}}
        feats.append({"type": "Feature", "id": o.id, "properties": props, "geometry": o.geometry})
    return {"type": "FeatureCollection", "name": f"geoai-{res.id}", "features": feats,
            "properties": {"analysis_id": res.id, "question": res.question, "crs": "EPSG:4326 (features with null geometry are not georeferenced; see bbox_px)",
                           "needs_review": res.needs_review, "review_status": res.review_status}}


def to_csv(res: AnalysisResult) -> str:
    buf = io.StringIO()
    cols = ["id", "label", "kind", "severity", "confidence", "area_m2", "area_px", "bbox_x0", "bbox_y0", "bbox_x1", "bbox_y1", "attributes"]
    w = csv.writer(buf)
    w.writerow(cols)
    for o in res.observations:
        b = o.bbox_px or [None] * 4
        w.writerow([_cell(o.id), _cell(o.label), o.kind, o.severity or "", o.confidence, o.area_m2 or "", o.area_px or "", *b, _cell(json.dumps(o.attributes, default=str))])
    return buf.getvalue()


def _cell(v: object) -> str:
    """Neutralise spreadsheet formula injection."""
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@", "\t", "\r") else s


def to_html(res: AnalysisResult) -> str:
    e = html.escape
    rows = "".join(
        f"<tr><td>{e(o.id)}</td><td>{e(o.label)}</td><td>{e(o.kind)}</td><td>{e(o.severity or '')}</td><td>{o.confidence:.0%}</td>"
        f"<td>{'' if o.area_m2 is None else f'{o.area_m2:,.0f} m²'}</td><td>{e(str(o.bbox_px))}</td></tr>" for o in res.observations)
    metrics = "".join(f"<tr><td>{e(k)}</td><td>{e(str(m.value))} {e(m.unit or '')}</td><td>{e(m.method or '')}</td></tr>" for k, m in res.metrics.items())
    li = lambda items: "".join(f"<li>{e(i)}</li>" for i in items) or "<li>—</li>"  # noqa: E731
    prov = res.provenance
    review = f"<p><strong>Analyst review:</strong> {e(res.review_status)}" + (f" — {e('; '.join(res.review_reasons))}" if res.review_reasons else "") + "</p>"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>GeoAI report {e(res.id)}</title>
<style>body{{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2rem auto;padding:0 1rem;color:#111}}table{{border-collapse:collapse;width:100%;margin:.5rem 0 1.5rem}}
td,th{{border:1px solid #ccc;padding:.3rem .5rem;text-align:left;font-size:12px}}th{{background:#f2f2f2}}.note{{background:#fff7e0;border:1px solid #e6c36a;padding:.6rem;border-radius:6px}}</style></head><body>
<h1>GeoAI analysis report</h1><p><code>{e(res.id)}</code> · {e(res.domain.value)} · mode: {e(res.mode)} · {res.created_at.isoformat()}</p>
<p class="note">Image-derived estimate. Not an official emergency measurement; confirm with ground truth before operational use.</p>
<h2>Question</h2><p>{e(res.question)}</p><h2>Answer</h2><p>{e(res.answer)}</p>{review}
<p>Overall confidence: {'n/a' if res.confidence is None else f'{res.confidence:.0%}'}</p>
<h2>Measurements</h2><table><tr><th>Metric</th><th>Value</th><th>Method</th></tr>{metrics}</table>
<h2>Observations</h2><table><tr><th>ID</th><th>Label</th><th>Kind</th><th>Severity</th><th>Confidence</th><th>Area</th><th>bbox (px)</th></tr>{rows}</table>
<h2>Assumptions</h2><ul>{li(res.assumptions)}</ul><h2>Limitations</h2><ul>{li(res.limitations)}</ul><h2>Next actions</h2><ul>{li(res.next_actions)}</ul>
<h2>Sources</h2><ul>{li(f'{s.role}: {s.asset_id} · {s.captured_at or "no timestamp"} · sha256 {s.sha256 or "n/a"}' for s in res.sources)}</ul>
<h2>Provenance</h2><p>Pipeline {e(prov.pipeline_version) if prov else ''} · models {e(json.dumps(prov.models)) if prov else ''}<br>
Record SHA-256: <code>{e(prov.record_sha256 or '') if prov else ''}</code></p></body></html>"""
