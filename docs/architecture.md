# Architecture notes

## Agent graph

```
START → normalize ─┬→ classify ─┬→ plan → execute → validate → review_gate → synthesize → finalize → END
                   │            └→ respond_directly (clarification / out-of-scope) ─────────────────┘
                   └→ finalize (asset error → status=failed)
```

| Node | Responsibility |
|---|---|
| `normalize` | resolve assets (tenant-scoped), compare image pair dimensions, record assumptions (no CRS / timestamp) |
| `classify` | intent (`describe_scene`, `visual_question`, `locate_object`, `domain_assessment`, `change_analysis`, `out_of_scope`) + domain + target. Injection patterns are declined **before** any model call |
| `plan` | deterministic baseline plan → (live) LLM refinement → `validate_plan` (known tools, schema-valid args, ≤5 steps, measurement tools preserved) |
| `execute` | runs tools via the registry, emits `tool_start/tool_end` stream events; a failing tool becomes a limitation, not a crash |
| `validate` | drop invalid geometry, check time ordering / seasonal gap, detect model-vs-measurement contradictions |
| `review_gate` | low confidence, VLM-sourced footprints, contradictions, high-consequence findings → `needs_review` |
| `synthesize` | answer from structured findings only; LLM output is sanitised; deterministic fallback |
| `finalize` | assemble `AnalysisResult`, provenance (+ record hash), persist (append-only) |

## Tools (all typed, registry-bound)

`describe_scene` · `visual_qa` · `ground_object` · `assess_flood` · `assess_wildfire` · `assess_earthquake` ·
`assess_urban` · `assess_crop_health` · `weather_context`

Deterministic domain tools combine **raster measurements** (water/burn/fire/smoke/rubble/vegetation masks, NDVI/NDWI) with
**reference layers** (buildings, roads, parks, fields, zoning, evacuation zones) in a CRS-aware way (`geo/georef.py`).
When no layer exists, live mode asks the VLM for boxes and then measures overlap; demo mode reports
*not determinable* instead of guessing.

## Data contract

See `backend/src/geoai/schemas.py` (also mirrored in `frontend/src/types.ts`). Key rules:

* `bbox_px = [x0, y0, x1, y1]`, `pixel_polygon = [[x, y], …]` — image pixels, origin top-left, in the source asset's resolution.
* `geometry` — GeoJSON, EPSG:4326 (lon, lat). `null` when not georeferenced.
* `metrics[k] = {value, unit, method}`; areas are `ha`/`m²` only with a metric CRS, otherwise `px²`.
* `kind ∈ observed | inferred | not_determinable`; `severity ∈ info | low | moderate | high | critical`.

## Scaling path

| Concern | Today | Next |
|---|---|---|
| Execution | in-process thread pool + SSE | job queue (Celery/arq) + progress via the same SSE contract |
| Storage | local FS (`AssetService`) | S3-compatible object store behind the same interface |
| DB | SQLite / PostgreSQL | PostgreSQL + PostGIS for spatial queries |
| Rate limiting | in-process sliding window | ingress / Redis |
| Observability | JSON logs, request IDs, Prometheus | OpenTelemetry traces; LangSmith tracing works via standard env vars |
| Evaluation | ground-truth unit tests on synthetic scenes | versioned benchmark incl. real imagery (IoU, calibration, escalation rate) per `docs/plan.md` §12 |
