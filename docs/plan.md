# GeoAI Agent — Enhanced Implementation Plan

## 1. Product intent

Build an agentic geospatial intelligence assistant that combines large language models
(LLMs), vision-language models (VLMs), geospatial data, and specialist tools to answer
questions about satellite/aerial imagery and produce evidence-backed operational insights.

The reference screenshot defines the core interaction pattern:

- Describe an image in detail.
- Answer visual questions about a scene.
- Locate a requested object with a bounding box.
- Assess flood impact, including affected roads and buildings.

The use case expands this foundation to five domains:

1. Flood monitoring
2. Wildfire monitoring
3. Earthquake impact monitoring
4. Urban planning
5. Agricultural crop-health assessment

## 2. Target users and outcomes

### Primary users

- Emergency operations and disaster-response teams
- Municipal and urban-planning teams
- Agriculture and crop-monitoring analysts
- GIS analysts and researchers

### Expected outcomes

- Faster first-pass interpretation of imagery
- Consistent, auditable answers grounded in image and geographic evidence
- Human-reviewable detections, measurements, and confidence
- Reusable reports and machine-readable outputs for downstream GIS systems

## 3. Product principles

- **Evidence first:** every material conclusion links to an image region, layer,
  timestamp, source, or calculation.
- **Agentic, not opaque:** the agent plans and invokes tools, while deterministic
  geospatial operations perform measurements and comparisons.
- **Human-in-the-loop for impact decisions:** low-confidence or high-consequence
  findings require review.
- **Temporal awareness:** always preserve acquisition time and compare compatible
  imagery where possible.
- **Graceful uncertainty:** distinguish observed, inferred, and unavailable facts.
- **Privacy and security by design:** protect uploaded imagery, credentials, and
  location-sensitive data.

## 4. Proposed system architecture

```text
Web/API client
    |
    v
Session and project service
    |
    v
GeoAI orchestrator (planner/router)
    |-- image understanding tools (VLM: captioning, VQA, grounding)
    |-- geospatial tools (CRS, geometry, distance, area, zonal statistics)
    |-- raster tools (band math, masks, change detection, tiling)
    |-- data tools (catalog search, weather, alerts, basemaps)
    |-- report/export tools (GeoJSON, GeoTIFF, CSV, PDF/JSON)
    |
    v
Evidence and provenance store
    |-- original assets and derived layers
    |-- model/tool outputs
    |-- prompts, versions, timestamps, confidence
    |-- human corrections and approvals
```

### Recommended component boundaries

- **Frontend:** map-first workspace with image viewer, chat, layer controls,
  bounding-box/polygon overlays, timeline, and export actions.
- **API:** upload/session/project endpoints, streaming agent responses, job status,
  and artifact retrieval.
- **Orchestrator:** intent classification, task decomposition, tool selection,
  validation, confidence aggregation, and response synthesis.
- **Model gateway:** provider-neutral adapters for VLMs, LLMs, embeddings, and
  optional object-detection/segmentation models.
- **Geospatial engine:** raster/vector processing, reprojection, geometry operations,
  and deterministic calculations.
- **Persistence:** object storage for imagery, relational metadata, and a spatial
  database or spatially enabled query layer.
- **Async workers:** large image processing, temporal analysis, report generation,
  and batch monitoring jobs.

## 5. Agent workflow

1. **Receive request and context**
   - User question, uploaded image or area of interest, date range, and domain.
2. **Normalize inputs**
   - Validate file type, dimensions, CRS, coordinates, timestamps, and permissions.
3. **Classify intent**
   - Grounding/VQA, object localization, change detection, impact assessment,
     planning analysis, or crop-health assessment.
4. **Create an execution plan**
   - Select the minimum tools and data needed; ask a clarifying question when the
     area, date, sensor, or requested object is ambiguous.
5. **Execute specialist tools**
   - Keep raw observations separate from model-generated interpretation.
6. **Validate results**
   - Check geometry bounds, units, temporal compatibility, confidence thresholds,
     and contradictions between tools.
7. **Request review when needed**
   - Escalate uncertain detections, safety-relevant conclusions, or destructive
     recommendations.
8. **Synthesize answer**
   - Return a concise answer, evidence overlays, confidence, assumptions,
     timestamps, and suggested next actions.
9. **Persist provenance**
   - Store inputs, tool calls, model versions, evidence references, and feedback.

## 6. Domain capabilities

### 6.1 Flood monitoring

**MVP**

- Detect and describe visible inundation.
- Answer whether roads/buildings/fields are affected.
- Ground affected objects with boxes or polygons.
- Estimate affected area when imagery and scale permit.

**Next**

- Before/after water extent comparison.
- Water-index or segmentation-assisted mapping.
- Exposure intersection with roads, buildings, critical infrastructure, and
  population layers.
- Severity tiers and prioritized response locations.

### 6.2 Wildfire monitoring

**MVP**

- Identify visible smoke, burn scars, and fire-related scene changes.
- Ground smoke/fire/burn regions with confidence and timestamp.

**Next**

- Compare pre-fire and post-fire imagery.
- Integrate weather and wind context.
- Estimate burned area and intersect with assets or evacuation zones.

### 6.3 Earthquake monitoring

**MVP**

- Identify visible structural damage, debris, blocked roads, and collapsed areas.
- Return per-object evidence and confidence.

**Next**

- Pre/post change detection.
- Building and road exposure summaries.
- Triage ranking for field inspection.

### 6.4 Urban planning

**MVP**

- Describe land use and visible infrastructure.
- Locate buildings, roads, open space, parking, and construction activity.

**Next**

- Land-use change over time.
- Walkability/accessibility and service-area analysis.
- Scenario comparison with zoning or planning layers.

### 6.5 Agriculture crop-health assessment

**MVP**

- Describe field boundaries and visible crop condition.
- Identify anomalous or stressed regions when imagery supports it.

**Next**

- Multispectral index analysis (for example NDVI/NDRE where bands exist).
- Time-series trend analysis.
- Field-level summaries, anomaly polygons, and scouting recommendations.

## 7. MVP scope

Deliver one coherent vertical slice before implementing all domains:

1. Upload one or more georeferenced or ordinary aerial/satellite images.
2. Chat with the image using VQA and detailed scene description.
3. Request an object and receive a bounding box or “not found” result.
4. Run flood assessment for roads and buildings.
5. Show overlays and evidence in a map/image viewer.
6. Return structured JSON plus a readable answer.
7. Preserve provenance and confidence for every result.

The MVP should support an explicit **demo mode** using bundled sample imagery so the
workflow can be tested without external data-provider credentials.

## 8. Structured response contract

Every agent result should be representable as:

```json
{
  "answer": "Three buildings appear flooded near the top-left of the image.",
  "observations": [
    {
      "label": "flooded_building",
      "geometry": {"type": "Polygon", "coordinates": []},
      "confidence": 0.82,
      "evidence": [{"asset_id": "image-1", "region": [0, 0, 0, 0]}]
    }
  ],
  "assumptions": ["No acquisition timestamp was provided."],
  "sources": [{"asset_id": "image-1", "captured_at": null}],
  "needs_review": false,
  "next_actions": ["Provide a pre-flood image for change analysis."]
}
```

Coordinates should use a clearly documented convention. Keep pixel coordinates and
geospatial coordinates distinct; never silently treat one as the other.

## 9. Tool and model strategy

- Start with one production-capable VLM for captioning, VQA, and grounding.
- Use a separate deterministic geospatial toolkit for geometry and measurements.
- Add specialized detectors/segmenters only when evaluation shows the general VLM
  is insufficient.
- Use an LLM for planning and synthesis, not for area, distance, CRS, or geometry
  calculations.
- Add model routing by task, cost, latency, image size, and required precision.
- Record model name, version, prompt/template version, and inference parameters.

## 10. Data and integration strategy

### Initial inputs

- User-uploaded JPG/PNG/GeoTIFF imagery
- Optional point, bounding-box, polygon, CRS, and acquisition timestamp
- Optional before/after image pair

### Later integrations

- Satellite imagery catalog/search provider
- Weather, wildfire, earthquake, and flood alert feeds
- OpenStreetMap or authoritative roads/buildings/infrastructure layers
- Digital elevation and land-cover datasets

External integrations must be isolated behind adapters so the core agent remains
testable with local fixtures and demo data.

## 11. Safety, quality, and governance

- Clearly label findings as **observed**, **inferred**, or **not determinable**.
- Do not present an image-only estimate as an official emergency measurement.
- Require confirmation before publishing alerts or triggering external actions.
- Apply input validation, upload limits, malware scanning, access control, and
  tenant/project isolation.
- Redact or restrict sensitive imagery and precise locations in logs.
- Keep immutable provenance for generated reports.
- Provide user correction/feedback and use it for evaluation, not silent retraining.

## 12. Evaluation plan

Create a small versioned benchmark covering all five domains and the screenshot
interaction patterns.

### Core metrics

- VQA answer correctness and groundedness
- Bounding-box IoU / localization precision
- Segmentation IoU or area error where applicable
- Object count accuracy
- Flood/building/road classification precision and recall
- Calibration of confidence and escalation rate
- Tool-plan success rate
- Latency, cost, and failure recovery

### Required test categories

- Clear and ambiguous images
- No-object and partially occluded cases
- Different image sizes and aspect ratios
- Missing CRS/timestamp metadata
- Contradictory or unavailable external data
- Prompt injection text embedded in an image
- Unsafe or overconfident emergency conclusions

## 13. Delivery phases

### Phase 0 — Foundation

- Confirm product decisions, target users, deployment constraints, and model/provider
  availability.
- Choose initial stack and define response/provenance schemas.
- Assemble demo images and annotation format.

### Phase 1 — Multimodal core

- Implement upload and asset metadata.
- Add image description, VQA, and grounding tools.
- Build the initial orchestrator and structured response validation.
- Add demo-mode fixtures and automated tests.

### Phase 2 — Flood vertical slice

- Add flood assessment workflow.
- Add road/building overlays and deterministic geometry operations.
- Add confidence thresholds, review states, and export.

### Phase 3 — Geospatial and temporal intelligence

- Add CRS-aware processing, before/after comparison, raster indexes, and catalog
  adapters.
- Add asynchronous jobs and progress updates.

### Phase 4 — Domain expansion

- Add wildfire, earthquake, urban planning, and crop-health specialist tools.
- Add domain-specific prompts, validators, datasets, and evaluation sets.

### Phase 5 — Production hardening

- Authentication/authorization, quotas, observability, audit logs, retention,
  deployment automation, load testing, and incident procedures.

## 14. Definition of done for the first implementation

- A user can upload an image and ask the three screenshot-style questions.
- The agent returns readable answers and validated structured observations.
- Bounding boxes render correctly and use the documented coordinate system.
- Flood answers include uncertainty and evidence references.
- Invalid files, missing metadata, model failures, and unavailable tools surface
  explicit user-facing errors.
- Demo-mode tests run without network credentials.
- A README documents setup, run commands, configuration, and known limitations.

## 15. Decisions to confirm before implementation

- Preferred runtime and UI stack (Python, TypeScript, or another existing standard).
- Hosted versus local model execution and acceptable data residency.
- Initial imagery source and whether georeferenced GeoTIFF support is required in
  the MVP.
- Whether the first demo should optimize for a polished UI or an API/notebook.
- Required deployment target, authentication model, and expected scale.
