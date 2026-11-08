"""Public data contracts (request, structured response, provenance).

Coordinate conventions (never mixed silently):
  * ``bbox_px`` / ``pixel_polygon``: image pixel space, origin top-left, x right, y down,
    in the *source asset's* resolution.
  * ``geometry``: GeoJSON in WGS84 (EPSG:4326, lon/lat order). ``None`` unless the asset
    is georeferenced.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


def utcnow() -> datetime:
    return datetime.now(UTC)


class Domain(StrEnum):
    flood = "flood"
    wildfire = "wildfire"
    earthquake = "earthquake"
    urban = "urban"
    agriculture = "agriculture"


class Intent(StrEnum):
    describe_scene = "describe_scene"
    visual_question = "visual_question"
    locate_object = "locate_object"
    domain_assessment = "domain_assessment"
    change_analysis = "change_analysis"
    out_of_scope = "out_of_scope"


Kind = Literal["observed", "inferred", "not_determinable"]
Severity = Literal["info", "low", "moderate", "high", "critical"]


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=2, max_length=1000)
    domain: Domain = Domain.flood
    asset_id: str = Field(min_length=1, max_length=128, description="Primary (latest) image asset")
    before_asset_id: str | None = Field(default=None, max_length=128, description="Optional earlier image")
    mode: Literal["auto", "demo", "live"] = "auto"


class Evidence(BaseModel):
    asset_id: str
    region_px: list[float] | None = None  # [x0, y0, x1, y1]
    source_tool: str
    note: str | None = None


class Observation(BaseModel):
    id: str
    label: str
    kind: Kind = "observed"
    severity: Severity | None = None
    confidence: float = Field(ge=0, le=1)
    bbox_px: list[float] | None = None
    pixel_polygon: list[list[float]] | None = None
    geometry: dict[str, Any] | None = None
    area_m2: float | None = None
    area_px: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    evidence: list[Evidence] = Field(default_factory=list)


class Measurement(BaseModel):
    value: float | int | str | None
    unit: str | None = None
    method: str | None = None


class Source(BaseModel):
    asset_id: str
    role: Literal["latest", "before"] = "latest"
    captured_at: datetime | None = None
    sensor: str | None = None
    crs: str | None = None
    gsd_m: float | None = None
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    georeferenced: bool = False


class TraceStep(BaseModel):
    step: str
    status: Literal["running", "ok", "error", "skipped"] = "ok"
    detail: str | None = None
    tool: str | None = None
    duration_ms: int | None = None
    data: dict[str, Any] | None = None


class Provenance(BaseModel):
    pipeline_version: str
    mode: Literal["demo", "live"]
    models: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    request_sha256: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    record_sha256: str | None = None


class AnalysisResult(BaseModel):
    id: str
    status: Literal["queued", "running", "completed", "failed"] = "completed"
    created_at: datetime = Field(default_factory=utcnow)
    question: str
    domain: Domain
    intent: Intent | None = None
    mode: Literal["demo", "live"] = "demo"
    answer: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    metrics: dict[str, Measurement] = Field(default_factory=dict)
    observations: list[Observation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    needs_review: bool = False
    review_reasons: list[str] = Field(default_factory=list)
    review_status: Literal["not_required", "pending", "approved", "rejected", "corrected"] = "not_required"
    next_actions: list[str] = Field(default_factory=list)
    clarification: str | None = None
    error: str | None = None
    trace: list[TraceStep] = Field(default_factory=list)
    provenance: Provenance | None = None


class AssetInfo(BaseModel):
    id: str
    filename: str | None = None
    content_type: str
    width: int
    height: int
    bands: list[str]
    georeferenced: bool
    crs: str | None = None
    gsd_m: float | None = None
    captured_at: datetime | None = None
    bounds_wgs84: list[float] | None = None  # [west, south, east, north]
    sha256: str
    size_bytes: int
    sample: bool = False


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approved", "rejected", "corrected"]
    reviewer: str = Field(default="analyst", max_length=80)
    comment: str | None = Field(default=None, max_length=2000)
    corrections: list[dict[str, Any]] = Field(default_factory=list)
