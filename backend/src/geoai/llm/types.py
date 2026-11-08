"""Types exchanged with the model gateway."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from geoai.schemas import Domain, Intent


class IntentDecision(BaseModel):
    intent: Intent
    domain: Domain
    target: str | None = None  # object to localise, if any
    clarification: str | None = None  # question back to the user when the request is ambiguous
    rationale: str = ""
    topics: list[str] = Field(default_factory=list)


class PlanStep(BaseModel):
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    why: str = ""


class Plan(BaseModel):
    steps: list[PlanStep]
    rationale: str = ""


class VLMBox(BaseModel):
    label: str
    bbox_px: list[float]  # [x0, y0, x1, y1] in source pixels
    confidence: float = Field(ge=0, le=1, default=0.5)


class VLMDescription(BaseModel):
    description: str
    objects: list[VLMBox] = Field(default_factory=list)


class VLMAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0, le=1, default=0.5)
    boxes: list[VLMBox] = Field(default_factory=list)
