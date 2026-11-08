"""Provider-neutral model gateway interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Literal

from geoai.llm.types import IntentDecision, Plan, VLMAnswer, VLMBox, VLMDescription
from geoai.schemas import Domain


class ModelGateway(ABC):
    mode: Literal["demo", "live"]
    models: dict[str, str]
    prompt_versions: dict[str, str]

    @abstractmethod
    def classify(self, question: str, domain: Domain, has_before: bool, tools: list[dict[str, Any]]) -> IntentDecision: ...

    def refine_plan(self, decision: IntentDecision, plan: Plan, tools: list[dict[str, Any]], question: str) -> Plan:
        return plan

    @abstractmethod
    def describe(self, png: bytes, width: int, height: int, context: dict[str, Any]) -> VLMDescription: ...

    @abstractmethod
    def answer(self, images: list[tuple[str, bytes]], question: str, width: int, height: int, context: dict[str, Any]) -> VLMAnswer: ...

    @abstractmethod
    def locate(self, png: bytes, query: str, width: int, height: int) -> list[VLMBox]: ...

    @abstractmethod
    def synthesize(self, question: str, evidence: dict[str, Any], draft: str) -> str: ...
