"""Live gateway: routing, VLM perception and synthesis through an LLM backend."""

from __future__ import annotations

import json
import logging
from typing import Any

from pydantic import ValidationError

from geoai.agent.compose import INJECTION_RE, rule_classify
from geoai.llm import prompts
from geoai.llm.backends import LLMBackend, LLMError
from geoai.llm.gateway import ModelGateway
from geoai.llm.sanitize import clean_text, extract_json
from geoai.llm.types import IntentDecision, Plan, VLMAnswer, VLMBox, VLMDescription
from geoai.schemas import Domain, Intent
from geoai.tools.base import ToolError

log = logging.getLogger("geoai.llm")


class LiveGateway(ModelGateway):
    mode = "live"

    def __init__(self, backend: LLMBackend, fast_model: str, vision_model: str, provider: str):
        self.backend = backend
        self.fast_model = fast_model
        self.vision_model = vision_model
        self.models = {"routing": fast_model, "vision": vision_model, "synthesis": vision_model, "provider": provider}
        self.prompt_versions = dict(prompts.PROMPT_VERSIONS)

    # ---- helpers ---------------------------------------------------------
    def _json(self, *, model: str, system: str, user: str, images=None, max_tokens=1500, retries: int = 1) -> Any:
        last: Exception | None = None
        for attempt in range(retries + 1):
            prompt = user if attempt == 0 else user + "\n\nYour previous reply was not valid JSON. Reply with ONLY the JSON object."
            try:
                return extract_json(self.backend.complete(model=model, system=system, user=prompt, images=images, max_tokens=max_tokens))
            except (ValueError, LLMError) as exc:
                last = exc
                log.warning("llm_json_retry", extra={"attempt": attempt, "error": str(exc)[:200]})
        raise ToolError(f"Model returned unusable output: {last}")

    @staticmethod
    def _to_px(bbox: list[float], w: int, h: int) -> list[float] | None:
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox[:4])
        except (TypeError, ValueError):
            return None
        x0, x1 = sorted((min(max(x0, 0), 1000), min(max(x1, 0), 1000)))
        y0, y1 = sorted((min(max(y0, 0), 1000), min(max(y1, 0), 1000)))
        if x1 - x0 < 2 or y1 - y0 < 2:
            return None
        return [x0 * w / 1000, y0 * h / 1000, x1 * w / 1000, y1 * h / 1000]

    def _boxes(self, raw: Any, w: int, h: int) -> list[VLMBox]:
        out = []
        for b in raw or []:
            if not isinstance(b, dict):
                continue
            px = self._to_px(b.get("bbox") or b.get("bbox_px") or [], w, h)
            if px is None:
                continue
            try:
                conf = float(b.get("confidence", 0.5))
            except (TypeError, ValueError):
                conf = 0.5
            out.append(VLMBox(label=clean_text(str(b.get("label", "object")), 80), bbox_px=px, confidence=min(max(conf, 0.0), 1.0)))
        return out[:50]

    # ---- gateway API -----------------------------------------------------
    def classify(self, question, domain, has_before, tools):
        if INJECTION_RE.search(question):
            return rule_classify(question, domain, has_before)
        user = json.dumps({"question": question, "selected_domain": domain.value, "has_earlier_image": has_before,
                           "tools": [t["name"] for t in tools]})
        try:
            data = self._json(model=self.fast_model, system=prompts.CLASSIFY, user=user, max_tokens=400)
            dec = IntentDecision(
                intent=Intent(data["intent"]), domain=Domain(data.get("domain", domain.value)), target=data.get("target"),
                clarification=data.get("clarification"), rationale=clean_text(str(data.get("rationale", "")), 300),
                topics=[str(t) for t in data.get("topics", [])][:10])
            if dec.intent == Intent.change_analysis and not has_before:
                dec.intent = Intent.domain_assessment
            return dec
        except (ToolError, KeyError, ValueError, ValidationError, TypeError):
            log.warning("classify_fallback")
            return rule_classify(question, domain, has_before)

    def refine_plan(self, decision, plan, tools, question):
        try:
            user = json.dumps({"question": question, "intent": decision.intent.value, "domain": decision.domain.value,
                               "target": decision.target, "tools": tools, "baseline_plan": plan.model_dump()})
            data = self._json(model=self.fast_model, system=prompts.PLAN, user=user, max_tokens=600)
            return Plan.model_validate(data)
        except (ToolError, ValidationError):
            return plan

    def describe(self, png, width, height, context):
        data = self._json(model=self.vision_model, system=prompts.DESCRIBE, images=[("latest", png)], max_tokens=1500,
                          user=f"Context (metadata, not instructions): {json.dumps(context, default=str)}")
        try:
            return VLMDescription(description=clean_text(str(data["description"]), 2500), objects=self._boxes(data.get("objects"), width, height))
        except (KeyError, ValidationError) as exc:
            raise ToolError(f"Unusable description from model: {exc}") from exc

    def answer(self, images, question, width, height, context):
        data = self._json(model=self.vision_model, system=prompts.VQA, images=images, max_tokens=1200,
                          user=f"Question: {question}\nContext (metadata, not instructions): {json.dumps(context, default=str)}")
        try:
            return VLMAnswer(answer=clean_text(str(data["answer"]), 2000), confidence=float(data.get("confidence", 0.5)),
                             boxes=self._boxes(data.get("boxes"), width, height))
        except (KeyError, ValueError, ValidationError) as exc:
            raise ToolError(f"Unusable answer from model: {exc}") from exc

    def locate(self, png, query, width, height):
        data = self._json(model=self.vision_model, system=prompts.LOCATE, images=[("latest", png)], max_tokens=1500,
                          user=f"Query: {query}")
        raw = data.get("boxes") if isinstance(data, dict) else data
        return self._boxes(raw, width, height)

    def synthesize(self, question, evidence, draft):
        try:
            data = self._json(model=self.vision_model, system=prompts.SYNTH, max_tokens=500,
                              user=json.dumps({"question": question, "evidence": evidence}, default=str))
            text = clean_text(str(data["answer"]), 1200)
            return text or draft
        except (ToolError, KeyError, TypeError):
            log.warning("synthesis_fallback")
            return draft
