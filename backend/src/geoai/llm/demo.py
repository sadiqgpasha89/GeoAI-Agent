"""Offline demo gateway: rule-based routing, deterministic synthesis, no network, no credentials."""

from __future__ import annotations

from geoai.agent.compose import compose_answer, rule_classify
from geoai.llm.gateway import ModelGateway
from geoai.schemas import Intent
from geoai.tools.base import ToolError


class DemoGateway(ModelGateway):
    mode = "demo"
    models = {"engine": "geoai-demo-rules (classical CV + reference layers)"}
    prompt_versions = {"demo": "v1"}

    def classify(self, question, domain, has_before, tools):
        return rule_classify(question, domain, has_before)

    def describe(self, png, width, height, context):
        raise ToolError("Vision-language descriptions need a live model provider")

    def answer(self, images, question, width, height, context):
        raise ToolError("Open-ended visual Q&A needs a live model provider")

    def locate(self, png, query, width, height):
        raise ToolError("Open-vocabulary grounding needs a live model provider")

    def synthesize(self, question, evidence, draft):
        return compose_answer(question, Intent(evidence["intent"]), evidence["findings"], evidence.get("texts", []))
