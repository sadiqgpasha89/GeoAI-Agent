"""Plan construction and validation. The LLM may refine a plan, but only inside these guardrails."""

from __future__ import annotations

from geoai.llm.types import IntentDecision, Plan, PlanStep
from geoai.schemas import Domain, Intent
from geoai.tools.base import ToolError, ToolRegistry

DOMAIN_TOOL = {
    Domain.flood: "assess_flood",
    Domain.wildfire: "assess_wildfire",
    Domain.earthquake: "assess_earthquake",
    Domain.urban: "assess_urban",
    Domain.agriculture: "assess_crop_health",
}
MAX_STEPS = 5


def default_plan(decision: IntentDecision, question: str, mode: str) -> Plan:
    dt = DOMAIN_TOOL[decision.domain]
    i = decision.intent
    if i == Intent.locate_object:
        steps = [PlanStep(tool="ground_object", args={"query": decision.target or question}, why="Localise the requested object")]
    elif i == Intent.describe_scene:
        steps = [PlanStep(tool=dt, why="Measure the scene deterministically"), PlanStep(tool="describe_scene", why="Describe the scene")]
        if mode == "live":
            steps.reverse()
    elif i == Intent.visual_question and mode == "live":
        steps = [PlanStep(tool="visual_qa", args={"question": question}, why="Answer the open-ended question"),
                 PlanStep(tool=dt, why="Cross-check with deterministic measurements")]
    elif "weather" in decision.topics and decision.domain == Domain.wildfire and not ({"burn", "fire", "risk"} & set(decision.topics)):
        steps = [PlanStep(tool="weather_context", why="Fetch weather context")]
    else:
        steps = [PlanStep(tool=dt, why="Measure impact with deterministic geospatial tools")]
    return Plan(steps=steps, rationale=decision.rationale or "Rule-based plan")


def validate_plan(plan: Plan, registry: ToolRegistry, baseline: Plan) -> Plan:
    """Reject unknown tools / invalid args / over-long plans; fall back to the baseline plan."""
    try:
        if not plan.steps or len(plan.steps) > MAX_STEPS:
            raise ToolError("plan size out of range")
        clean = []
        for st in plan.steps:
            parsed = registry.validate_call(st.tool, st.args)
            clean.append(PlanStep(tool=st.tool, args=parsed.model_dump(), why=st.why[:200]))
        # the baseline's measurement tools must survive refinement (an LLM cannot talk its way out of evidence)
        needed = {s.tool for s in baseline.steps if s.tool.startswith("assess_") or s.tool == "ground_object"}
        if not needed <= {s.tool for s in clean}:
            raise ToolError("plan dropped mandatory measurement tools")
        return Plan(steps=clean, rationale=plan.rationale[:300])
    except ToolError:
        return baseline
