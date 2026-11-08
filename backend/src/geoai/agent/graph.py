"""LangGraph orchestrator: normalize -> classify -> plan -> execute -> validate -> review gate -> synthesize -> finalize."""

from __future__ import annotations

import hashlib
import json
import operator
import time
import uuid
from collections.abc import Callable
from datetime import timedelta
from typing import Annotated, Any, TypedDict

import numpy as np
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from geoai import __version__
from geoai.agent.compose import compose_answer
from geoai.agent.planner import default_plan, validate_plan
from geoai.config import Settings
from geoai.llm.gateway import ModelGateway
from geoai.llm.sanitize import clean_text
from geoai.llm.types import IntentDecision, Plan
from geoai.schemas import (
    AnalysisRequest,
    AnalysisResult,
    Intent,
    Measurement,
    Observation,
    Provenance,
    Source,
    TraceStep,
    utcnow,
)
from geoai.storage.assets import AssetNotFound, AssetService
from geoai.tools.base import ToolContext, ToolError, ToolOutput, ToolRegistry

PIPELINE_VERSION = f"geoai-agent/{__version__}"


class AgentState(TypedDict, total=False):
    request: AnalysisRequest
    analysis_id: str
    tenant: str
    gateway: ModelGateway
    started_at: Any
    ctx: ToolContext
    decision: IntentDecision
    plan: Plan
    outputs: list[ToolOutput]
    runs: list[dict]
    issues: list[str]
    assumptions: list[str]
    review_reasons: list[str]
    needs_review: bool
    answer: str
    clarification: str | None
    failed: str | None
    result: AnalysisResult
    trace: Annotated[list[TraceStep], operator.add]


def _emit(event: str, **data: Any) -> None:
    try:
        get_stream_writer()({"event": event, **data})
    except Exception:  # noqa: BLE001 - not streaming
        pass


def build_graph(settings: Settings, assets: AssetService, registry: ToolRegistry, persist: Callable[[AnalysisResult, str], None]):
    # ------------------------------------------------------------------ nodes
    def normalize(state: AgentState) -> dict:
        req, tenant = state["request"], state["tenant"]
        try:
            latest = assets.resolve(req.asset_id, tenant)
            before = assets.resolve(req.before_asset_id, tenant) if req.before_asset_id else None
        except AssetNotFound as exc:
            return {"failed": str(exc), "trace": [TraceStep(step="normalize", status="error", detail=str(exc))]}
        issues, assumptions = [], []
        if before is not None and (before.raster.width, before.raster.height) != (latest.raster.width, latest.raster.height):
            issues.append("The earlier image has different dimensions from the latest image; change analysis is limited.")
        if latest.raster.captured_at is None:
            assumptions.append("No acquisition timestamp was provided.")
        if latest.raster.georef is None:
            assumptions.append("The image is not georeferenced: measurements are in pixels and no map coordinates are available.")
        ctx = ToolContext(latest=latest, before=before, gateway=state["gateway"], settings=settings, question=req.question, domain=req.domain)
        detail = f"{latest.raster.width}×{latest.raster.height} px, bands: {', '.join(latest.raster.bands)}" + (", georeferenced" if latest.raster.georef else ", not georeferenced")
        return {"ctx": ctx, "issues": issues, "assumptions": assumptions, "outputs": [], "runs": [],
                "trace": [TraceStep(step="normalize", detail=detail, data={"asset": latest.asset_id, "before": before.asset_id if before else None})]}

    def classify(state: AgentState) -> dict:
        req, gw = state["request"], state["gateway"]
        t0 = time.perf_counter()
        dec = gw.classify(req.question, req.domain, state["ctx"].before is not None, registry.describe())
        dec.domain = dec.domain or req.domain
        ms = int((time.perf_counter() - t0) * 1000)
        state["ctx"].domain = dec.domain
        return {"decision": dec, "clarification": dec.clarification,
                "trace": [TraceStep(step="classify", detail=f"intent={dec.intent.value}, domain={dec.domain.value}" + (f", target={dec.target}" if dec.target else ""),
                                    duration_ms=ms, data={"intent": dec.intent.value, "domain": dec.domain.value})]}

    def after_normalize(state: AgentState) -> str:
        return "finalize" if state.get("failed") else "classify"

    def after_classify(state: AgentState) -> str:
        dec = state["decision"]
        if dec.intent == Intent.out_of_scope or dec.clarification:
            return "respond_directly"
        return "plan"

    def respond_directly(state: AgentState) -> dict:
        dec = state["decision"]
        if dec.clarification:
            ans = dec.clarification
            step = TraceStep(step="clarify", detail="Asking the user for clarification", status="skipped")
        else:
            ans = ("I can only help with analysing the supplied imagery (floods, wildfires, earthquakes, urban planning, crop health). "
                   "I can't act on instructions that change how I operate.")
            step = TraceStep(step="decline", detail="Request is out of scope or attempts to override instructions", status="skipped")
        return {"answer": ans, "trace": [step]}

    def plan(state: AgentState) -> dict:
        dec, gw, req = state["decision"], state["gateway"], state["request"]
        baseline = default_plan(dec, req.question, gw.mode)
        chosen = baseline
        if gw.mode == "live":
            refined = gw.refine_plan(dec, baseline, registry.describe(), req.question)
            chosen = validate_plan(refined, registry, baseline)
        steps = " → ".join(s.tool for s in chosen.steps)
        return {"plan": chosen, "trace": [TraceStep(step="plan", detail=steps, data={"steps": [s.model_dump() for s in chosen.steps], "rationale": chosen.rationale})]}

    def execute(state: AgentState) -> dict:
        ctx = state["ctx"]
        outputs, runs, trace = [], [], []
        for st in state["plan"].steps:
            _emit("tool_start", tool=st.tool)
            t0 = time.perf_counter()
            try:
                out, ms = registry.run(st.tool, st.args, ctx)
                status, err = "ok", None
                outputs.append(out)
                trace.append(TraceStep(step="tool", tool=st.tool, status="ok", duration_ms=ms, detail=f"{len(out.observations)} observation(s), {len(out.metrics)} metric(s)"))
            except ToolError as exc:
                ms, status, err = int((time.perf_counter() - t0) * 1000), "error", str(exc)
                trace.append(TraceStep(step="tool", tool=st.tool, status="error", duration_ms=ms, detail=err))
            except Exception as exc:  # noqa: BLE001 - never let one tool take the run down
                ms, status, err = int((time.perf_counter() - t0) * 1000), "error", f"{type(exc).__name__}: {exc}"
                trace.append(TraceStep(step="tool", tool=st.tool, status="error", duration_ms=ms, detail="Unexpected tool failure"))
            digest = hashlib.sha256(json.dumps(out.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:16] if status == "ok" else None
            runs.append({"tool": st.tool, "args": st.args, "status": status, "error": err, "duration_ms": ms, "output_sha256_16": digest})
            _emit("tool_end", tool=st.tool, status=status)
        return {"outputs": outputs, "runs": runs, "trace": trace}

    def validate(state: AgentState) -> dict:
        ctx, issues = state["ctx"], list(state.get("issues", []))
        h, w = ctx.shape
        dropped = 0
        for out in state["outputs"]:
            keep = []
            for o in out.observations:
                b = o.bbox_px
                ok = b is not None and all(np.isfinite(b)) and 0 <= b[0] <= b[2] <= w + 1e-6 and 0 <= b[1] <= b[3] <= h + 1e-6 and 0 <= o.confidence <= 1
                if ok:
                    keep.append(o)
                else:
                    dropped += 1
            out.observations = keep
        if dropped:
            issues.append(f"{dropped} observation(s) with invalid geometry were discarded.")
        lt, bt = ctx.latest.raster.captured_at, ctx.before.raster.captured_at if ctx.before else None
        if lt and bt and bt >= lt:
            issues.append("The 'earlier' image is not older than the latest image; change results may be reversed.")
        elif lt and bt and lt - bt > timedelta(days=365 * 3):
            issues.append("The two images are more than three years apart; seasonal differences may dominate.")
        # contradiction: model prose vs deterministic flood measurement
        text = " ".join(o.text or "" for o in state["outputs"]).lower()
        cover = next((m.value for out in state["outputs"] for k, m in out.metrics.items() if k == "flood_cover_pct"), None)
        if text and cover is not None and isinstance(cover, (int, float)):
            says_no = any(p in text for p in ("no flood", "not flooded", "no visible water", "no standing water"))
            says_yes = "flood" in text and not says_no
            if says_no and cover > 5:
                issues.append(f"Contradiction: the vision model reports no flooding but the water mask covers {cover}% of the image.")
            if says_yes and cover < 1:
                issues.append("Contradiction: the vision model reports flooding but the water mask finds almost no water.")
        trace = [TraceStep(step="validate", status="ok" if not issues else "ok", detail=("; ".join(issues) if issues else "geometry, units, time order and cross-tool consistency checks passed"))]
        return {"issues": issues, "trace": trace}

    def review_gate(state: AgentState) -> dict:
        thr = settings.review_confidence_threshold
        reasons = list(state.get("review_reasons", []))
        obs = [o for out in state["outputs"] for o in out.observations]
        low = [o for o in obs if o.confidence < thr and o.kind == "observed"]
        if low:
            reasons.append(f"{len(low)} observation(s) fall below the {thr:.0%} confidence threshold.")
        for out in state["outputs"]:
            reasons += [r for r in out.review_reasons if r not in reasons]
        reasons += [i for i in state.get("issues", []) if i.startswith("Contradiction") and i not in reasons]
        if any(r["status"] == "error" for r in state["runs"]) and state["outputs"]:
            reasons.append("One or more tools failed; the result is partial.")
        needs = bool(reasons)
        return {"needs_review": needs, "review_reasons": reasons,
                "trace": [TraceStep(step="review_gate", status="ok", detail=f"analyst review {'required' if needs else 'not required'}" + (f" ({len(reasons)} reason(s))" if needs else ""))]}

    def synthesize(state: AgentState) -> dict:
        gw, req, dec = state["gateway"], state["request"], state["decision"]
        findings = [f.model_dump() for out in state["outputs"] for f in out.findings]
        texts = [o.text for o in state["outputs"] if o.text and dec.intent in (Intent.visual_question,)]
        if dec.intent == Intent.describe_scene:
            texts = [o.text for o in state["outputs"] if o.text]
            findings = [f for f in findings if f["text"] not in texts]
            findings = [dict(f, topics=[t for t in f["topics"] if t != "describe"]) for f in findings]
        draft = compose_answer(req.question, dec.intent, findings, texts)
        if not state["outputs"]:
            errs = "; ".join(r["error"] for r in state["runs"] if r.get("error"))
            draft = f"I could not complete this analysis: {errs or 'no tool produced a result'}."
            ans = draft
        else:
            evidence = {"intent": dec.intent.value, "findings": findings, "texts": texts,
                        "limitations": list(dict.fromkeys(x for out in state["outputs"] for x in out.limitations))[:6],
                        "needs_review": state.get("needs_review", False)}
            t0 = time.perf_counter()
            ans = clean_text(gw.synthesize(req.question, evidence, draft), 1500)
            _ = t0
        return {"answer": ans, "trace": [TraceStep(step="synthesize", detail=f"{len(ans)} characters")]}

    def finalize(state: AgentState) -> dict:
        req, tenant, gw = state["request"], state["tenant"], state["gateway"]
        started = state.get("started_at") or utcnow()
        finished = utcnow()
        if state.get("failed"):
            result = AnalysisResult(id=state["analysis_id"], status="failed", question=req.question, domain=req.domain, mode=gw.mode,
                                    error=state["failed"], answer="", trace=state.get("trace", []))
            result.provenance = Provenance(pipeline_version=PIPELINE_VERSION, mode=gw.mode, started_at=started, finished_at=finished)
            persist(result, tenant)
            return {"result": result}
        ctx = state["ctx"]
        outs = state.get("outputs", [])
        dec = state["decision"]
        observations: list[Observation] = [o for out in outs for o in out.observations]
        metrics: dict[str, Measurement] = {}
        for out in outs:
            metrics.update(out.metrics)
        merged = ToolOutput()
        for out in outs:
            merged.merge(out)
        assumptions = list(dict.fromkeys(state.get("assumptions", []) + merged.assumptions))
        limitations = list(dict.fromkeys(merged.limitations + state.get("issues", [])))
        failed_tools = [r for r in state.get("runs", []) if r["status"] == "error"]
        limitations += [f"Tool '{r['tool']}' failed: {r['error']}" for r in failed_tools]
        sources = []
        for role, b in (("latest", ctx.latest), ("before", ctx.before)):
            if b is None:
                continue
            g = b.raster.georef
            sources.append(Source(asset_id=b.asset_id, role=role, captured_at=b.raster.captured_at, sensor=b.raster.sensor or (b.scene_meta or {}).get("sensor"),
                                  crs=g.crs if g else None, gsd_m=round(g.pixel_size_m, 4) if g and g.pixel_size_m else None,
                                  width=b.raster.width, height=b.raster.height, sha256=b.info.sha256, georeferenced=g is not None))
        confs = [o.confidence for o in observations]
        conf = float(np.mean(confs)) if confs else next((o.confidence for o in outs if o.confidence is not None), None)
        if conf is not None and failed_tools:
            conf *= 0.8
        next_actions = list(dict.fromkeys(merged.next_actions))
        if state.get("needs_review"):
            next_actions.insert(0, "An analyst should review and approve this result before it is used operationally.")
        result = AnalysisResult(
            id=state["analysis_id"], status="completed", created_at=started, question=req.question, domain=dec.domain, intent=dec.intent, mode=gw.mode,
            answer=state.get("answer", ""), confidence=None if conf is None else round(conf, 3), metrics=metrics, observations=observations,
            assumptions=assumptions, limitations=limitations, sources=sources, needs_review=bool(state.get("needs_review")),
            review_reasons=state.get("review_reasons", []), review_status="pending" if state.get("needs_review") else "not_required",
            next_actions=next_actions, clarification=state.get("clarification"), trace=state.get("trace", []))
        req_hash = hashlib.sha256(req.model_dump_json().encode()).hexdigest()
        prov = Provenance(pipeline_version=PIPELINE_VERSION, mode=gw.mode, models=dict(gw.models), prompt_versions=dict(gw.prompt_versions),
                          tool_calls=state.get("runs", []), request_sha256=req_hash, started_at=started, finished_at=finished)
        result.provenance = prov
        body = result.model_dump_json(exclude={"provenance": {"record_sha256"}})
        prov.record_sha256 = hashlib.sha256(body.encode()).hexdigest()
        persist(result, tenant)
        return {"result": result}

    # ------------------------------------------------------------------ wiring
    g = StateGraph(AgentState)
    for name, fn in (("normalize", normalize), ("classify", classify), ("respond_directly", respond_directly), ("plan", plan), ("execute", execute),
                     ("validate", validate), ("review_gate", review_gate), ("synthesize", synthesize), ("finalize", finalize)):
        g.add_node(name, fn)
    g.add_edge(START, "normalize")
    g.add_conditional_edges("normalize", after_normalize, {"classify": "classify", "finalize": "finalize"})
    g.add_conditional_edges("classify", after_classify, {"plan": "plan", "respond_directly": "respond_directly"})
    g.add_edge("respond_directly", "finalize")
    g.add_edge("plan", "execute")
    g.add_edge("execute", "validate")
    g.add_edge("validate", "review_gate")
    g.add_edge("review_gate", "synthesize")
    g.add_edge("synthesize", "finalize")
    g.add_edge("finalize", END)
    return g.compile()


def new_analysis_id() -> str:
    return f"an_{uuid.uuid4().hex[:16]}"
