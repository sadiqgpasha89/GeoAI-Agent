"""Application service: runs the agent graph (sync or streaming) and exposes persistence helpers."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from geoai.agent.graph import AgentState, build_graph, new_analysis_id
from geoai.config import Settings
from geoai.llm.gateway import ModelGateway
from geoai.schemas import AnalysisRequest, AnalysisResult, utcnow
from geoai.storage.assets import AssetService
from geoai.storage.db import Database
from geoai.tools import build_registry
from geoai.tools.base import ToolRegistry


class ModeUnavailable(ValueError):
    pass


class AnalysisService:
    def __init__(self, settings: Settings, db: Database, assets: AssetService, default_gateway: ModelGateway,
                 live_gateway: ModelGateway | None, demo_gateway: ModelGateway, registry: ToolRegistry | None = None):
        self.settings, self.db, self.assets = settings, db, assets
        self.default_gateway, self.live_gateway, self.demo_gateway = default_gateway, live_gateway, demo_gateway
        self.registry = registry or build_registry()
        self.graph = build_graph(settings, assets, self.registry, lambda res, tenant: db.save_analysis(res, tenant))

    def pick_gateway(self, mode: str) -> ModelGateway:
        if mode == "demo":
            return self.demo_gateway
        if mode == "live":
            if self.live_gateway is None:
                raise ModeUnavailable("Live mode requires a configured model provider (set ANTHROPIC_API_KEY or an OpenAI-compatible endpoint).")
            return self.live_gateway
        return self.default_gateway

    def _initial(self, req: AnalysisRequest, tenant: str, analysis_id: str | None) -> AgentState:
        return {"request": req, "tenant": tenant, "analysis_id": analysis_id or new_analysis_id(), "gateway": self.pick_gateway(req.mode),
                "started_at": utcnow(), "trace": []}

    def run(self, req: AnalysisRequest, tenant: str, analysis_id: str | None = None) -> AnalysisResult:
        final = self.graph.invoke(self._initial(req, tenant, analysis_id))
        return final["result"]

    def stream(self, req: AnalysisRequest, tenant: str, analysis_id: str | None = None) -> Iterator[dict[str, Any]]:
        """Yield UI events: {'event': 'step'|'tool_start'|'tool_end'|'result', ...}."""
        state = self._initial(req, tenant, analysis_id)
        yield {"event": "start", "analysis_id": state["analysis_id"], "mode": state["gateway"].mode}
        result = None
        for mode, payload in self.graph.stream(state, stream_mode=["updates", "custom"]):
            if mode == "custom":
                yield payload
                continue
            for node, update in payload.items():
                for step in (update or {}).get("trace", []):
                    yield {"event": "step", "node": node, "step": step.model_dump(mode="json")}
                if update and "result" in update:
                    result = update["result"]
        if result is not None:
            yield {"event": "result", "result": result.model_dump(mode="json")}
