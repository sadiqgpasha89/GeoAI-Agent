"""FastAPI application factory and routes."""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from geoai import __version__
from geoai.agent.service import AnalysisService, ModeUnavailable
from geoai.config import Settings, get_settings
from geoai.exports import to_csv, to_geojson, to_html
from geoai.llm.demo import DemoGateway
from geoai.llm.factory import build_gateways
from geoai.observability import Metrics, configure_logging, request_id_var
from geoai.schemas import AnalysisRequest, AnalysisResult, AssetInfo, Domain, ReviewDecision
from geoai.storage.assets import AssetNotFound, AssetService, UploadError
from geoai.storage.db import Database

log = logging.getLogger("geoai.api")


def get_service(request: Request) -> AnalysisService:
    return request.app.state.service


def get_tenant(request: Request) -> str:
    tenants = request.app.state.settings.tenants
    if not tenants:
        return "public"
    supplied = request.headers.get("x-api-key", "")
    for key, t in tenants.items():
        if hmac.compare_digest(supplied.encode(), key.encode()):
            return t
    raise HTTPException(401, "Missing or invalid API key", headers={"WWW-Authenticate": "ApiKey"})


Svc = Annotated[AnalysisService, Depends(get_service)]
Tenant = Annotated[str, Depends(get_tenant)]


class RateLimiter:
    """Sliding-window limiter (per process). Use an ingress / Redis-backed limiter when running several replicas."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> float | None:
        if self.per_minute <= 0:
            return None
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return 60 - (now - q[0])
            q.append(now)
            if len(self.hits) > 10_000:
                self.hits.clear()
        return None


def create_app(settings: Settings | None = None, service: AnalysisService | None = None) -> FastAPI:
    s = settings or get_settings()
    configure_logging(s.log_level)
    if s.env == "production" and not s.tenants:
        raise RuntimeError("GEOAI_API_KEYS must be set when GEOAI_ENV=production")
    s.data_dir.mkdir(parents=True, exist_ok=True)
    metrics = Metrics()
    limiter = RateLimiter(s.rate_limit_per_minute)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if service is None:
            db = Database(s.db_url)
            assets = AssetService(s, db)
            default, live = build_gateways(s)
            app.state.service = AnalysisService(s, db, assets, default, live, DemoGateway())
        else:
            app.state.service = service
        log.info("startup", extra={"provider": app.state.service.default_gateway.mode, "env": s.env})
        yield

    app = FastAPI(title="GeoAI Agent API", version=__version__, lifespan=lifespan, docs_url="/api/docs", openapi_url="/api/openapi.json",
                  description="Agentic geospatial intelligence (LLM + VLM + deterministic geospatial tools).")
    app.add_middleware(CORSMiddleware, allow_origins=s.cors_origins, allow_methods=["GET", "POST"], allow_headers=["X-API-Key", "Content-Type"],
                       expose_headers=["X-Request-ID"], max_age=600)

    @app.middleware("http")
    async def observe(request: Request, call_next):  # noqa: ANN001
        rid = request.headers.get("x-request-id", "")[:64] or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)
        t0 = time.perf_counter()
        status = 500
        try:
            ip = request.client.host if request.client else "?"
            key = request.headers.get("x-api-key") or ip
            if request.url.path.startswith("/api/") and request.method != "OPTIONS":
                wait = limiter.check(key)
                if wait is not None:
                    status = 429
                    return JSONResponse({"detail": "Rate limit exceeded"}, status_code=429, headers={"Retry-After": str(int(wait) + 1), "X-Request-ID": rid})
            response: Response = await call_next(request)
            status = response.status_code
            response.headers["X-Request-ID"] = rid
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            if request.url.path.startswith("/api/") and not request.url.path.startswith("/api/docs"):
                response.headers.setdefault("Cache-Control", "no-store")
                response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
            return response
        finally:
            route = request.scope.get("route")
            rpath = getattr(route, "path", "unmatched")
            dur = time.perf_counter() - t0
            metrics.http_requests.labels(request.method, rpath, str(status)).inc()
            metrics.http_latency.labels(rpath).observe(dur)
            log.info("request", extra={"method": request.method, "path": request.url.path, "status": status, "ms": int(dur * 1000)})
            request_id_var.reset(token)

    app.state.settings = s

    def count(res: AnalysisResult) -> None:
        metrics.analyses.labels(res.mode, res.domain.value, res.status, str(res.needs_review).lower()).inc()
        if res.provenance:
            for c in res.provenance.tool_calls:
                metrics.tool_seconds.labels(c["tool"], c["status"]).observe(c["duration_ms"] / 1000)

    def preflight(service: AnalysisService, req: AnalysisRequest, t: str) -> None:
        try:
            service.assets.resolve(req.asset_id, t)
            if req.before_asset_id:
                service.assets.resolve(req.before_asset_id, t)
            service.pick_gateway(req.mode)
        except AssetNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        except ModeUnavailable as exc:
            raise HTTPException(400, str(exc)) from exc

    # -------------------------------------------------------------- routes
    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> dict:
        await run_in_threadpool(get_service(request).db.ping)
        return {"status": "ready"}

    @app.get("/metrics", include_in_schema=False)
    async def prom() -> Response:
        return Response(generate_latest(metrics.registry), media_type=CONTENT_TYPE_LATEST)

    api = "/api/v1"

    @app.get(f"{api}/meta")
    async def meta(service: Svc, _t: Tenant) -> dict:
        gw = service.default_gateway
        return {"version": __version__, "default_mode": gw.mode, "live_available": service.live_gateway is not None, "models": gw.models,
                "tools": service.registry.describe(), "domains": [d.value for d in Domain],
                "limits": {"max_upload_mb": s.max_upload_mb, "max_image_pixels": s.max_image_pixels},
                "review_confidence_threshold": s.review_confidence_threshold}

    @app.get(f"{api}/samples")
    async def samples(service: Svc, _t: Tenant) -> list[dict]:
        out = []
        for sid in service.assets.sample_ids():
            m = service.assets.sample_meta(sid)
            out.append({**m, "assets": {"after": f"sample:{sid}:after", "before": f"sample:{sid}:before"}})
        return out

    @app.post(f"{api}/assets", response_model=AssetInfo, status_code=201)
    async def upload(service: Svc, t: Tenant, file: UploadFile = File(...)) -> AssetInfo:
        data = await file.read(s.max_upload_mb * 1024 * 1024 + 1)
        try:
            return await run_in_threadpool(service.assets.register_upload, data, file.filename, t)
        except UploadError as exc:
            raise HTTPException(exc.status, str(exc)) from exc

    @app.get(f"{api}/assets/{{asset_id}}", response_model=AssetInfo)
    async def asset_info(asset_id: str, service: Svc, t: Tenant) -> AssetInfo:
        try:
            return (await run_in_threadpool(service.assets.resolve, asset_id, t)).info
        except AssetNotFound as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get(f"{api}/assets/{{asset_id}}/preview")
    async def asset_preview(asset_id: str, service: Svc, t: Tenant, max_side: Annotated[int, Query(ge=64, le=4096)] = 2048) -> Response:
        try:
            b = await run_in_threadpool(service.assets.resolve, asset_id, t)
        except AssetNotFound as exc:
            raise HTTPException(404, str(exc)) from exc
        png = await run_in_threadpool(b.raster.to_png_bytes, max_side)
        return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    @app.post(f"{api}/analyses", response_model=AnalysisResult)
    async def analyze(req: AnalysisRequest, service: Svc, t: Tenant) -> AnalysisResult:
        await run_in_threadpool(preflight, service, req, t)
        res = await run_in_threadpool(service.run, req, t)
        count(res)
        return res

    @app.post(f"{api}/analyses/stream")
    async def analyze_stream(req: AnalysisRequest, service: Svc, t: Tenant) -> StreamingResponse:
        await run_in_threadpool(preflight, service, req, t)
        loop = asyncio.get_running_loop()
        q: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                for ev in service.stream(req, t):
                    if ev.get("event") == "result":
                        count(AnalysisResult.model_validate(ev["result"]))
                    loop.call_soon_threadsafe(q.put_nowait, ev)
            except Exception:  # noqa: BLE001
                log.exception("stream_failed")
                loop.call_soon_threadsafe(q.put_nowait, {"event": "error", "message": "The analysis failed unexpectedly."})
            finally:
                loop.call_soon_threadsafe(q.put_nowait, None)

        threading.Thread(target=worker, daemon=True, name="analysis-stream").start()

        async def gen() -> AsyncIterator[str]:
            while (ev := await q.get()) is not None:
                yield f"event: {ev['event']}\ndata: {json.dumps(ev, default=str)}\n\n"

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})

    @app.get(f"{api}/analyses", response_model=list[AnalysisResult])
    async def list_analyses(service: Svc, t: Tenant, limit: Annotated[int, Query(ge=1, le=200)] = 50, needs_review: bool | None = None):
        return await run_in_threadpool(service.db.list_analyses, t, limit, needs_review)

    @app.get(f"{api}/analyses/{{analysis_id}}", response_model=AnalysisResult)
    async def get_analysis(analysis_id: str, service: Svc, t: Tenant) -> AnalysisResult:
        res = await run_in_threadpool(service.db.get_analysis, analysis_id, t)
        if res is None:
            raise HTTPException(404, "Analysis not found")
        return res

    @app.get(f"{api}/analyses/{{analysis_id}}/export")
    async def export(analysis_id: str, service: Svc, t: Tenant, format: Annotated[str, Query(pattern="^(geojson|csv|json|html)$")] = "json") -> Response:
        res = await run_in_threadpool(service.db.get_analysis, analysis_id, t)
        if res is None:
            raise HTTPException(404, "Analysis not found")
        name = f"geoai-{analysis_id}"
        if format == "geojson":
            return Response(json.dumps(to_geojson(res)), media_type="application/geo+json", headers={"Content-Disposition": f'attachment; filename="{name}.geojson"'})
        if format == "csv":
            return Response(to_csv(res), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{name}.csv"'})
        if format == "html":
            return HTMLResponse(to_html(res), headers={"Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'", "Content-Disposition": f'inline; filename="{name}.html"'})
        return Response(res.model_dump_json(indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="{name}.json"'})

    @app.post(f"{api}/analyses/{{analysis_id}}/review")
    async def review(analysis_id: str, decision: ReviewDecision, service: Svc, t: Tenant) -> dict:
        ok = await run_in_threadpool(service.db.add_review, analysis_id, t, decision)
        if not ok:
            raise HTTPException(404, "Analysis not found")
        return {"analysis_id": analysis_id, "review_status": decision.decision}

    @app.get(f"{api}/analyses/{{analysis_id}}/reviews")
    async def reviews(analysis_id: str, service: Svc, t: Tenant) -> list[dict]:
        return await run_in_threadpool(service.db.list_reviews, analysis_id, t)

    return app


def app_factory() -> FastAPI:  # uvicorn --factory geoai.api.app:app_factory
    return create_app()
