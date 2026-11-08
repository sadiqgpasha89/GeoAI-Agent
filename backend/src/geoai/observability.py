"""Structured JSON logging, request-id propagation and Prometheus metrics."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
import time
from datetime import UTC, datetime

from prometheus_client import CollectorRegistry, Counter, Histogram

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {"ts": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"), "level": record.levelname,
                   "logger": record.name, "msg": record.getMessage(), "request_id": request_id_var.get()}
        for k, v in record.__dict__.items():
            if k not in _RESERVED and not k.startswith("_"):
                payload[k] = v
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO") -> None:
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level.upper())
    logging.getLogger("uvicorn.access").disabled = True  # replaced by our access log (no query strings / bodies)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.http_requests = Counter("geoai_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=self.registry)
        self.http_latency = Histogram("geoai_http_request_seconds", "HTTP latency", ["route"], registry=self.registry,
                                      buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60))
        self.analyses = Counter("geoai_analyses_total", "Analyses run", ["mode", "domain", "status", "needs_review"], registry=self.registry)
        self.tool_seconds = Histogram("geoai_tool_seconds", "Tool latency", ["tool", "status"], registry=self.registry,
                                      buckets=(0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10, 30, 60))


class Timer:
    def __enter__(self) -> Timer:
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a: object) -> None:
        self.elapsed = time.perf_counter() - self.t0
