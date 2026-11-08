from __future__ import annotations

from geoai.config import Settings
from geoai.llm.backends import AnthropicBackend, OpenAICompatBackend
from geoai.llm.demo import DemoGateway
from geoai.llm.gateway import ModelGateway
from geoai.llm.live import LiveGateway


def build_live_gateway(s: Settings) -> LiveGateway | None:
    prov = s.resolved_provider
    if prov == "anthropic" and s.anthropic_api_key:
        backend = AnthropicBackend(s.anthropic_api_key.get_secret_value(), s.llm_timeout_s, s.llm_max_image_px)
        return LiveGateway(backend, s.model_fast, s.model_vision, "anthropic")
    if prov == "openai_compat" and s.openai_base_url and s.openai_model:
        key = s.openai_api_key.get_secret_value() if s.openai_api_key else None
        backend = OpenAICompatBackend(s.openai_base_url, key, s.llm_timeout_s, s.llm_max_image_px)
        return LiveGateway(backend, s.openai_model, s.openai_model, "openai_compat")
    return None


def build_gateways(s: Settings) -> tuple[ModelGateway, ModelGateway | None]:
    """Return (default gateway, live gateway or None)."""
    live = build_live_gateway(s)
    return (live or DemoGateway()), live
