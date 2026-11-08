"""Raw LLM transports (one HTTP/SDK call each). Gateways build on these; tests inject fakes."""

from __future__ import annotations

import base64
import io
from abc import ABC, abstractmethod

import httpx
from PIL import Image


def shrink_png(data: bytes, max_side: int) -> bytes:
    with Image.open(io.BytesIO(data)) as im:
        if max(im.size) <= max_side:
            return data
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
        return buf.getvalue()


class LLMError(RuntimeError):
    pass


class LLMBackend(ABC):
    @abstractmethod
    def complete(self, *, model: str, system: str, user: str, images: list[tuple[str, bytes]] | None = None,
                 max_tokens: int = 1500) -> str: ...


class AnthropicBackend(LLMBackend):
    def __init__(self, api_key: str, timeout: float, max_image_px: int):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)
        self.max_image_px = max_image_px

    def complete(self, *, model, system, user, images=None, max_tokens=1500) -> str:
        import anthropic

        content: list[dict] = []
        for label, png in images or []:
            content.append({"type": "text", "text": f"Image: {label}"})
            data = base64.standard_b64encode(shrink_png(png, self.max_image_px)).decode()
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        content.append({"type": "text", "text": user})
        try:
            msg = self.client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                              messages=[{"role": "user", "content": content}])
        except anthropic.APIError as exc:
            raise LLMError(f"Anthropic API error: {exc}") from exc
        return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")


class OpenAICompatBackend(LLMBackend):
    """Any OpenAI-compatible chat-completions server (vLLM, Ollama, LM Studio, Azure OpenAI proxy...)."""

    def __init__(self, base_url: str, api_key: str | None, timeout: float, max_image_px: int, client: httpx.Client | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.client = client or httpx.Client(timeout=timeout)
        self.max_image_px = max_image_px

    def complete(self, *, model, system, user, images=None, max_tokens=1500) -> str:
        content: list[dict] = []
        for label, png in images or []:
            data = base64.standard_b64encode(shrink_png(png, self.max_image_px)).decode()
            content.append({"type": "text", "text": f"Image: {label}"})
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}})
        content.append({"type": "text", "text": user})
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", headers=self.headers, json={
                "model": model, "max_tokens": max_tokens, "temperature": 0,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}]})
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"OpenAI-compatible API error: {exc}") from exc
