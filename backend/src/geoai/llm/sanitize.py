"""Output hygiene for model-generated text (defence in depth against prompt injection)."""

from __future__ import annotations

import json
import re

_URL = re.compile(r"https?://\S+|www\.\S+", re.I)
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_TAG = re.compile(r"<[^>]{1,200}>")
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def clean_text(text: str, max_len: int = 2000) -> str:
    text = _MD_LINK.sub(r"\1", text)
    text = _URL.sub("[link removed]", text)
    text = _TAG.sub("", text)
    text = _CTRL.sub("", text)
    return text.strip()[:max_len]


def extract_json(text: str) -> object:
    """Parse the first JSON object/array in ``text`` (tolerates code fences and chatter)."""
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|```$", "", t.strip(), flags=re.M).strip()
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        start = t.find(open_c)
        end = t.rfind(close_c)
        if start != -1 and end > start:
            try:
                return json.loads(t[start : end + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError("No JSON found in model output")
