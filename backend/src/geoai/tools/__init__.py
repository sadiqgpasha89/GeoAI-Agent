from __future__ import annotations

from geoai.tools import crop, earthquake, flood, perception, urban, weather, wildfire
from geoai.tools.base import ToolRegistry, ToolSpec


def build_registry() -> ToolRegistry:
    reg = ToolRegistry()
    for spec in (*perception.SPECS, flood.SPEC, wildfire.SPEC, earthquake.SPEC, urban.SPEC, crop.SPEC, weather.SPEC):
        reg.register(ToolSpec(**spec))
    return reg
