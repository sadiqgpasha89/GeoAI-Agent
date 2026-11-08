"""Weather context adapter: scene fixture (demo) or Open-Meteo (live, opt-in, georeferenced scenes only)."""

from __future__ import annotations

import httpx

from geoai.schemas import Domain
from geoai.tools.base import Finding, Measurement, NoArgs, ToolContext, ToolOutput

TOOL = "weather_context"


def compass(deg: float) -> str:
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int((deg % 360) / 45 + 0.5) % 8]


def fetch_open_meteo(lat: float, lon: float, timeout: float = 8.0) -> dict:
    r = httpx.get(
        "https://api.open-meteo.com/v1/forecast",
        params={"latitude": round(lat, 3), "longitude": round(lon, 3), "timezone": "UTC",
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m"},
        timeout=timeout,
    )
    r.raise_for_status()
    c = r.json()["current"]
    return {"source": "open-meteo", "wind_from_deg": c["wind_direction_10m"], "wind_speed_kmh": c["wind_speed_10m"],
            "gust_kmh": c["wind_gusts_10m"], "relative_humidity_pct": c["relative_humidity_2m"], "temperature_c": c["temperature_2m"]}


def get_weather(ctx: ToolContext) -> dict | None:
    if ctx.meta.get("weather"):
        return ctx.meta["weather"]
    if ctx.settings.enable_external_data and ctx.georef:
        lon, lat = ctx.georef.px_to_lonlat(ctx.georef.width / 2, ctx.georef.height / 2)
        try:
            return fetch_open_meteo(lat, lon)
        except (httpx.HTTPError, KeyError, ValueError):
            return None
    return None


def weather_context(ctx: ToolContext, _args: NoArgs) -> ToolOutput:
    out = ToolOutput()
    wx = ctx.cached("weather", lambda: get_weather(ctx))
    if not wx:
        out.limitations.append("Weather context unavailable (no fixture, external data disabled or provider unreachable).")
        out.findings.append(Finding(text="Weather and wind context was not available.", kind="not_determinable", topics=["weather", "wind"]))
        return out
    frm = float(wx["wind_from_deg"])
    out.metrics["wind_from"] = Measurement(value=round(frm), unit="°", method=wx["source"])
    out.metrics["wind_speed"] = Measurement(value=wx["wind_speed_kmh"], unit="km/h", method=wx["source"])
    out.metrics["relative_humidity"] = Measurement(value=wx["relative_humidity_pct"], unit="%", method=wx["source"])
    out.findings.append(Finding(
        text=f"Wind from the {compass(frm)} at {wx['wind_speed_kmh']:.0f} km/h (gusts {wx.get('gust_kmh', wx['wind_speed_kmh']):.0f} km/h), relative humidity {wx['relative_humidity_pct']:.0f}% [{wx['source']}].",
        topics=["weather", "wind", "risk"]))
    if wx["source"] == "demo fixture":
        out.assumptions.append("Weather values come from the sample scene's bundled fixture, not a live feed.")
    return out


SPEC = dict(name=TOOL, description="Fetch wind, humidity and temperature context for the scene.", args_model=NoArgs, fn=weather_context,
            domains=(Domain.wildfire,))
