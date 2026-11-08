"""Application settings (12-factor: everything comes from the environment)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="GEOAI_", env_file=".env", extra="ignore")

    env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # --- storage -----------------------------------------------------------
    data_dir: Path = Path("data")
    database_url: str = ""  # default: sqlite file inside data_dir

    # --- security ----------------------------------------------------------
    # "key1:tenant-a,key2:tenant-b". Required when env=production.
    api_keys: str = ""
    cors_origins: Annotated[list[str], NoDecode] = ["http://localhost:5173", "http://127.0.0.1:5173"]  # comma-separated or JSON list
    max_upload_mb: int = 25
    max_image_pixels: int = 80_000_000
    rate_limit_per_minute: int = 120
    clamav_host: str = ""  # optional clamd host for upload malware scanning
    clamav_port: int = 3310

    # --- model gateway -----------------------------------------------------
    provider: Literal["auto", "demo", "anthropic", "openai_compat"] = "auto"
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("GEOAI_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY")
    )
    model_fast: str = "claude-haiku-5-5"  # routing / planning
    model_vision: str = "claude-sonnet-5-5"  # VLM + synthesis
    openai_base_url: str = ""  # e.g. http://localhost:8000/v1 (vLLM / Ollama / Azure)
    openai_api_key: SecretStr | None = None
    openai_model: str = ""
    llm_timeout_s: float = 90.0
    llm_max_image_px: int = 1568

    # --- agent behaviour ---------------------------------------------------
    review_confidence_threshold: float = 0.6
    enable_external_data: bool = False  # allow live weather lookups (Open-Meteo)

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                import json

                return json.loads(v)
            return [o.strip() for o in v.split(",") if o.strip()]
        return v

    # --- derived -----------------------------------------------------------
    @property
    def sample_dir(self) -> Path:
        return self.data_dir / "samples"

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def db_url(self) -> str:
        return self.database_url or f"sqlite:///{(self.data_dir / 'geoai.db').resolve()}"

    @property
    def tenants(self) -> dict[str, str]:
        out: dict[str, str] = {}
        for item in filter(None, (p.strip() for p in self.api_keys.split(","))):
            key, _, tenant = item.partition(":")
            if key:
                out[key] = tenant or "default"
        return out

    @property
    def resolved_provider(self) -> Literal["demo", "anthropic", "openai_compat"]:
        if self.provider != "auto":
            return self.provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_base_url and self.openai_model:
            return "openai_compat"
        return "demo"


@lru_cache
def get_settings() -> Settings:
    return Settings()
