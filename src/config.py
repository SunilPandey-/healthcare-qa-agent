"""Centralised, externalised configuration.

All tunables (provider, model, temperature, timeouts, API keys) are read from
environment variables / a local ``.env`` file so that *nothing* is hard-coded in
the agent logic. This is the single source of truth for runtime settings.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "mock"]


class Settings(BaseSettings):
    """Strongly-typed application settings loaded from the environment.

    Pydantic validates types and applies defaults, so a misconfigured
    deployment fails fast with a clear error instead of surfacing deep in the
    agent loop.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Provider selection ---
    llm_provider: Provider = "anthropic"

    # --- Anthropic ---
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-opus-4-8"

    # --- OpenAI ---
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o"

    # --- Generation settings ---
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=1500, gt=0)
    llm_timeout_seconds: float = Field(default=45.0, gt=0)
    llm_max_retries: int = Field(default=3, ge=0)

    # --- Agent settings ---
    agent_max_steps: int = Field(default=6, gt=0)

    # --- PubMed / NCBI E-utilities ---
    ncbi_tool: str = "healthcare-qa-agent"
    ncbi_email: str = "you@example.com"
    ncbi_api_key: Optional[str] = None

    # --- Logging ---
    log_level: str = "INFO"

    @property
    def active_model(self) -> str:
        """The model name for the currently selected provider."""
        if self.llm_provider == "openai":
            return self.openai_model
        if self.llm_provider == "mock":
            return "mock-model"
        return self.anthropic_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached, process-wide Settings instance."""
    return Settings()
