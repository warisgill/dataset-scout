"""ScoutContext — explicit configuration & auth carrier.

Every public library entry point (`recon`, `inspect`, `curate`) takes a
`ScoutContext`. The CLI populates it from environment variables and the
TOML config file; a future HTTP server would populate it from request
context.

This keeps the library free of global state so it stays trivially
re-entrant and trivially wrappable behind an API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Iterable


def _default_cache_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "dataset-scout" / "cache"
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg) / "dataset-scout"
    return Path.home() / ".cache" / "dataset-scout"


def _default_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "dataset-scout"
    xdg = os.environ.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "dataset-scout"
    return Path.home() / ".config" / "dataset-scout"


class SourceConfig(BaseModel):
    """Per-source enablement and auth posture."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    enabled: bool = True


class ScoutContext(BaseModel):
    """Explicit runtime context. No global state.

    Construct via `ScoutContext.from_env()` for normal use, or build one
    directly in tests / API callers.

    Azure OpenAI + Entra is the LLM auth model: callers configure the
    AOAI endpoint and deployment via env vars (or directly), and the
    bearer token is acquired lazily via `DefaultAzureCredential` (which
    chains `az login`, managed identity, env-var service-principal,
    etc.). No API keys are stored in `ScoutContext` for the LLM path.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    config_dir: Path = Field(default_factory=_default_config_dir)
    out_dir: Path = Field(default_factory=lambda: Path("dscout-out"))

    sources: Mapping[str, SourceConfig] = Field(
        default_factory=lambda: {
            "huggingface": SourceConfig(enabled=True),
            "kaggle": SourceConfig(enabled=False),  # opt-in: needs Kaggle creds
            "pwc": SourceConfig(enabled=True),
        }
    )

    # Source-specific tokens (HF, Kaggle). LLM auth is Entra, not API
    # keys, and lives in the AOAI fields below.
    api_keys: Mapping[str, str] = Field(default_factory=dict)

    # ─── Azure OpenAI ────────────────────────────────────────────
    aoai_endpoint: str | None = None
    aoai_deployment: str | None = None
    aoai_api_version: str = "2024-10-21"

    is_tty: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        env: Mapping[str, str] | None = None,
        is_tty: bool | None = None,
    ) -> Self:
        """Build a ScoutContext from environment variables.

        Recognized variables:
            DATASET_SCOUT_CACHE_DIR
            DATASET_SCOUT_CONFIG_DIR
            DATASET_SCOUT_OUT_DIR
            HUGGINGFACE_HUB_TOKEN / HF_TOKEN
            KAGGLE_USERNAME / KAGGLE_KEY

            AZURE_OPENAI_ENDPOINT       — e.g. https://my-aoai.openai.azure.com/
            AZURE_OPENAI_DEPLOYMENT     — deployment name (e.g. gpt-4o-mini)
            AZURE_OPENAI_API_VERSION    — overrides default
        """
        e = env if env is not None else os.environ
        api_keys: dict[str, str] = {}
        for k in (
            "HUGGINGFACE_HUB_TOKEN",
            "HF_TOKEN",
            "KAGGLE_USERNAME",
            "KAGGLE_KEY",
        ):
            if v := e.get(k):
                api_keys[k] = v

        kwargs: dict[str, object] = {"api_keys": api_keys}
        if v := e.get("DATASET_SCOUT_CACHE_DIR"):
            kwargs["cache_dir"] = Path(v)
        if v := e.get("DATASET_SCOUT_CONFIG_DIR"):
            kwargs["config_dir"] = Path(v)
        if v := e.get("DATASET_SCOUT_OUT_DIR"):
            kwargs["out_dir"] = Path(v)
        if v := e.get("AZURE_OPENAI_ENDPOINT"):
            kwargs["aoai_endpoint"] = v.rstrip("/")
        if v := e.get("AZURE_OPENAI_DEPLOYMENT"):
            kwargs["aoai_deployment"] = v
        if v := e.get("AZURE_OPENAI_API_VERSION"):
            kwargs["aoai_api_version"] = v
        if is_tty is not None:
            kwargs["is_tty"] = is_tty

        return cls(**kwargs)

    def enabled_sources(self) -> Iterable[str]:
        """Names of enabled sources, in stable order."""
        return [name for name, cfg in self.sources.items() if cfg.enabled]

    @property
    def aoai_configured(self) -> bool:
        """True iff endpoint + deployment are present.

        Bearer-token availability (e.g., `az login` having been run) is
        checked lazily by the LLM call site — it requires a network
        round-trip we don't want to pay here.
        """
        return bool(self.aoai_endpoint) and bool(self.aoai_deployment)
