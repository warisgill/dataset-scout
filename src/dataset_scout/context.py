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

    api_keys: Mapping[str, str] = Field(default_factory=dict)

    llm_model: str = "gpt-4o-mini"

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
            DATASET_SCOUT_LLM_MODEL
            HUGGINGFACE_HUB_TOKEN / HF_TOKEN
            OPENAI_API_KEY, ANTHROPIC_API_KEY, ... (forwarded to litellm)
        """
        e = env if env is not None else os.environ
        api_keys: dict[str, str] = {}
        for k in (
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
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
        if v := e.get("DATASET_SCOUT_LLM_MODEL"):
            kwargs["llm_model"] = v
        if is_tty is not None:
            kwargs["is_tty"] = is_tty

        return cls(**kwargs)

    def enabled_sources(self) -> Iterable[str]:
        """Names of enabled sources, in stable order."""
        return [name for name, cfg in self.sources.items() if cfg.enabled]
