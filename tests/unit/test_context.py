"""ScoutContext behavior tests."""

from __future__ import annotations

import pytest

from dataset_scout import ScoutContext

pytestmark = pytest.mark.unit


def test_default_context_has_expected_sources():
    ctx = ScoutContext.from_env(env={})
    names = list(ctx.sources)
    assert "huggingface" in names
    assert "kaggle" in names
    assert "pwc" in names
    # Kaggle is opt-in (needs creds) — disabled by default.
    assert ctx.sources["kaggle"].enabled is False
    assert ctx.sources["huggingface"].enabled is True
    assert ctx.sources["pwc"].enabled is True


def test_enabled_sources_filters():
    ctx = ScoutContext.from_env(env={})
    enabled = list(ctx.enabled_sources())
    assert "huggingface" in enabled
    assert "pwc" in enabled
    assert "kaggle" not in enabled


def test_from_env_picks_up_api_keys():
    ctx = ScoutContext.from_env(
        env={
            "OPENAI_API_KEY": "sk-test",
            "HUGGINGFACE_HUB_TOKEN": "hf_test",
            "UNRELATED_VAR": "ignored",
        }
    )
    assert ctx.api_keys.get("OPENAI_API_KEY") == "sk-test"
    assert ctx.api_keys.get("HUGGINGFACE_HUB_TOKEN") == "hf_test"
    assert "UNRELATED_VAR" not in ctx.api_keys


def test_from_env_overrides_paths(tmp_path):
    ctx = ScoutContext.from_env(
        env={
            "DATASET_SCOUT_CACHE_DIR": str(tmp_path / "c"),
            "DATASET_SCOUT_OUT_DIR": str(tmp_path / "o"),
            "DATASET_SCOUT_LLM_MODEL": "claude-haiku",
        }
    )
    assert ctx.cache_dir == tmp_path / "c"
    assert ctx.out_dir == tmp_path / "o"
    assert ctx.llm_model == "claude-haiku"


def test_context_is_frozen():
    import pydantic

    ctx = ScoutContext.from_env(env={})
    with pytest.raises(pydantic.ValidationError):
        ctx.llm_model = "other"  # type: ignore[misc]
