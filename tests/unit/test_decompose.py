"""Unit tests for the LLM decomposition module (M2a)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dataset_scout import (
    DecompositionDirection,
    Intent,
    LLMError,
    ScoutContext,
)
from dataset_scout.decompose import (
    DecomposeResponse,
    decompose_intent,
    llm_available,
    render_decompose_prompt,
)

pytestmark = pytest.mark.unit


# ─── fakes for litellm ──────────────────────────────────────────────


@dataclass
class _Msg:
    content: str


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Resp:
    choices: list[_Choice]


def _resp(payload: Any) -> _Resp:
    """Build a fake OpenAI-shape completion response carrying a JSON payload."""
    return _Resp(choices=[_Choice(message=_Msg(content=json.dumps(payload)))])


def _good_directions(n: int = 3) -> list[dict[str, Any]]:
    return [
        {
            "name": f"dir_{i}",
            "rationale": f"because reason {i}",
            "keywords": [f"k{i}a", f"k{i}b"],
            "threat_families": ["prompt_injection"],
            "expected_finds": f"useful data {i}",
        }
        for i in range(n)
    ]


# ─── render_decompose_prompt ────────────────────────────────────────


def test_render_returns_nonempty_with_key_phrases() -> None:
    p = render_decompose_prompt(Intent(raw_brief="x"))
    assert p
    assert "3-7" in p
    assert "JSON matching" in p
    assert "lexical" in p.lower()


def test_render_renders_none_for_empty_fields() -> None:
    p = render_decompose_prompt(Intent(raw_brief=""))
    # raw_brief, detection_target, threat_families, deployment_context all default → (none)
    assert "Brief: (none)" in p
    assert "Detection target: (none)" in p
    assert "Threat families: (none)" in p
    assert "Deployment context: (none)" in p


def test_render_includes_provided_fields() -> None:
    intent = Intent(
        raw_brief="find prompt injection data",
        detection_target="prompt injection",
        threat_families=["prompt_injection", "jailbreak"],
        deployment_context="rag_pipeline",
    )
    p = render_decompose_prompt(intent)
    assert "Brief: find prompt injection data" in p
    assert "Detection target: prompt injection" in p
    assert "Threat families: prompt_injection, jailbreak" in p
    assert "Deployment context: rag_pipeline" in p


_SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "decompose_prompt.txt"


def test_render_snapshot_stable() -> None:
    """Prompt drift surfaces as a PR diff. First run writes the snapshot."""
    intent = Intent(
        raw_brief="find prompt injection corpora for our RAG service",
        detection_target="prompt injection",
        threat_families=["prompt_injection", "indirect_injection"],
        deployment_context="rag_pipeline",
    )
    rendered = render_decompose_prompt(intent)

    if not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
    expected = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Decomposition prompt drifted from snapshot. If intentional, delete "
        f"{_SNAPSHOT_PATH} and re-run the test."
    )


# ─── decompose_intent ───────────────────────────────────────────────


def _ctx(api_keys: dict[str, str] | None = None, model: str = "gpt-4o-mini") -> ScoutContext:
    return ScoutContext(api_keys=api_keys or {}, llm_model=model)


def test_decompose_happy_path(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _Resp:
        captured.update(kwargs)
        return _resp({"directions": _good_directions(3)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(
        Intent(raw_brief="prompt injection"),
        ctx=_ctx({"OPENAI_API_KEY": "sk-test"}),
    )
    assert len(result) == 3
    assert all(isinstance(d, DecompositionDirection) for d in result)
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "sk-test"
    assert captured["timeout"] == 30.0
    assert captured["response_format"] is DecomposeResponse
    msgs = captured["messages"]
    assert msgs[0]["role"] == "user"
    assert "3-7" in msgs[0]["content"]


def test_decompose_retries_once_on_validation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[Any] = []

    def fake_completion(**kwargs: Any) -> _Resp:
        calls.append(kwargs)
        if len(calls) == 1:
            return _resp({"oops": "wrong shape"})
        return _resp({"directions": _good_directions(3)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="x"), ctx=_ctx())
    assert len(calls) == 2
    assert len(result) == 3


def test_decompose_raises_after_two_validation_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"oops": "still wrong"})

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError):
        decompose_intent(Intent(raw_brief="x"), ctx=_ctx())


def test_decompose_clips_to_seven(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"directions": _good_directions(9)})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="x"), ctx=_ctx())
    assert len(result) == 7


def test_decompose_empty_directions_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"directions": []})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = decompose_intent(Intent(raw_brief="x"), ctx=_ctx())
    assert result == []


def test_decompose_wraps_completion_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        raise RuntimeError("boom")

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError, match="boom"):
        decompose_intent(Intent(raw_brief="x"), ctx=_ctx())


# ─── llm_available ──────────────────────────────────────────────────


def test_llm_available_false_with_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    # Strip any ambient keys from the process env so litellm's own
    # validate_environment can't quietly answer "yes".
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert llm_available(_ctx()) is False


def test_llm_available_true_with_openai_key_in_ctx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    ctx = _ctx({"OPENAI_API_KEY": "sk-fake"})
    assert llm_available(ctx) is True


def test_llm_available_true_for_anthropic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ctx = _ctx({"ANTHROPIC_API_KEY": "sk-ant-fake"}, model="claude-3-5-sonnet-20241022")
    assert llm_available(ctx) is True
