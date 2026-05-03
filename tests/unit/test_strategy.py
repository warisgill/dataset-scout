"""Unit tests for the LLM strategy assessor module (Azure OpenAI + Entra)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    Intent,
    LLMError,
    ScoutContext,
    Strategy,
    StrategyKind,
)
from dataset_scout.strategy import (
    assess_strategies,
    render_assessor_prompt,
)

pytestmark = pytest.mark.unit


# ─── fakes for litellm + azure-identity ─────────────────────────────


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
    return _Resp(choices=[_Choice(message=_Msg(content=json.dumps(payload)))])


def _entry(
    kind: str = "direct_use",
    confidence: float = 0.8,
    rationale: str = "matches cleanly",
) -> dict[str, Any]:
    return {
        "kind": kind,
        "confidence": confidence,
        "rationale": rationale,
        "caveats": ["small sample"],
        "transform": {
            "text_column": "text",
            "label_column": "label",
            "label_value_map": {"1": "positive", "0": "benign"},
            "label_kind_map": {"1": "ground_truth", "0": "ground_truth"},
            "filter": None,
            "take": "all",
        },
    }


@pytest.fixture
def fake_token_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace `llm_client.make_token_provider` with a stub so tests
    don't touch real Azure credentials."""

    def _stub() -> object:
        return lambda: "fake-bearer-token"

    monkeypatch.setattr("dataset_scout.llm_client.make_token_provider", _stub)


def _ctx(
    *,
    endpoint: str = "https://example.openai.azure.com",
    deployment: str = "gpt-4o-mini",
    api_version: str = "2024-10-21",
) -> ScoutContext:
    return ScoutContext(
        aoai_endpoint=endpoint,
        aoai_deployment=deployment,
        aoai_api_version=api_version,
    )


def _intent() -> Intent:
    return Intent(
        raw_brief="find prompt injection corpora",
        detection_target="prompt injection",
        threat_families=["prompt_injection", "indirect_injection"],
        deployment_context="rag_pipeline",
    )


def _candidate() -> Candidate:
    return Candidate(
        source="huggingface",
        id="acme/pi-corpus",
        revision="main",
        metadata=CandidateMetadata(
            description="A corpus of prompt-injection examples.",
            card_url="https://huggingface.co/datasets/acme/pi-corpus",
            license_raw="apache-2.0",
            license_spdx="Apache-2.0",
            languages_declared=["en"],
            task_categories=["text-classification"],
            tags=["prompt-injection", "security"],
        ),
        surfaced_by=["prompt_injection_core", "indirect_injection"],
    )


# ─── render_assessor_prompt ────────────────────────────────────────


def test_render_returns_nonempty_with_key_phrases() -> None:
    p = render_assessor_prompt(_candidate(), _intent())
    assert p
    assert "STRATEGIES" in p
    assert "schema" in p
    assert "not_useful" in p


def test_render_renders_none_for_empty_fields() -> None:
    cand = Candidate(source="hf", id="x/y")
    intent = Intent(raw_brief="")
    p = render_assessor_prompt(cand, intent)
    assert "Brief: (none)" in p
    assert "Detection target: (none)" in p
    assert "Threat families: (none)" in p
    assert "Deployment context: (none)" in p
    assert "Card URL: (none)" in p
    assert "Description: (none)" in p
    assert "License (raw / SPDX guess): (none) / (none)" in p
    assert "Declared languages: (none)" in p
    assert "Declared task categories: (none)" in p
    assert "Tags: (none)" in p
    assert "Surfaced by direction(s): (none)" in p


def test_render_includes_provided_fields() -> None:
    p = render_assessor_prompt(_candidate(), _intent())
    assert "Source: huggingface" in p
    assert "Id: acme/pi-corpus" in p
    assert "Card URL: https://huggingface.co/datasets/acme/pi-corpus" in p
    assert "Description: A corpus of prompt-injection examples." in p
    assert "License (raw / SPDX guess): apache-2.0 / Apache-2.0" in p
    assert "Declared languages: en" in p
    assert "Declared task categories: text-classification" in p
    assert "Tags: prompt-injection, security" in p
    assert "Surfaced by direction(s): prompt_injection_core, indirect_injection" in p


_SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "assessor_prompt.txt"


def test_render_snapshot_stable() -> None:
    """Prompt drift surfaces as a PR diff. First run writes the snapshot."""
    rendered = render_assessor_prompt(_candidate(), _intent())

    if not _SNAPSHOT_PATH.exists():
        _SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        _SNAPSHOT_PATH.write_text(rendered, encoding="utf-8")
    expected = _SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert rendered == expected, (
        "Assessor prompt drifted from snapshot. If intentional, delete "
        f"{_SNAPSHOT_PATH} and re-run the test."
    )


# ─── assess_strategies — call wiring ──────────────────────────────


def test_assess_routes_via_azure_with_token_provider(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    captured: dict[str, Any] = {}

    def fake_completion(**kwargs: Any) -> _Resp:
        captured.update(kwargs)
        return _resp(
            {
                "strategies": [
                    _entry("direct_use", 0.9),
                    _entry("subset_extraction", 0.7),
                    _entry("signal_proxy", 0.4),
                ]
            }
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = assess_strategies(_candidate(), _intent(), ctx=_ctx())

    assert len(result) == 3
    assert all(isinstance(s, Strategy) for s in result)
    # Descending confidence in returned input order is preserved.
    assert [s.confidence for s in result] == [0.9, 0.7, 0.4]
    assert [s.kind for s in result] == [
        StrategyKind.DIRECT_USE,
        StrategyKind.SUBSET_EXTRACTION,
        StrategyKind.SIGNAL_PROXY,
    ]

    # Azure routing.
    assert captured["model"] == "azure/gpt-4o-mini"
    assert captured["api_base"] == "https://example.openai.azure.com"
    assert captured["api_version"] == "2024-10-21"
    assert callable(captured["azure_ad_token_provider"])
    assert "api_key" not in captured

    assert captured["timeout"] == 30.0
    assert captured["response_format"] == {"type": "json_object"}
    msgs = captured["messages"]
    assert msgs[0]["role"] == "user"
    assert "STRATEGIES" in msgs[0]["content"]


def test_assess_sorts_by_confidence_descending(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp(
            {
                "strategies": [
                    _entry("subset_extraction", 0.3),
                    _entry("direct_use", 0.95),
                    _entry("signal_proxy", 0.6),
                ]
            }
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = assess_strategies(_candidate(), _intent(), ctx=_ctx())
    assert [s.confidence for s in result] == [0.95, 0.6, 0.3]
    assert result[0].kind is StrategyKind.DIRECT_USE


def test_assess_retries_once_on_validation_error(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    calls: list[Any] = []

    def fake_completion(**kwargs: Any) -> _Resp:
        calls.append(kwargs)
        if len(calls) == 1:
            return _resp({"oops": "wrong shape"})
        return _resp({"strategies": [_entry("direct_use", 0.8)]})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = assess_strategies(_candidate(), _intent(), ctx=_ctx())
    assert len(calls) == 2
    assert len(result) == 1
    assert result[0].kind is StrategyKind.DIRECT_USE


def test_assess_raises_after_two_validation_failures(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"oops": "still wrong"})

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError):
        assess_strategies(_candidate(), _intent(), ctx=_ctx())


def test_assess_drops_composition_only_silently(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp(
            {
                "strategies": [
                    _entry("direct_use", 0.8),
                    _entry("composition_only", 0.9),
                    _entry("signal_proxy", 0.4),
                ]
            }
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = assess_strategies(_candidate(), _intent(), ctx=_ctx())
    kinds = [s.kind for s in result]
    assert StrategyKind.COMPOSITION_ONLY not in kinds
    assert kinds == [StrategyKind.DIRECT_USE, StrategyKind.SIGNAL_PROXY]


def test_assess_raises_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {"n": 0}

    def fake_completion(**kwargs: Any) -> _Resp:
        called["n"] += 1
        return _resp({"strategies": []})

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError, match="Azure OpenAI is not configured"):
        assess_strategies(_candidate(), _intent(), ctx=ScoutContext())
    assert called["n"] == 0


def test_assess_empty_strategies_returns_empty(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp({"strategies": []})

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = assess_strategies(_candidate(), _intent(), ctx=_ctx())
    assert result == []


def test_assess_single_not_useful_passes_through(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        return _resp(
            {
                "strategies": [
                    {
                        "kind": "not_useful",
                        "confidence": 0.9,
                        "rationale": "wrong domain entirely",
                        "caveats": [],
                        "transform": {
                            "text_column": None,
                            "label_column": None,
                            "label_value_map": {},
                            "label_kind_map": {},
                            "filter": None,
                            "take": "all",
                        },
                    }
                ]
            }
        )

    monkeypatch.setattr("litellm.completion", fake_completion)
    result = assess_strategies(_candidate(), _intent(), ctx=_ctx())
    assert len(result) == 1
    assert result[0].kind is StrategyKind.NOT_USEFUL


def test_assess_wraps_completion_exception(
    monkeypatch: pytest.MonkeyPatch, fake_token_provider: None
) -> None:
    def fake_completion(**kwargs: Any) -> _Resp:
        raise RuntimeError("boom")

    monkeypatch.setattr("litellm.completion", fake_completion)
    with pytest.raises(LLMError, match="boom"):
        assess_strategies(_candidate(), _intent(), ctx=_ctx())
