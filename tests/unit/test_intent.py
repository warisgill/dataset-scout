"""Unit tests for the heuristic intent parser."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dataset_scout import LicensePolicy, SensitiveDomain
from dataset_scout.intent import HeuristicIntentParser, IntentParser

pytestmark = pytest.mark.unit


@pytest.fixture
def parser() -> HeuristicIntentParser:
    return HeuristicIntentParser()


def test_protocol_membership(parser: HeuristicIntentParser) -> None:
    assert isinstance(parser, IntentParser)


def test_empty_brief_yields_defaults(parser: HeuristicIntentParser) -> None:
    i = parser.parse("")
    assert i.raw_brief == ""
    assert i.languages == ["en"]
    assert i.threat_families == []
    assert i.sensitive_domain is SensitiveDomain.NONE
    assert i.deployment_context is None
    assert i.license_policy == LicensePolicy()
    assert i.must_have == []
    assert i.nice_to_have == []
    assert i.must_not == []
    assert i.min_strategy_confidence == 0.5


def test_minimal_brief_keeps_defaults(parser: HeuristicIntentParser) -> None:
    i = parser.parse("hello world")
    assert i.languages == ["en"]
    assert i.threat_families == []
    assert i.sensitive_domain is SensitiveDomain.NONE


@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("prompt injection", "prompt_injection"),
        ("prompt-injection", "prompt_injection"),
        ("prompt_injection", "prompt_injection"),
        ("indirect injection", "indirect_injection"),
        ("indirect prompt injection", "indirect_injection"),
        ("jailbreak", "jailbreak"),
        ("unsafe output", "unsafe_output"),
        ("unsafe-output", "unsafe_output"),
        ("harmful output", "unsafe_output"),
        ("exfiltration", "exfiltration"),
        ("data exfiltration", "exfiltration"),
        ("memory poisoning", "memory_poisoning"),
        ("context poisoning", "memory_poisoning"),
        ("tool call", "tool_use_safety"),
        ("tool-use", "tool_use_safety"),
        ("tool use", "tool_use_safety"),
        ("malware", "malware"),
    ],
)
def test_threat_family_token_in_isolation(
    parser: HeuristicIntentParser, phrase: str, expected: str
) -> None:
    i = parser.parse(f"detect {phrase} stuff")
    assert expected in i.threat_families


def test_indirect_does_not_double_count_with_prompt_injection(
    parser: HeuristicIntentParser,
) -> None:
    i = parser.parse("indirect prompt injection corpora")
    assert i.threat_families == ["indirect_injection"]


def test_prompt_injection_english(parser: HeuristicIntentParser) -> None:
    i = parser.parse("find me labeled prompt injection datasets in english")
    assert i.languages == ["en"]
    assert i.threat_families == ["prompt_injection"]


def test_japanese_jailbreak(parser: HeuristicIntentParser) -> None:
    i = parser.parse("japanese jailbreak corpora")
    assert i.languages == ["ja"]
    assert i.threat_families == ["jailbreak"]


def test_multilingual_multifamily_ordering(parser: HeuristicIntentParser) -> None:
    i = parser.parse("multilingual prompt injection and jailbreak data")
    # multilingual expands to a 4-language non-English-centric mix.
    assert set(i.languages) >= {"en", "zh", "es", "ar"}
    # Family ordering preserves first occurrence in the brief.
    assert i.threat_families == ["prompt_injection", "jailbreak"]


def test_malware_marks_sensitive_domain(parser: HeuristicIntentParser) -> None:
    i = parser.parse("malware classification dataset")
    assert "malware" in i.threat_families
    assert i.sensitive_domain is SensitiveDomain.GATED


def test_flag_language_overrides_parser(parser: HeuristicIntentParser) -> None:
    i = parser.parse("english brief", language=["ja"])
    assert i.languages == ["ja"]


def test_flag_license_builds_new_policy(parser: HeuristicIntentParser) -> None:
    i = parser.parse("anything", license=["MIT"])
    assert i.license_policy.allow == frozenset({"MIT"})


def test_flag_threat_families_override_drops_malware_gating(
    parser: HeuristicIntentParser,
) -> None:
    # Brief mentions malware, but the user explicitly narrows the
    # families — the override wins and the gating clears.
    i = parser.parse("malware variants", threat_families=["jailbreak"])
    assert i.threat_families == ["jailbreak"]
    assert i.sensitive_domain is SensitiveDomain.NONE


def test_flag_min_strategy_confidence_override(parser: HeuristicIntentParser) -> None:
    i = parser.parse("anything", min_strategy_confidence=0.8)
    assert i.min_strategy_confidence == 0.8


def test_flag_detection_target_and_deployment_context_override(
    parser: HeuristicIntentParser,
) -> None:
    i = parser.parse(
        "detect prompt injection in production",
        detection_target="custom target",
        deployment_context="custom-context",
    )
    assert i.detection_target == "custom target"
    assert i.deployment_context == "custom-context"


def test_unknown_flags_are_ignored(parser: HeuristicIntentParser) -> None:
    i = parser.parse("hello", made_up_flag=True, another=42)
    assert i.raw_brief == "hello"


def test_raw_brief_preserved_verbatim(parser: HeuristicIntentParser) -> None:
    brief = "  Find me   PROMPT INJECTION   data!  "
    i = parser.parse(brief)
    assert i.raw_brief == brief


def test_detection_target_inference_from_detect(parser: HeuristicIntentParser) -> None:
    i = parser.parse("detect indirect injection in agent traces")
    assert i.detection_target is not None
    assert "indirect injection" in i.detection_target.lower()


def test_detection_target_inference_classifier_for(parser: HeuristicIntentParser) -> None:
    i = parser.parse("classifier for jailbreak attempts")
    assert i.detection_target is not None
    assert "jailbreak" in i.detection_target.lower()


def test_detection_target_falls_back_to_brief_trimmed(parser: HeuristicIntentParser) -> None:
    long_brief = "x" * 500
    i = parser.parse(long_brief)
    assert i.detection_target is not None
    assert len(i.detection_target) <= 80


def test_conservative_no_spurious_families(parser: HeuristicIntentParser) -> None:
    i = parser.parse("hello world")
    assert i.threat_families == []
    assert i.sensitive_domain is SensitiveDomain.NONE


def test_deployment_context_detected(parser: HeuristicIntentParser) -> None:
    assert parser.parse("for our MCP server").deployment_context == "mcp_server"
    assert parser.parse("a RAG pipeline").deployment_context == "rag_pipeline"
    assert parser.parse("user-facing chatbot").deployment_context == "user-facing"
    assert parser.parse("production deployment").deployment_context == "production"
    assert parser.parse("agentic systems").deployment_context == "agent"
    assert parser.parse("nothing relevant here").deployment_context is None


def test_license_any_means_no_filtering(parser: HeuristicIntentParser) -> None:
    i = parser.parse("any license is fine")
    assert i.license_policy.allow == frozenset()
    assert i.license_policy.warn_only == frozenset()


def test_license_research_use_extends_default(parser: HeuristicIntentParser) -> None:
    i = parser.parse("research use ok please")
    assert "CC-BY-SA-4.0" in i.license_policy.allow
    assert "GPL-3.0" in i.license_policy.allow
    assert "MIT" in i.license_policy.allow  # default still present


def test_intent_is_frozen(parser: HeuristicIntentParser) -> None:
    i = parser.parse("hello")
    with pytest.raises(ValidationError):
        i.raw_brief = "mutated"  # type: ignore[misc]


def test_multilingual_plus_explicit_language_unions(parser: HeuristicIntentParser) -> None:
    i = parser.parse("multilingual including japanese sources")
    assert "ja" in i.languages
    for code in ("en", "zh", "es", "ar"):
        assert code in i.languages


def test_languages_first_occurrence_order(parser: HeuristicIntentParser) -> None:
    i = parser.parse("japanese and chinese and english")
    assert i.languages == ["ja", "zh", "en"]
