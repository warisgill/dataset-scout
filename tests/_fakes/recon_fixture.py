"""Canned `ReconResult` fixture for renderer / report tests.

Used by `test_html_report`, `test_report_redesign`, and the render
round-trip tests in `test_cli`. Mirrors a realistic mini-recon for
an over-refusal detection program so renderer assertions can exercise
all sections (decomposition, candidates with strategies, coverage
gaps, notices).

Test infrastructure only — no production code path imports this.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from dataset_scout.core import (
    Candidate,
    CandidateMetadata,
    CoverageGap,
    CoverageReport,
    DecompositionDirection,
    Evidence,
    Intent,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    SubScore,
    TransformSpec,
)

_NOW = datetime.now(UTC)


def build_demo_recon_result() -> ReconResult:
    """Construct a fully-populated `ReconResult` for renderer tests."""
    intent = Intent(
        raw_brief="refusal-labeled corpora for customer-support agents",
        detection_target="over-refusal in customer support",
        threat_families=["over_refusal"],
        deployment_context="customer support agent",
        languages=["en"],
    )

    directions = [
        DecompositionDirection(
            name="safety_refusal_benchmarks",
            rationale="Direct fits live in benchmarks designed to study refusal behaviour on borderline-but-benign prompts.",
            keywords=["over-refusal benchmark", "xstest", "or-bench"],
            threat_families=["over_refusal"],
            expected_finds="Datasets with explicit refusal-vs-comply labels.",
        ),
        DecompositionDirection(
            name="customer_support_dialogue",
            rationale="Real support conversations as benign baselines and realistic user-intent distributions.",
            keywords=["customer support dataset", "ticket dialogues", "helpdesk corpus"],
            threat_families=[],
            expected_finds="Benign conversational data scoped to support workflows.",
        ),
        DecompositionDirection(
            name="helpfulness_harmlessness_pairs",
            rationale="RLHF-style preference data captures the helpful/harmless tradeoff that drives over-refusal.",
            keywords=["helpfulness dataset", "harmlessness preference", "rlhf"],
            threat_families=[],
            expected_finds="Paired prompts with helpful + refusal responses.",
        ),
    ]

    candidates: list[Scorecard] = [
        _scorecard(
            "huggingface",
            "bench-llm/or-bench",
            "Benchmark of over-refusal triggers on benign prompts.",
            license_raw="cc-by-4.0",
            license_spdx="CC-BY-4.0",
            languages=["en"],
            downloads=4321,
            uploaded_days_ago=80,
            surfaced_by=["safety_refusal_benchmarks"],
            strategies=[
                _strategy(
                    StrategyKind.DIRECT_USE,
                    0.88,
                    "Purpose-built for over-refusal detection. The `category` column is the refusal label; map directly.",
                    text_col="prompt",
                    label_col="category",
                    label_map={"over-refusal": "positive", "ok": "benign"},
                ),
            ],
        ),
        _scorecard(
            "huggingface",
            "walledai/XSTest",
            "Exaggerated-safety test suite.",
            license_raw="apache-2.0",
            license_spdx="Apache-2.0",
            languages=["en"],
            downloads=2150,
            uploaded_days_ago=200,
            surfaced_by=["safety_refusal_benchmarks"],
            strategies=[
                _strategy(
                    StrategyKind.DIRECT_USE,
                    0.82,
                    "XSTest's safe prompts are the canonical 'should not be refused' test set; over-refusals on these rows are positives.",
                    text_col="prompt",
                    label_col="type",
                    label_map={"safe": "positive"},
                ),
            ],
        ),
        _scorecard(
            "huggingface",
            "Anthropic/hh-rlhf",
            "Helpful + harmless preference data.",
            license_raw="mit",
            license_spdx="MIT",
            languages=["en"],
            downloads=33500,
            uploaded_days_ago=520,
            surfaced_by=["helpfulness_harmlessness_pairs"],
            strategies=[
                _strategy(
                    StrategyKind.SIGNAL_PROXY,
                    0.55,
                    "Refusal-shaped chosen responses can serve as proxy positives during cold-start training; do NOT use for evaluation.",
                    text_col="prompt",
                    label_col="chosen_label",
                    label_map={"refused": "positive", "complied": "benign"},
                ),
            ],
        ),
        _scorecard(
            "huggingface",
            "matefh/bitext-customer-support-intent-classification",
            "Customer support intent classification corpus.",
            license_raw="cdla-sharing-1.0",
            license_spdx=None,
            languages=["en"],
            downloads=560,
            uploaded_days_ago=120,
            surfaced_by=["customer_support_dialogue"],
            strategies=[
                _strategy(
                    StrategyKind.BENIGN_BASELINE,
                    0.62,
                    "Real support utterances make a strong benign baseline; the support-domain language matters for detector calibration.",
                    text_col="utterance",
                    label_col=None,
                    label_map={"all": "benign"},
                ),
            ],
        ),
    ]

    coverage = CoverageReport(
        decomposition=directions,
        semantic_gaps=[
            CoverageGap(
                aspect="multi_turn_procedural_handling",
                description="Refusal benchmarks are single-turn; correct support behaviour often involves clarification or escalation, not simple comply/refuse.",
                suggestion="Augment with task-oriented support dialogue datasets where labels distinguish 'asking for missing info' from 'refusing inappropriately'.",
            ),
            CoverageGap(
                aspect="policy_decision_boundary_labels",
                description="Over-refusals frequently happen near policy boundaries (account access, privacy). The candidates lack policy-decision labels.",
                suggestion="Source or synthesise prompts with allow / allow-with-steps / escalate / deny labels using your own product policy.",
            ),
            CoverageGap(
                aspect="non_english_support_over_refusal",
                description="Candidates are English-only; multilingual support traffic is the long tail of real over-refusal incidents.",
                suggestion="Translate the strongest English refusal benchmarks with human validation, or source multilingual support corpora.",
            ),
        ],
    )

    return ReconResult(
        intent=intent,
        candidates=candidates,
        sources_searched=["huggingface (canned demo fixture)"],
        coverage=coverage,
        elapsed_seconds=0.42,
        notices=[
            "Demo fixture: candidates and strategies are canned data for renderer tests, not a real recon."
        ],
    )


def _scorecard(
    source: str,
    sid: str,
    description: str,
    *,
    license_raw: str,
    license_spdx: str | None,
    languages: list[str],
    downloads: int,
    uploaded_days_ago: int,
    surfaced_by: list[str],
    strategies: list[Strategy],
) -> Scorecard:
    meta = CandidateMetadata(
        description=description,
        card_url=f"https://huggingface.co/datasets/{sid}",
        license_raw=license_raw,
        license_spdx=license_spdx,
        languages_declared=languages,
        downloads=downloads,
        uploaded_at=_NOW - timedelta(days=uploaded_days_ago),
        last_modified=_NOW - timedelta(days=max(1, uploaded_days_ago // 2)),
        card_fields_present=frozenset({"license", "language", "task_categories"}),
    )
    cand = Candidate(
        source=source,
        id=sid,
        revision="abc123def456",
        metadata=meta,
        surfaced_by=surfaced_by,
    )
    cheap: dict[str, SubScore] = {
        "license": SubScore(
            value=1.0 if license_spdx in {"MIT", "Apache-2.0", "CC-BY-4.0"} else None,
            evidence=[
                Evidence(kind="license_spdx", detail=license_spdx or "(unknown)"),
                Evidence(
                    kind="policy_match",
                    detail=(
                        "allow"
                        if license_spdx in {"MIT", "Apache-2.0", "CC-BY-4.0"}
                        else "outside_policy"
                    ),
                ),
            ],
        ),
        "freshness": SubScore(
            value=0.9 if uploaded_days_ago < 180 else 0.6,
            evidence=[
                Evidence(
                    kind="bucket",
                    detail="fresh" if uploaded_days_ago < 180 else "current",
                ),
            ],
        ),
    }
    return Scorecard(candidate=cand, cheap_probes=cheap, strategies=strategies)


def _strategy(
    kind: StrategyKind,
    confidence: float,
    rationale: str,
    *,
    text_col: str | None,
    label_col: str | None,
    label_map: dict[str, str],
) -> Strategy:
    return Strategy(
        kind=kind,
        confidence=confidence,
        rationale=rationale,
        transform=TransformSpec(
            text_column=text_col,
            label_column=label_col,
            label_value_map={
                k: v for k, v in label_map.items() if v in ("positive", "benign", "hard_negative")
            },
        ),
    )
