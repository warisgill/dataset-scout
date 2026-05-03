"""Tests for cheap, metadata-driven probes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from dataset_scout import (
    Candidate,
    CandidateMetadata,
    Intent,
    LicensePolicy,
)
from dataset_scout.probes import (
    CardCompletenessProbe,
    FreshnessProbe,
    LanguagesProbe,
    LicenseProbe,
    Probe,
    RecencyProbe,
    SizeProbe,
    cheap_probes,
)

pytestmark = pytest.mark.unit


def _cand(meta: CandidateMetadata) -> Candidate:
    return Candidate(source="huggingface", id="org/x", revision="r1", metadata=meta)


def _intent(**overrides) -> Intent:
    return Intent(raw_brief="test", **overrides)


# ─── registry / protocol conformance ────────────────────────────────


def test_cheap_probes_returns_all_six():
    reg = cheap_probes()
    assert reg.names() == [
        "license",
        "size",
        "recency",
        "freshness",
        "languages",
        "card_completeness",
    ]
    assert len(reg) == 6


def test_each_probe_satisfies_the_protocol():
    for probe in cheap_probes():
        assert isinstance(probe, Probe)


def test_each_probe_has_name_and_version():
    for probe in cheap_probes():
        assert isinstance(probe.name, str) and probe.name
        assert isinstance(probe.version, str) and probe.version


# ─── LicenseProbe ──────────────────────────────────────────────────


def test_license_probe_recognizes_allow_set():
    cand = _cand(CandidateMetadata(license_raw="mit", license_spdx="MIT"))
    sub = LicenseProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.value == 1.0


def test_license_probe_recognizes_warn_only():
    cand = _cand(CandidateMetadata(license_raw="cc-by-sa-4.0", license_spdx="CC-BY-SA-4.0"))
    sub = LicenseProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.value == 0.5


def test_license_probe_outside_policy():
    cand = _cand(CandidateMetadata(license_raw="cc-by-nc-4.0", license_spdx="CC-BY-NC-4.0"))
    sub = LicenseProbe().run(
        cand,
        _intent(license_policy=LicensePolicy(allow=frozenset({"MIT"}), warn_only=frozenset())),
    )
    assert sub.status == "ok"
    assert sub.value == 0.0


def test_license_probe_low_confidence_when_only_raw():
    cand = _cand(CandidateMetadata(license_raw="see-license-file"))
    sub = LicenseProbe().run(cand, _intent())
    assert sub.status == "low_confidence"
    assert sub.value is None


def test_license_probe_not_applicable_when_missing():
    cand = _cand(CandidateMetadata())
    sub = LicenseProbe().run(cand, _intent())
    assert sub.status == "not_applicable"


# ─── SizeProbe ─────────────────────────────────────────────────────


def test_size_probe_uses_rows_when_present():
    cand = _cand(CandidateMetadata(rows=10_000, bytes=20_000_000, downloads=99))
    sub = SizeProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.n == 10_000
    kinds = {e.kind for e in sub.evidence}
    assert {"rows", "bytes", "downloads"} <= kinds


def test_size_probe_falls_back_to_bytes():
    cand = _cand(CandidateMetadata(bytes=4096))
    sub = SizeProbe().run(cand, _intent())
    assert sub.n == 4096


def test_size_probe_not_applicable_when_silent():
    cand = _cand(CandidateMetadata())
    sub = SizeProbe().run(cand, _intent())
    assert sub.status == "not_applicable"


# ─── RecencyProbe ──────────────────────────────────────────────────


def test_recency_probe_reports_days_since():
    now = datetime.now(UTC)
    cand = _cand(
        CandidateMetadata(
            uploaded_at=now - timedelta(days=30),
            last_modified=now - timedelta(days=5),
        )
    )
    sub = RecencyProbe().run(cand, _intent())
    assert sub.status == "ok"
    # n is the smaller of the two (most recent activity).
    assert sub.n == 5
    kinds = {e.kind for e in sub.evidence}
    assert "days_since_upload" in kinds
    assert "days_since_last_modified" in kinds


def test_recency_probe_applies_only_with_dates():
    cand = _cand(CandidateMetadata())
    assert RecencyProbe().applies(cand, _intent()) is False


# ─── FreshnessProbe ────────────────────────────────────────────────


@pytest.mark.parametrize(
    "days_ago,expected_bucket,expected_value",
    [
        (10, "fresh", 0.9),
        (90, "fresh", 0.9),
        (200, "current", 0.6),
        (400, "current", 0.6),
        (600, "aging", 0.3),
        (3_000, "aging", 0.3),
    ],
)
def test_freshness_probe_buckets(days_ago: int, expected_bucket: str, expected_value: float):
    now = datetime.now(UTC)
    cand = _cand(CandidateMetadata(last_modified=now - timedelta(days=days_ago)))
    sub = FreshnessProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.value == expected_value
    assert any(e.kind == "bucket" and e.detail == expected_bucket for e in sub.evidence)


def test_freshness_probe_uses_most_recent_signal():
    now = datetime.now(UTC)
    cand = _cand(
        CandidateMetadata(
            uploaded_at=now - timedelta(days=1000),  # aging
            last_modified=now - timedelta(days=10),  # fresh
        )
    )
    sub = FreshnessProbe().run(cand, _intent())
    # Most recent signal wins.
    assert sub.value == 0.9


def test_freshness_probe_not_applicable_when_no_dates():
    cand = _cand(CandidateMetadata())
    sub = FreshnessProbe().run(cand, _intent())
    assert sub.status == "not_applicable"


# ─── LanguagesProbe ────────────────────────────────────────────────


def test_languages_probe_full_overlap():
    cand = _cand(CandidateMetadata(languages_declared=["en", "ja"]))
    sub = LanguagesProbe().run(cand, _intent(languages=["en"]))
    assert sub.status == "ok"
    assert sub.value == 1.0


def test_languages_probe_partial_overlap():
    cand = _cand(CandidateMetadata(languages_declared=["en"]))
    sub = LanguagesProbe().run(cand, _intent(languages=["en", "ja"]))
    assert sub.status == "ok"
    assert sub.value == 0.5


def test_languages_probe_no_overlap():
    cand = _cand(CandidateMetadata(languages_declared=["zh"]))
    sub = LanguagesProbe().run(cand, _intent(languages=["en"]))
    assert sub.status == "ok"
    assert sub.value == 0.0


def test_languages_probe_not_applicable_when_card_silent():
    cand = _cand(CandidateMetadata())
    sub = LanguagesProbe().run(cand, _intent())
    assert sub.status == "not_applicable"


def test_languages_probe_is_case_insensitive():
    cand = _cand(CandidateMetadata(languages_declared=["EN"]))
    sub = LanguagesProbe().run(cand, _intent(languages=["en"]))
    assert sub.value == 1.0


# ─── CardCompletenessProbe ─────────────────────────────────────────


def test_card_completeness_full():
    cand = _cand(
        CandidateMetadata(
            card_fields_present=frozenset(
                {"license", "language", "task_categories", "size_categories", "pretty_name"}
            )
        )
    )
    sub = CardCompletenessProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.value == 1.0


def test_card_completeness_empty():
    cand = _cand(CandidateMetadata())
    sub = CardCompletenessProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.value == 0.0


def test_card_completeness_partial_lists_missing():
    cand = _cand(CandidateMetadata(card_fields_present=frozenset({"license", "language"})))
    sub = CardCompletenessProbe().run(cand, _intent())
    assert sub.status == "ok"
    assert sub.value == pytest.approx(2 / 5)
    missing_evidence = next(e for e in sub.evidence if e.kind == "missing")
    assert "task_categories" in missing_evidence.detail
    assert "pretty_name" in missing_evidence.detail


def test_card_completeness_ignores_unrelated_card_fields():
    """Fields outside the expected set count for nothing."""
    cand = _cand(CandidateMetadata(card_fields_present=frozenset({"some", "other", "fields"})))
    sub = CardCompletenessProbe().run(cand, _intent())
    assert sub.value == 0.0
