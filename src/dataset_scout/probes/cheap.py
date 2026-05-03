"""Cheap, metadata-driven probes (M1a).

Six probes consuming only `CandidateMetadata`. No row sampling, no LLM,
no embedding. Each emits a `SubScore` carrying the raw signal plus
evidence; the discovery slice surfaces these as annotations rather than
folding them into a single "quality" number.

Probes:
    license, size, recency, freshness, languages, card_completeness

Sample-driven probes (label_structure, schema_fingerprint) land in M1b
when sources support `stream_sample`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from dataset_scout.core import (
    Candidate,
    Evidence,
    Intent,
    SubScore,
)
from dataset_scout.probes.base import ProbeRegistry

# ─── helpers ────────────────────────────────────────────────────────


def _now() -> datetime:
    """Indirection so tests can monkeypatch the clock if they ever need to."""
    return datetime.now(UTC)


def _aware(dt: datetime) -> datetime:
    """Coerce naive datetimes to UTC. HF returns aware datetimes already."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt


def _days_since(dt: datetime | None) -> int | None:
    if dt is None:
        return None
    return max(0, (_now() - _aware(dt)).days)


# ─── license ────────────────────────────────────────────────────────


class LicenseProbe:
    """Reports the candidate's declared license against the user's policy.

    `value` semantics:
        1.0  - SPDX guess is in the allow set
        0.5  - SPDX guess is in the warn_only set
        0.0  - SPDX guess is outside both sets (potentially incompatible)
        None - no SPDX guess available; raw string (if any) is in evidence
    """

    name: str = "license"
    version: str = "1"

    def applies(self, candidate: Candidate, intent: Intent) -> bool:
        return True

    def run(self, candidate: Candidate, intent: Intent) -> SubScore:
        meta = candidate.metadata
        evidence: list[Evidence] = []
        if meta.license_raw:
            evidence.append(Evidence(kind="license_raw", detail=meta.license_raw))
        if meta.license_spdx:
            evidence.append(Evidence(kind="license_spdx", detail=meta.license_spdx))

        if not meta.license_spdx:
            status = "low_confidence" if meta.license_raw else "not_applicable"
            return SubScore(
                value=None,
                status=status,
                evidence=evidence,
                probe_version=self.version,
            )

        spdx = meta.license_spdx
        policy = intent.license_policy
        if spdx in policy.allow:
            value = 1.0
            evidence.append(Evidence(kind="policy_match", detail="allow", value=1.0))
        elif spdx in policy.warn_only:
            value = 0.5
            evidence.append(Evidence(kind="policy_match", detail="warn_only", value=0.5))
        else:
            value = 0.0
            evidence.append(Evidence(kind="policy_match", detail="outside_policy", value=0.0))

        return SubScore(
            value=value,
            status="ok",
            evidence=evidence,
            probe_version=self.version,
        )


# ─── size ───────────────────────────────────────────────────────────


class SizeProbe:
    """Reports row count, byte count, and download popularity.

    No `value` is reported — size doesn't map onto a 0-1 score. Use the
    `n` field (set to row count when known, else byte count) for display.
    """

    name: str = "size"
    version: str = "1"

    def applies(self, candidate: Candidate, intent: Intent) -> bool:
        return True

    def run(self, candidate: Candidate, intent: Intent) -> SubScore:
        meta = candidate.metadata
        evidence: list[Evidence] = []
        if meta.rows is not None:
            evidence.append(Evidence(kind="rows", detail=str(meta.rows), value=float(meta.rows)))
        if meta.bytes is not None:
            evidence.append(Evidence(kind="bytes", detail=str(meta.bytes), value=float(meta.bytes)))
        if meta.downloads is not None:
            evidence.append(
                Evidence(kind="downloads", detail=str(meta.downloads), value=float(meta.downloads))
            )
        if meta.likes is not None:
            evidence.append(Evidence(kind="likes", detail=str(meta.likes), value=float(meta.likes)))

        n = meta.rows if meta.rows is not None else meta.bytes
        if not evidence:
            return SubScore(
                value=None,
                status="not_applicable",
                evidence=[Evidence(kind="size", detail="no size signals declared")],
                probe_version=self.version,
            )
        return SubScore(
            value=None,
            n=n,
            status="ok",
            evidence=evidence,
            probe_version=self.version,
        )


# ─── recency ────────────────────────────────────────────────────────


class RecencyProbe:
    """Reports raw days-since for upload and last-modified timestamps.

    Discovery slice: we display this as an annotation; we do NOT score
    on recency in M1a (recency alone means little).
    """

    name: str = "recency"
    version: str = "1"

    def applies(self, candidate: Candidate, intent: Intent) -> bool:
        meta = candidate.metadata
        return meta.uploaded_at is not None or meta.last_modified is not None

    def run(self, candidate: Candidate, intent: Intent) -> SubScore:
        meta = candidate.metadata
        evidence: list[Evidence] = []
        days_uploaded = _days_since(meta.uploaded_at)
        days_modified = _days_since(meta.last_modified)
        if days_uploaded is not None:
            evidence.append(
                Evidence(
                    kind="days_since_upload",
                    detail=str(days_uploaded),
                    value=float(days_uploaded),
                )
            )
        if days_modified is not None:
            evidence.append(
                Evidence(
                    kind="days_since_last_modified",
                    detail=str(days_modified),
                    value=float(days_modified),
                )
            )
        if not evidence:
            return SubScore(
                value=None,
                status="not_applicable",
                evidence=[Evidence(kind="recency", detail="no dates declared")],
                probe_version=self.version,
            )
        # n = the most recent (smallest days-since) of the two.
        candidates = [d for d in (days_uploaded, days_modified) if d is not None]
        n = min(candidates) if candidates else None
        return SubScore(
            value=None,
            n=n,
            status="ok",
            evidence=evidence,
            probe_version=self.version,
        )


# ─── freshness ──────────────────────────────────────────────────────


# Bucket thresholds in days. fresh < 180; current 180-540; aging > 540.
# 6 months = 180d, 18 months = 540d.
_FRESH_DAYS = 180
_CURRENT_DAYS = 540


class FreshnessProbe:
    """Buckets the candidate into fresh / current / aging / unknown.

    `value` semantics:
        0.9 - fresh   (< 6 months)
        0.6 - current (6-18 months)
        0.3 - aging   (> 18 months)
        None - no dates declared
    """

    name: str = "freshness"
    version: str = "1"

    def applies(self, candidate: Candidate, intent: Intent) -> bool:
        meta = candidate.metadata
        return meta.uploaded_at is not None or meta.last_modified is not None

    def run(self, candidate: Candidate, intent: Intent) -> SubScore:
        meta = candidate.metadata
        d_upload = _days_since(meta.uploaded_at)
        d_modified = _days_since(meta.last_modified)
        candidates = [d for d in (d_upload, d_modified) if d is not None]
        if not candidates:
            return SubScore(
                value=None,
                status="not_applicable",
                evidence=[Evidence(kind="freshness", detail="no dates declared")],
                probe_version=self.version,
            )
        days = min(candidates)
        if days < _FRESH_DAYS:
            bucket, value = "fresh", 0.9
        elif days < _CURRENT_DAYS:
            bucket, value = "current", 0.6
        else:
            bucket, value = "aging", 0.3
        return SubScore(
            value=value,
            n=days,
            status="ok",
            evidence=[
                Evidence(kind="bucket", detail=bucket, value=value),
                Evidence(kind="days_since_most_recent", detail=str(days), value=float(days)),
            ],
            probe_version=self.version,
        )


# ─── languages ──────────────────────────────────────────────────────


class LanguagesProbe:
    """Reports declared-language overlap with the user's intent.

    `value` is the fraction of intent languages found in the candidate's
    declared set. 1.0 = full coverage; 0.0 = no overlap; None = card
    declares no languages at all.
    """

    name: str = "languages"
    version: str = "1"

    def applies(self, candidate: Candidate, intent: Intent) -> bool:
        return True

    def run(self, candidate: Candidate, intent: Intent) -> SubScore:
        declared = candidate.metadata.languages_declared
        if not declared:
            return SubScore(
                value=None,
                status="not_applicable",
                evidence=[Evidence(kind="languages", detail="card declares no languages")],
                probe_version=self.version,
            )
        intent_set = {lang.lower() for lang in intent.languages}
        declared_set = {lang.lower() for lang in declared}
        overlap = intent_set & declared_set
        value = len(overlap) / len(intent_set) if intent_set else 0.0
        return SubScore(
            value=value,
            n=len(declared),
            status="ok",
            evidence=[
                Evidence(kind="declared", detail=", ".join(sorted(declared_set))),
                Evidence(kind="intent", detail=", ".join(sorted(intent_set))),
                Evidence(
                    kind="overlap",
                    detail=", ".join(sorted(overlap)) if overlap else "(none)",
                    value=float(len(overlap)),
                ),
            ],
            probe_version=self.version,
        )


# ─── card completeness ──────────────────────────────────────────────


# Fields we expect a well-cared-for dataset card to declare.
_EXPECTED_CARD_FIELDS = frozenset(
    {
        "license",
        "language",
        "task_categories",
        "size_categories",
        "pretty_name",
    }
)


class CardCompletenessProbe:
    """Fraction of expected card-YAML fields actually declared.

    Documentation-hygiene annotation, not a major rank component.
    `value` ∈ [0, 1].
    """

    name: str = "card_completeness"
    version: str = "1"

    def applies(self, candidate: Candidate, intent: Intent) -> bool:
        return True

    def run(self, candidate: Candidate, intent: Intent) -> SubScore:
        present = candidate.metadata.card_fields_present
        present_expected = _EXPECTED_CARD_FIELDS & present
        missing = _EXPECTED_CARD_FIELDS - present
        value = len(present_expected) / len(_EXPECTED_CARD_FIELDS)
        return SubScore(
            value=value,
            n=len(_EXPECTED_CARD_FIELDS),
            status="ok",
            evidence=[
                Evidence(
                    kind="present",
                    detail=", ".join(sorted(present_expected)) or "(none)",
                ),
                Evidence(
                    kind="missing",
                    detail=", ".join(sorted(missing)) or "(none)",
                ),
            ],
            probe_version=self.version,
        )


# ─── registry helper ────────────────────────────────────────────────


def cheap_probes() -> ProbeRegistry:
    """Return the canonical set of cheap probes for M1a."""
    return ProbeRegistry(
        [
            LicenseProbe(),
            SizeProbe(),
            RecencyProbe(),
            FreshnessProbe(),
            LanguagesProbe(),
            CardCompletenessProbe(),
        ]
    )
