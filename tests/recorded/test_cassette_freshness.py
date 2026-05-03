"""Fail CI when recorded HTTP cassettes go stale.

External API shapes drift. A cassette older than the threshold is more
likely to be lying than telling the truth, so we force a re-record.
"""

from __future__ import annotations

import time
from datetime import timedelta
from pathlib import Path

import pytest

# 90 days ≈ one quarter. Long enough that re-recording is rare;
# short enough that any breaking upstream change is caught within a
# release cycle. Tune (with a code review) if real pain emerges.
FRESHNESS_THRESHOLD_DAYS = 90

CASSETTE_DIR = Path(__file__).parent / "cassettes"

pytestmark = pytest.mark.recorded


def _candidate_cassettes(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return [p for p in directory.iterdir() if p.is_file() and not p.name.startswith(".")]


def test_recorded_cassettes_are_fresh() -> None:
    cassettes = _candidate_cassettes(CASSETTE_DIR)
    if not cassettes:
        pytest.skip(
            "No recorded cassettes yet. The freshness gate activates as soon as "
            f"any non-hidden file lands in {CASSETTE_DIR}."
        )

    threshold = timedelta(days=FRESHNESS_THRESHOLD_DAYS).total_seconds()
    now = time.time()
    stale = [
        (p, (now - p.stat().st_mtime) / 86400.0)
        for p in cassettes
        if (now - p.stat().st_mtime) > threshold
    ]

    if stale:
        lines = [
            f"  - {p.relative_to(CASSETTE_DIR.parent)} (age: {age_days:.1f} days)"
            for p, age_days in stale
        ]
        msg = (
            f"{len(stale)} cassette(s) older than {FRESHNESS_THRESHOLD_DAYS} days. "
            "Re-record them against the live API to keep the recorded suite honest:\n"
            + "\n".join(lines)
        )
        pytest.fail(msg)
