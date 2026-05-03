"""Unit tests for the Wilson CI helper."""

from __future__ import annotations

import math

import pytest

from dataset_scout.stats import wilson_ci

pytestmark = pytest.mark.unit


def test_zero_n_returns_full_range():
    assert wilson_ci(0, 0) == (0.0, 1.0)


def test_p_within_ci_for_classic_examples():
    """Wilson CI should always contain the point estimate."""
    for s, n in [(5, 10), (1, 4), (95, 100), (0, 50), (50, 50)]:
        lo, hi = wilson_ci(s, n)
        p = s / n
        assert lo <= p <= hi


def test_ci_shrinks_as_n_grows():
    lo10, hi10 = wilson_ci(5, 10)
    lo100, hi100 = wilson_ci(50, 100)
    assert (hi100 - lo100) < (hi10 - lo10)


def test_known_value_50_in_100():
    """Known Wilson 95% CI for 50/100 ≈ (0.402, 0.598)."""
    lo, hi = wilson_ci(50, 100)
    assert math.isclose(lo, 0.40, abs_tol=0.01)
    assert math.isclose(hi, 0.60, abs_tol=0.01)


def test_rejects_invalid_input():
    with pytest.raises(ValueError):
        wilson_ci(11, 10)
    with pytest.raises(ValueError):
        wilson_ci(-1, 10)
    with pytest.raises(ValueError):
        wilson_ci(5, 10, confidence=0.42)
