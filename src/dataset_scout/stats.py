"""Wilson score confidence intervals.

A tiny, dependency-free implementation. Used by `inspect` to put
reasonable error bars on label-distribution estimates from small
samples so users don't over-interpret a 50-row peek.

Reference: Wilson, E.B. (1927). "Probable inference, the law of
succession, and statistical inference." Journal of the American
Statistical Association.
"""

from __future__ import annotations

import math


def wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score CI for a binomial proportion.

    successes -- count of positive observations
    n         -- sample size
    confidence -- two-sided coverage (default 0.95)

    Returns (lower, upper) on [0, 1]. Falls back to (0, 1) for n=0.
    """
    if n <= 0:
        return (0.0, 1.0)
    if not (0 <= successes <= n):
        raise ValueError(f"successes ({successes}) must be in [0, {n}]")

    z = _z_for(confidence)
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)

    lower = (centre - spread) / denom
    upper = (centre + spread) / denom
    return (max(0.0, lower), min(1.0, upper))


def _z_for(confidence: float) -> float:
    """Two-sided z* for common confidence levels.

    Hard-coded common values rather than pulling in scipy or hand-rolling
    the inverse normal CDF — Wilson's CI doesn't need more precision
    than this for the small-N reports `inspect` produces.
    """
    table = {
        0.80: 1.2816,
        0.90: 1.6449,
        0.95: 1.96,
        0.98: 2.3263,
        0.99: 2.5758,
    }
    if confidence in table:
        return table[confidence]
    raise ValueError(f"unsupported confidence level {confidence}; supported: {sorted(table)}")
