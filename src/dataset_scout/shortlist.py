"""Two-stage shortlist for the LLM strategy assessor.

The assessor is per-candidate and expensive (one LLM call each). To
keep costs and latency reasonable we send only the most promising
~15-20 candidates. Selection is two-stage per duck guidance:

Stage 1 — breadth: ensure each surfacing direction (plus the original
Intent) gets representation. Take the top-k from each group.

Stage 2 — global shortlist: re-rank the union by cheap non-LLM signals
(multi-direction hits, license sanity, card completeness) and cap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dataset_scout.core import Scorecard


# Keys for the stage-1 grouping. None means "the original Intent".
_OriginalKey = None | str


def _primary_direction(sc: Scorecard) -> _OriginalKey:
    """Return the direction we'll attribute this candidate to in stage 1.

    Multi-direction hits are attributed to their first surfacing
    direction; original-only hits are attributed to None.
    """
    if not sc.candidate.surfaced_by:
        return None
    return sc.candidate.surfaced_by[0]


def _quality_signal(sc: Scorecard) -> tuple[int, float, float, int]:
    """A cheap sort key for stage 2; higher is better.

    1. Number of surfacing directions (multi-direction hits float up).
    2. License probe value (`1.0` allow / `0.5` warn / `0.0` outside).
    3. card_completeness probe value.
    4. Negation of the original index — preserves intra-tie HF order
       when the rest of the key matches.
    """
    multi = len(sc.candidate.surfaced_by)
    license_sub = sc.cheap_probes.get("license")
    license_value = license_sub.value if license_sub and license_sub.value is not None else 0.0
    cc_sub = sc.cheap_probes.get("card_completeness")
    cc_value = cc_sub.value if cc_sub and cc_sub.value is not None else 0.0
    return (multi, license_value, cc_value, 0)


def select_top_for_assessor(
    scorecards: list[Scorecard],
    *,
    top_per_direction: int = 5,
    total_cap: int = 20,
) -> list[Scorecard]:
    """Two-stage shortlist for the strategy assessor.

    Returns a NEW list; the input is not mutated. Order in the result
    reflects the stage-2 re-ranking (highest quality first).
    """
    if not scorecards:
        return []

    # ── Stage 1: top-k per surfacing direction (or original-Intent group) ──
    by_direction: dict[_OriginalKey, list[Scorecard]] = {}
    for sc in scorecards:
        by_direction.setdefault(_primary_direction(sc), []).append(sc)

    union: list[Scorecard] = []
    seen: set[tuple[str, str]] = set()
    for picks in by_direction.values():
        for sc in picks[:top_per_direction]:
            key = (sc.candidate.source, sc.candidate.id)
            if key in seen:
                continue
            seen.add(key)
            union.append(sc)

    # ── Stage 2: re-rank globally by cheap quality signals ──
    union.sort(key=_quality_signal, reverse=True)
    return union[:total_cap]
