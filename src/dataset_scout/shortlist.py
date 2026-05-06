"""Two-stage shortlist for the LLM strategy assessor.

The assessor is per-candidate and expensive (one LLM call each). To
keep costs and latency reasonable we send only the most promising
~20-35 candidates. Selection is two-stage per duck guidance:

Stage 0 — recalled-name rescue: if a candidate id matches a name the
LLM recalled in any decomposition direction (case-insensitive
substring), force-include it regardless of retrieval rank. These are
known canonical benchmarks the LLM expects to be useful and shouldn't
fall off the shortlist just because HF id-search ranked them low.

Stage 1 — breadth: ensure each surfacing direction (plus the original
Intent) gets representation. Take the top-k from each group.

Stage 2 — global shortlist: re-rank the union by cheap non-LLM signals
(multi-direction hits, license sanity, card completeness) and cap.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
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


_NAME_BOUNDARY = re.compile(r"[^a-z0-9]+")


def _normalise_for_match(s: str) -> str:
    """Lowercase + strip non-alphanum, so 'Persona-Chat' ~ 'PersonaChat'."""
    return _NAME_BOUNDARY.sub("", s.lower())


def _matches_recalled(sc: Scorecard, recalled_normalised: list[str]) -> bool:
    """True if the candidate id contains any recalled name as a substring.

    Match is on the alphanum-only normalised form so 'AI-companionship/INTIMA'
    matches recalled name 'INTIMA', 'PersonaChat' matches both 'Persona Chat'
    and 'persona-chat', etc. We also check the dataset id's last path segment
    so 'org/Foo-Bar' matches recalled 'Foo Bar' without polluting via org slug.
    """
    if not recalled_normalised:
        return False
    full = _normalise_for_match(sc.candidate.id)
    tail = _normalise_for_match(sc.candidate.id.split("/", 1)[-1])
    return any(n in full or n in tail for n in recalled_normalised)


def select_top_for_assessor(
    scorecards: list[Scorecard],
    *,
    top_per_direction: int = 5,
    total_cap: int = 35,
    recalled_names: Iterable[str] = (),
) -> list[Scorecard]:
    """Two-stage (+rescue) shortlist for the strategy assessor.

    Returns a NEW list; the input is not mutated. Order in the result
    reflects the stage-2 re-ranking (highest quality first), with
    recalled-name rescues kept regardless of where they would have
    landed under the cheap quality signals.
    """
    if not scorecards:
        return []

    recalled_norm = [_normalise_for_match(n) for n in recalled_names if n.strip()]
    recalled_norm = [n for n in recalled_norm if n]  # drop empties post-norm

    # ── Stage 0: recalled-name rescue ──
    rescued: list[Scorecard] = []
    rescued_keys: set[tuple[str, str]] = set()
    if recalled_norm:
        for sc in scorecards:
            if _matches_recalled(sc, recalled_norm):
                key = (sc.candidate.source, sc.candidate.id)
                if key not in rescued_keys:
                    rescued_keys.add(key)
                    rescued.append(sc)

    # ── Stage 1: top-k per surfacing direction (or original-Intent group) ──
    by_direction: dict[_OriginalKey, list[Scorecard]] = {}
    for sc in scorecards:
        by_direction.setdefault(_primary_direction(sc), []).append(sc)

    union: list[Scorecard] = []
    seen: set[tuple[str, str]] = set(rescued_keys)
    for picks in by_direction.values():
        for sc in picks[:top_per_direction]:
            key = (sc.candidate.source, sc.candidate.id)
            if key in seen:
                continue
            seen.add(key)
            union.append(sc)

    # ── Stage 2: re-rank globally by cheap quality signals ──
    union.sort(key=_quality_signal, reverse=True)

    # Rescues land at the front; remaining slots filled by ranked union.
    remaining = max(total_cap - len(rescued), 0)
    return rescued + union[:remaining]
