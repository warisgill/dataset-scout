"""Reusable in-memory `FakeSource` for tests.

Implements the `Source` protocol (`dataset_scout.sources.base.Source`)
without touching the network. Tests that need a Source double should
construct one of these with canned `Candidate` objects and (optionally)
canned sample rows keyed by candidate id.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from dataset_scout.core import Candidate, DecompositionDirection, Intent
from dataset_scout.sources.base import Budget, Obligation


class FakeSource:
    """A canned Source implementation for use in tests.

    Does not inherit from the `Source` Protocol type — Protocols are
    structural in Python. `isinstance(fake, Source)` works because
    `Source` is `@runtime_checkable`.
    """

    name: str = "fake"

    def __init__(
        self,
        candidates: list[Candidate],
        samples: dict[str, list[dict[str, Any]]] | None = None,
        *,
        obligations: list[Obligation] | None = None,
    ) -> None:
        self._candidates: list[Candidate] = list(candidates)
        self._samples: dict[str, list[dict[str, Any]]] = dict(samples or {})
        self._obligations: list[Obligation] = list(obligations or [])

        self.search_calls: int = 0
        self.metadata_calls: int = 0
        self.sample_calls: int = 0
        self.stream_rows_calls: int = 0
        self.card_url_calls: int = 0
        self.terms_check_calls: int = 0

    def search(
        self,
        intent: Intent,
        directions: list[DecompositionDirection],
        *,
        budget: Budget,
    ) -> Iterator[Candidate]:
        self.search_calls += 1
        yield from self._candidates

    def fetch_metadata(self, candidate: Candidate) -> dict[str, Any]:
        self.metadata_calls += 1
        return candidate.metadata.model_dump()

    def stream_sample(
        self,
        candidate: Candidate,
        n: int,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        self.sample_calls += 1
        rows = self._samples.get(candidate.id, [])
        for row in rows[:n]:
            yield dict(row)

    def stream_rows(
        self,
        candidate: Candidate,
        *,
        config: str | None = None,
        split: str = "train",
        take: int | None = None,
        seed: int = 42,
    ) -> Iterator[dict[str, Any]]:
        self.stream_rows_calls += 1
        rows = self._samples.get(candidate.id, [])
        if take is not None:
            rows = rows[:take]
        for row in rows:
            yield dict(row)

    def card_url(self, candidate: Candidate) -> str:
        self.card_url_calls += 1
        return f"https://fake.example/{candidate.id}"

    def terms_check(self, intent: Intent) -> list[Obligation]:
        self.terms_check_calls += 1
        return list(self._obligations)
