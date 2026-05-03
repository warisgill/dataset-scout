"""Probe protocol — the contract every probe implements.

A probe takes a Candidate + Intent and produces a SubScore (a value with
status, evidence, optional Wilson CI). Probes are stateless and trivially
parallelizable.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from dataset_scout.core import Candidate, Intent, SubScore


@runtime_checkable
class Probe(Protocol):
    name: str
    version: str

    def applies(self, candidate: Candidate, intent: Intent) -> bool: ...

    def run(self, candidate: Candidate, intent: Intent) -> SubScore: ...


class ProbeRegistry:
    """A list-like container of probes with predictable iteration order."""

    def __init__(self, probes: list[Probe]) -> None:
        self._probes = list(probes)

    def __iter__(self) -> Iterator[Probe]:
        return iter(self._probes)

    def __len__(self) -> int:
        return len(self._probes)

    def names(self) -> list[str]:
        return [p.name for p in self._probes]
