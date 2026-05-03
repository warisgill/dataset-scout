"""Source plugin contract.

Reference: `TECH_DESIGN.md` §4. Concrete sources (HF, Kaggle, PWC) land
in M1+. This module exists in M0 so plugin authors and tests have a
stable target to type against.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from dataset_scout.core import Candidate, DecompositionDirection, Intent


class Obligation(BaseModel):
    """A user-visible obligation a source imposes (terms, gating, etc.)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str
    summary: str
    url: str


class Budget(BaseModel):
    """Per-run external-call budget. Concrete shape stabilizes in M1."""

    model_config = ConfigDict(extra="forbid")

    max_http_calls: int = 200
    max_llm_calls: int = 50


@runtime_checkable
class Source(Protocol):
    """The plugin contract every source implements."""

    name: str

    def search(
        self,
        intent: Intent,
        directions: list[DecompositionDirection],
        *,
        budget: Budget,
    ) -> Iterator[Candidate]:
        """Yield candidates across the original Intent and all directions."""

    def fetch_metadata(self, candidate: Candidate) -> dict[str, Any]: ...

    def stream_sample(
        self,
        candidate: Candidate,
        n: int,
        seed: int,
    ) -> Iterator[dict[str, Any]]: ...

    def card_url(self, candidate: Candidate) -> str: ...

    def terms_check(self, intent: Intent) -> list[Obligation]: ...
