"""Shared view-model for recon report renderers.

Single source of truth for the *decisions* both renderers make
(metadata-only vs full, lead with gaps vs candidates, etc.). Per
the rubber-duck pass, decoupling presentation framing from rendering
prevents Markdown and HTML reports from drifting as the product evolves.

This module is intentionally tiny: it only exposes a `ReconReportContext`
dataclass that the renderers consume. All HTML/Markdown formatting
lives in the respective `*_report.py` modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from dataset_scout.core import ReconResult


@dataclass(frozen=True)
class ReconReportContext:
    """Pre-computed flags + derived counts for a recon report render.

    Every "what mode are we in?" question answered exactly once.
    """

    metadata_only: bool
    llm_runtime_error: bool
    has_strategies: bool
    has_decomposition: bool
    has_gaps: bool
    notable_gaps: bool
    sparse_coverage: bool
    no_direct_fits: bool
    n_candidates: int
    n_strategy_assessed: int
    n_directions: int
    n_papers: int
    n_paper_dataset_citations: int

    @classmethod
    def from_result(cls, result: ReconResult) -> ReconReportContext:
        metadata_only = any("Azure OpenAI is not configured" in n for n in result.notices)
        llm_runtime_error = any("Azure OpenAI was configured but" in n for n in result.notices)
        has_strategies = any(sc.strategies for sc in result.candidates)
        has_decomposition = bool(result.coverage and result.coverage.decomposition)
        has_gaps = bool(result.coverage and result.coverage.semantic_gaps)
        notable_gaps = bool(result.coverage and len(result.coverage.semantic_gaps) >= 2)

        sparse_coverage = (
            has_decomposition
            and has_gaps
            and len(result.candidates) <= 5
            and not metadata_only
            and not llm_runtime_error
        )
        no_direct_fits = bool(
            has_strategies
            and not any(
                (sc.best_strategy is not None and sc.best_strategy.kind.value == "direct_use")
                for sc in result.candidates
            )
        )

        n_directions = len(result.coverage.decomposition) if result.coverage else 0
        n_strategy_assessed = sum(1 for sc in result.candidates if sc.strategies)
        n_paper_dataset_citations = sum(
            len(p.referenced_datasets) for p in result.papers
        )

        return cls(
            metadata_only=metadata_only,
            llm_runtime_error=llm_runtime_error,
            has_strategies=has_strategies,
            has_decomposition=has_decomposition,
            has_gaps=has_gaps,
            notable_gaps=notable_gaps,
            sparse_coverage=sparse_coverage,
            no_direct_fits=no_direct_fits,
            n_candidates=len(result.candidates),
            n_strategy_assessed=n_strategy_assessed,
            n_directions=n_directions,
            n_papers=len(result.papers),
            n_paper_dataset_citations=n_paper_dataset_citations,
        )

    @property
    def show_gaps_lead(self) -> bool:
        """True when the report should lead with the sourcing-roadmap section."""
        return self.has_gaps and (self.notable_gaps or self.sparse_coverage)

    @property
    def show_papers(self) -> bool:
        """True when the report should render the academic-papers section."""
        return self.n_papers > 0


__all__ = ["ReconReportContext"]
