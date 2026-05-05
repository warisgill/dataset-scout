"""Unit tests for `dataset_scout.render.html_report`."""

from __future__ import annotations

import typing
from html.parser import HTMLParser

import pytest

from dataset_scout.render._view import ReconReportContext
from dataset_scout.render.html_report import (
    render_recon_report_html,
    write_recon_report_html,
)
from dataset_scout.tour import build_tour_result

pytestmark = pytest.mark.unit


# ─── helpers ────────────────────────────────────────────────────────


class _SimpleHTMLValidator(HTMLParser):
    """Lightweight well-formedness check: every opening tag is closed.

    Skips void elements that don't need closing in HTML5 (meta, br, etc.)
    The HTML report is hand-rolled so the cost of being strict is real;
    we just want a minimal sanity gate that rendering produced
    structurally valid HTML.
    """

    _VOID: typing.ClassVar[set[str]] = {"meta", "br", "img", "input", "link", "hr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in self._VOID:
            return
        self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self._VOID:
            return
        if not self.stack:
            self.errors.append(f"end tag </{tag}> with empty stack")
            return
        if self.stack[-1] != tag:
            self.errors.append(f"mismatch: closing </{tag}>, expected </{self.stack[-1]}>")
        else:
            self.stack.pop()


def _validate_html(html: str) -> None:
    validator = _SimpleHTMLValidator()
    validator.feed(html)
    assert not validator.errors, f"HTML errors: {validator.errors}"
    assert not validator.stack, f"unclosed tags: {validator.stack}"


# ─── tests against the tour fixture ────────────────────────────────


def test_renders_well_formed_html_for_tour_result():
    result = build_tour_result()
    html = render_recon_report_html(result)
    assert html.startswith("<!doctype html>")
    assert "<title>" in html
    assert "</html>" in html
    _validate_html(html)


def test_includes_brief_section():
    result = build_tour_result()
    html = render_recon_report_html(result)
    assert "Brief" in html
    assert result.intent.raw_brief in html


def test_includes_run_summary():
    result = build_tour_result()
    html = render_recon_report_html(result)
    assert "Run summary" in html
    assert "Wall-clock" in html


def test_lists_all_candidates_with_card_links():
    result = build_tour_result()
    html = render_recon_report_html(result)
    assert "Candidates" in html
    for sc in result.candidates:
        # Card URL appears as an anchor href.
        if sc.candidate.metadata.card_url:
            assert sc.candidate.metadata.card_url in html


def test_renders_strategies_when_assessed():
    result = build_tour_result()
    html = render_recon_report_html(result)
    # The tour result has strategy assessments; expect kind labels.
    assert "direct use" in html or "signal proxy" in html
    assert "confidence" in html


def test_renders_coverage_gaps_when_present():
    result = build_tour_result()
    if result.coverage and result.coverage.semantic_gaps:
        html = render_recon_report_html(result)
        for gap in result.coverage.semantic_gaps:
            assert gap.aspect in html


def test_html_escaping_blocks_brief_xss():
    """A brief containing HTML metacharacters is escaped, not injected."""
    result = build_tour_result()
    # Mutate the intent's raw_brief to include payload; Pydantic v2
    # frozen models support model_copy.
    result = result.model_copy(
        update={"intent": result.intent.model_copy(update={"raw_brief": "<script>alert(1)</script>"})}
    )
    html = render_recon_report_html(result)
    assert "<script>" not in html  # raw payload absent
    assert "&lt;script&gt;" in html  # escaped form present


def test_write_recon_report_html_creates_file(tmp_path):
    result = build_tour_result()
    path = write_recon_report_html(result, tmp_path)
    assert path.exists()
    assert path.name == "report.html"
    content = path.read_text(encoding="utf-8")
    assert "<!doctype html>" in content


# ─── view-context tests ────────────────────────────────────────────


def test_view_context_from_tour_result():
    """The shared view-model derives consistent flags from a real result."""
    result = build_tour_result()
    ctx = ReconReportContext.from_result(result)
    assert ctx.n_candidates == len(result.candidates)
    # Tour has strategies + decomposition + gaps (mini-recon shape).
    assert ctx.has_strategies is True
    assert ctx.has_decomposition is True


def test_view_context_metadata_only_flag():
    from dataset_scout.core import Intent, ReconResult

    result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[],
        sources_searched=[],
        notices=["Azure OpenAI is not configured: ..."],
    )
    ctx = ReconReportContext.from_result(result)
    assert ctx.metadata_only is True
    assert ctx.has_strategies is False


def test_view_context_show_gaps_lead():
    """show_gaps_lead is True only when gaps are notable or coverage sparse."""
    from dataset_scout.core import (
        CoverageGap,
        CoverageReport,
        DecompositionDirection,
        Intent,
        ReconResult,
    )

    # Two gaps + a few candidates → notable_gaps → show_gaps_lead True.
    result = ReconResult(
        intent=Intent(raw_brief="x"),
        candidates=[],
        sources_searched=[],
        notices=[],
        coverage=CoverageReport(
            decomposition=[
                DecompositionDirection(
                    name="d1",
                    rationale="r",
                    keywords=[],
                    threat_families=[],
                    expected_finds="",
                ),
            ],
            semantic_gaps=[
                CoverageGap(aspect="a1", description="d1", suggestion="s1"),
                CoverageGap(aspect="a2", description="d2", suggestion="s2"),
            ],
        ),
    )
    ctx = ReconReportContext.from_result(result)
    assert ctx.show_gaps_lead is True


def test_includes_label_intent_fit_in_badge_when_present():
    """If a scorecard has label_intent_fit, the HTML surfaces a semantic-fit badge."""
    from dataset_scout.core import Evidence, SubScore

    result = build_tour_result()
    sc = result.candidates[0]
    sc.label_intent_fit = SubScore(
        value=0.72,
        status="ok",
        evidence=[Evidence(kind="embedding_fit", detail="cosine=0.72")],
        probe_version="1",
    )
    html = render_recon_report_html(result)
    assert "semantic fit" in html
    assert "0.72" in html
