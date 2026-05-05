"""HTML report renderer for `ReconResult`.

Self-contained HTML — embedded CSS, no JS, no external assets. Designed
to be opened directly in any browser, including by reviewers who don't
read Markdown well. Mirrors the section structure of `markdown_report.py`
via the shared `ReconReportContext` so the two renderers stay in sync.

Strategy kinds are color-coded; coverage gaps live in a callout box;
each candidate's badges and strategies are visually grouped.
"""

from __future__ import annotations

from html import escape
from io import StringIO
from pathlib import Path

from dataset_scout.core import (
    PaperReference,
    ReconResult,
    Scorecard,
    Strategy,
    StrategyKind,
    SubScore,
)
from dataset_scout.render._view import ReconReportContext

# ─── public entry points ───────────────────────────────────────────


def write_recon_report_html(result: ReconResult, out_dir: Path) -> Path:
    """Render and write `<out_dir>/report.html`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "report.html"
    target.write_text(render_recon_report_html(result), encoding="utf-8")
    return target


def render_recon_report_html(result: ReconResult) -> str:
    """Render a ReconResult to a self-contained HTML string."""
    ctx = ReconReportContext.from_result(result)
    buf = StringIO()
    _write_doc_head(buf, result.intent.raw_brief)
    buf.write('<body>\n<main class="container">\n')
    _write_header(buf, ctx)
    _write_brief_section(buf, result)
    if ctx.show_gaps_lead and result.coverage:
        _write_gaps_section(buf, result, lead=True)
    if ctx.has_decomposition and result.coverage:
        _write_decomposition_section(buf, result)
    _write_run_summary(buf, result, ctx)
    if result.candidates:
        _write_candidates_section(buf, result, ctx)
    if result.coverage and result.coverage.semantic_gaps and not ctx.show_gaps_lead:
        _write_gaps_section(buf, result, lead=False)
    if ctx.show_papers:
        _write_papers_section(buf, result, ctx)
    _write_footer(buf)
    buf.write("</main>\n</body>\n</html>\n")
    return buf.getvalue()


# ─── HTML head + CSS ───────────────────────────────────────────────


_CSS = """
:root {
  --fg: #1f2328;
  --fg-muted: #57606a;
  --bg: #ffffff;
  --bg-muted: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --good: #1f883d;
  --warn: #bf8700;
  --bad: #cf222e;
  --neutral: #57606a;
  --proxy: #8250df;
  --benign: #218bff;
  --code-bg: #f6f8fa;
}
@media (prefers-color-scheme: dark) {
  :root {
    --fg: #c9d1d9;
    --fg-muted: #8b949e;
    --bg: #0d1117;
    --bg-muted: #161b22;
    --border: #30363d;
    --accent: #58a6ff;
    --good: #56d364;
    --warn: #e3b341;
    --bad: #ff7b72;
    --neutral: #8b949e;
    --proxy: #d2a8ff;
    --benign: #79c0ff;
    --code-bg: #161b22;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--fg);
  line-height: 1.55;
  font-size: 16px;
}
.container { max-width: 920px; margin: 0 auto; padding: 32px 24px 80px; }
h1, h2, h3 { line-height: 1.25; margin-top: 1.5em; }
h1 { font-size: 1.75rem; border-bottom: 1px solid var(--border); padding-bottom: 0.25em; }
h2 { font-size: 1.35rem; border-bottom: 1px solid var(--border); padding-bottom: 0.2em; }
h3 { font-size: 1.1rem; }
p { margin: 0.6em 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
code, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
              background: var(--code-bg); padding: 0.1em 0.35em; border-radius: 3px;
              font-size: 0.92em; }
.callout {
  border-left: 4px solid var(--accent);
  background: var(--bg-muted);
  padding: 14px 18px;
  border-radius: 6px;
  margin: 1em 0;
}
.callout--warn { border-left-color: var(--warn); }
.callout--bad { border-left-color: var(--bad); }
.callout--good { border-left-color: var(--good); }
.callout--info { border-left-color: var(--accent); }
.callout--proxy { border-left-color: var(--proxy); }
.callout strong { display: block; margin-bottom: 0.25em; }
.muted { color: var(--fg-muted); }

.candidate {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 16px 20px;
  margin: 14px 0;
  background: var(--bg);
}
.candidate__head {
  display: flex; flex-wrap: wrap; gap: 8px 16px;
  align-items: baseline; margin-bottom: 4px;
}
.candidate__title { font-weight: 600; font-size: 1.05em; margin: 0; }
.candidate__meta { color: var(--fg-muted); font-size: 0.9em; }
.candidate__field { margin: 4px 0; font-size: 0.95em; }
.candidate__field b { color: var(--fg); }

.badges { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.badge {
  display: inline-block;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 0.82em;
  background: var(--bg-muted);
  border: 1px solid var(--border);
  color: var(--fg);
}
.badge--good { border-color: var(--good); color: var(--good); }
.badge--warn { border-color: var(--warn); color: var(--warn); }
.badge--bad { border-color: var(--bad); color: var(--bad); }

.strategies { margin: 10px 0; }
.strategy {
  display: block; padding: 8px 12px; margin: 6px 0;
  border-left: 4px solid var(--neutral);
  background: var(--bg-muted);
  border-radius: 0 6px 6px 0;
}
.strategy--direct_use { border-left-color: var(--good); }
.strategy--subset_extraction { border-left-color: var(--accent); }
.strategy--label_remapping { border-left-color: var(--accent); }
.strategy--cross_class_repurposing { border-left-color: var(--proxy); }
.strategy--signal_proxy { border-left-color: var(--proxy); }
.strategy--benign_baseline { border-left-color: var(--benign); }
.strategy--composition_only { border-left-color: var(--neutral); }
.strategy--not_useful { border-left-color: var(--bad); opacity: 0.85; }

.strategy__head { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
.strategy__kind { font-weight: 600; }
.strategy__conf { color: var(--fg-muted); font-size: 0.9em; }
.strategy__rationale { margin: 4px 0; }
.strategy__caveats, .strategy__transform { font-size: 0.9em; color: var(--fg-muted); }
.strategy__caveats li { margin: 2px 0; }
.strategy__transform { margin-top: 4px; }

.probe-grid {
  display: grid;
  grid-template-columns: 160px 1fr;
  gap: 4px 12px;
  font-size: 0.9em;
  margin: 8px 0;
}
.probe-grid dt { color: var(--fg-muted); }
.probe-grid dd { margin: 0; }

.run-summary { background: var(--bg-muted); padding: 12px 16px; border-radius: 6px; }
.run-summary ul { margin: 0; padding-left: 20px; }
.notices { margin-top: 10px; }
.notice { font-size: 0.92em; color: var(--fg-muted); margin: 4px 0; }

footer.disclaimer {
  margin-top: 60px; padding-top: 16px; border-top: 1px solid var(--border);
  font-size: 0.85em; color: var(--fg-muted); text-align: center;
}

.paper {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 14px 18px;
  margin: 12px 0;
  background: var(--bg);
}
.paper__title { margin: 0 0 6px; font-size: 1.05em; }
.paper__meta { color: var(--fg-muted); font-size: 0.9em; margin-bottom: 4px; }
.paper__authors { color: var(--fg-muted); font-size: 0.9em; margin-bottom: 4px; }
.paper__abstract { font-size: 0.95em; margin: 6px 0; }
.paper__surfaced { font-size: 0.85em; color: var(--fg-muted); margin: 4px 0; }
.paper__citations { background: var(--bg-muted); padding: 8px 12px; border-radius: 6px;
                    font-size: 0.92em; margin-top: 8px; }
.paper__citations ul { margin: 4px 0 0 0; padding-left: 20px; }
"""


def _write_doc_head(buf: StringIO, title: str) -> None:
    safe_title = escape(title[:120] or "dataset-scout recon report")
    buf.write("<!doctype html>\n<html lang=\"en\">\n<head>\n")
    buf.write("<meta charset=\"utf-8\">\n")
    buf.write("<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n")
    buf.write(f"<title>dataset-scout: {safe_title}</title>\n")
    buf.write("<style>\n")
    buf.write(_CSS)
    buf.write("\n</style>\n</head>\n")


# ─── sections ──────────────────────────────────────────────────────


def _write_header(buf: StringIO, ctx: ReconReportContext) -> None:
    buf.write("<h1>dataset-scout recon report</h1>\n")
    if ctx.metadata_only:
        buf.write(
            '<div class="callout callout--warn"><strong>⚠️ Metadata-only mode.</strong>'
            "Azure OpenAI is not configured, so decomposition, strategy assessment, "
            "and coverage gaps were skipped. To enable them, copy <code>.env.example</code> "
            "to <code>.env</code>, set <code>AZURE_OPENAI_ENDPOINT</code> and "
            "<code>AZURE_OPENAI_DEPLOYMENT</code>, and run <code>az login</code>."
            "</div>\n"
        )
    elif ctx.llm_runtime_error:
        buf.write(
            '<div class="callout callout--warn"><strong>⚠️ LLM call failed — running in metadata-only mode.</strong>'
            "Azure OpenAI was configured but a call failed at runtime "
            "(deployment name, token, network, or quota — see notices below). "
            "Decomposition, strategy assessment, and coverage gaps were skipped."
            "</div>\n"
        )
    elif ctx.sparse_coverage:
        buf.write(
            f'<div class="callout callout--info"><strong>🗺️ Sparse-coverage territory.</strong>'
            f"HuggingFace returned {ctx.n_candidates} candidate(s) for this brief. "
            "The decomposition + coverage gaps below are the actual sourcing roadmap — "
            "most of the data you need for this detection program lives outside HF."
            "</div>\n"
        )
    elif ctx.no_direct_fits:
        buf.write(
            '<div class="callout callout--proxy"><strong>🧩 No direct-fit corpora — every candidate is a reframing.</strong>'
            "Strategy assessment ran successfully; review the reframings "
            "(signal_proxy / cross_class_repurposing / benign_baseline) before committing. "
            "Direct ground-truth corpora for this brief don't appear to exist on HuggingFace yet."
            "</div>\n"
        )
    elif ctx.has_strategies:
        buf.write(
            f'<div class="callout callout--good"><strong>Strategy-assessed.</strong>'
            f"Search expanded across the original brief and {ctx.n_directions} related "
            "direction(s); shortlisted candidates were assessed for direct fits and reframings. "
            "Listed in best-strategy order; review the strategy + caveats before committing."
            "</div>\n"
        )
    elif ctx.has_decomposition:
        buf.write(
            f'<div class="callout callout--info"><strong>Discovery + decomposition.</strong>'
            f"Search expanded across the original brief and {ctx.n_directions} related "
            "direction(s) proposed by the LLM."
            "</div>\n"
        )
    else:
        buf.write(
            '<div class="callout callout--info"><strong>Pre-fit metadata screening.</strong>'
            "Candidates are returned in source/search relevance order. "
            "Probe outputs are annotations, not a ranking score."
            "</div>\n"
        )


def _write_brief_section(buf: StringIO, result: ReconResult) -> None:
    intent = result.intent
    buf.write("<h2>Brief</h2>\n")
    buf.write(f'<p><b>Raw brief:</b> {escape(intent.raw_brief)}</p>\n')
    if intent.detection_target:
        buf.write(f'<p><b>Detection target:</b> {escape(intent.detection_target)}</p>\n')
    if intent.threat_families:
        buf.write(
            f'<p><b>Threat families:</b> {escape(", ".join(intent.threat_families))}</p>\n'
        )
    buf.write(
        f'<p><b>Languages requested:</b> {escape(", ".join(intent.languages))}</p>\n'
    )


def _write_gaps_section(buf: StringIO, result: ReconResult, *, lead: bool) -> None:
    if result.coverage is None:
        return
    title = "Sourcing roadmap" if lead else "Coverage gaps"
    buf.write(f"<h2>{title}</h2>\n")
    if lead:
        buf.write(
            "<p>Where the data is — and isn't. The LLM identified specific aspects of "
            "your brief that the candidate set doesn't cover, with concrete next steps:</p>\n"
        )
    else:
        buf.write("<p>Aspects worth augmenting:</p>\n")
    for gap in result.coverage.semantic_gaps:
        buf.write(f"<h3>{escape(gap.aspect)}</h3>\n")
        buf.write(f"<p>{escape(gap.description)}</p>\n")
        buf.write(f'<p><b>→ Next step:</b> {escape(gap.suggestion)}</p>\n')


def _write_decomposition_section(buf: StringIO, result: ReconResult) -> None:
    if result.coverage is None:
        return
    buf.write("<h2>Decomposition</h2>\n")
    buf.write(
        "<p>The LLM proposed these search directions in addition to the original brief:</p>\n<ul>\n"
    )
    for d in result.coverage.decomposition:
        buf.write(f"<li><b>{escape(d.name)}</b> — {escape(d.rationale)}")
        if d.keywords:
            buf.write(
                f"<br><span class=\"muted\">keywords: <code>{escape(', '.join(d.keywords))}</code></span>"
            )
        if d.expected_finds:
            buf.write(
                f'<br><span class="muted">expected: {escape(d.expected_finds)}</span>'
            )
        buf.write("</li>\n")
    buf.write("</ul>\n")


def _write_run_summary(buf: StringIO, result: ReconResult, ctx: ReconReportContext) -> None:
    buf.write("<h2>Run summary</h2>\n")
    buf.write('<div class="run-summary"><ul>\n')
    buf.write(
        f'<li>Sources searched: {escape(", ".join(result.sources_searched) or "(none)")}</li>\n'
    )
    buf.write(f"<li>Candidates returned: <b>{ctx.n_candidates}</b></li>\n")
    if ctx.has_strategies:
        buf.write(f"<li>Strategy-assessed: <b>{ctx.n_strategy_assessed}</b></li>\n")
    buf.write(f"<li>Wall-clock: {result.elapsed_seconds:.2f}s</li>\n")
    buf.write(f"<li>dataset-scout version: <code>{escape(result.scout_version)}</code></li>\n")
    buf.write("</ul>\n")
    if result.notices:
        buf.write('<div class="notices"><b>Notices:</b><ul>\n')
        for n in result.notices:
            buf.write(f'<li class="notice">{escape(n)}</li>\n')
        buf.write("</ul></div>\n")
    buf.write("</div>\n")


def _write_candidates_section(
    buf: StringIO, result: ReconResult, ctx: ReconReportContext
) -> None:
    buf.write("<h2>Candidates</h2>\n")
    if ctx.has_strategies:
        buf.write(
            "<p>Listed in <b>best-strategy order</b>. Each candidate's primary "
            "strategy and confidence are shown; review caveats before committing.</p>\n"
        )
    elif ctx.has_decomposition:
        buf.write(
            "<p>Listed in search-relevance order, deduped across the original brief "
            "and decomposition directions. The <code>surfaced by</code> annotation "
            "shows which direction(s) found each candidate.</p>\n"
        )
    else:
        buf.write(
            "<p>Listed in <b>search-relevance order from the source</b>. "
            "This is not a fitness ranking.</p>\n"
        )
    for i, sc in enumerate(result.candidates, start=1):
        _write_candidate(buf, i, sc)


def _write_candidate(buf: StringIO, index: int, sc: Scorecard) -> None:
    cand = sc.candidate
    md = cand.metadata
    buf.write('<div class="candidate">\n')
    buf.write('<div class="candidate__head">\n')
    buf.write(
        f'<h3 class="candidate__title">{index}. <code>{escape(cand.source)}:{escape(cand.id)}</code></h3>\n'
    )
    buf.write("</div>\n")
    if md.card_url:
        buf.write(
            f'<div class="candidate__field">🔗 <b>Card:</b> '
            f'<a href="{escape(md.card_url)}">{escape(md.card_url)}</a></div>\n'
        )
    if cand.revision:
        buf.write(
            f'<div class="candidate__field">📌 <b>Revision:</b> '
            f"<code>{escape(cand.revision[:12])}</code></div>\n"
        )
    if md.description:
        desc = md.description.strip().splitlines()[0][:240]
        buf.write(f'<div class="candidate__field">📝 <b>Description:</b> {escape(desc)}</div>\n')
    if cand.surfaced_by:
        buf.write(
            f'<div class="candidate__field">🧭 <b>Surfaced by:</b> '
            f"{escape(', '.join(cand.surfaced_by))}</div>\n"
        )
    if cand.requires_auth:
        buf.write(
            '<div class="candidate__field">🔒 <b>Access:</b> gated / requires authentication</div>\n'
        )

    badges = list(_render_badges(sc))
    if badges:
        buf.write('<div class="badges">')
        for cls, text in badges:
            buf.write(f'<span class="badge badge--{cls}">{escape(text)}</span>')
        buf.write("</div>\n")

    if sc.strategies:
        _write_strategies(buf, sc.strategies)

    _write_probe_signals(buf, sc)

    buf.write("</div>\n")


def _write_strategies(buf: StringIO, strategies: list[Strategy]) -> None:
    buf.write('<div class="strategies">\n')
    for s in strategies:
        kind_class = s.kind.value
        buf.write(f'<div class="strategy strategy--{kind_class}">\n')
        buf.write(
            f'<div class="strategy__head">'
            f'<span class="strategy__kind">{escape(_strategy_label(s.kind))}</span>'
            f'<span class="strategy__conf">confidence {s.confidence:.2f}</span>'
            f"</div>\n"
        )
        buf.write(f'<div class="strategy__rationale">{escape(s.rationale)}</div>\n')
        if s.caveats:
            buf.write('<ul class="strategy__caveats">\n')
            for cav in s.caveats:
                buf.write(f"<li>⚠ {escape(cav)}</li>\n")
            buf.write("</ul>\n")
        bits = _transform_bits(s)
        if bits:
            buf.write(f'<div class="strategy__transform">transform: {bits}</div>\n')
        buf.write("</div>\n")
    buf.write("</div>\n")


def _strategy_label(kind: StrategyKind) -> str:
    return {
        StrategyKind.DIRECT_USE: "✅ direct use",
        StrategyKind.SUBSET_EXTRACTION: "🔎 subset extraction",
        StrategyKind.LABEL_REMAPPING: "🔁 label remapping",
        StrategyKind.CROSS_CLASS_REPURPOSING: "🔀 cross-class repurposing",
        StrategyKind.SIGNAL_PROXY: "📡 signal proxy",
        StrategyKind.BENIGN_BASELINE: "🧊 benign baseline",
        StrategyKind.COMPOSITION_ONLY: "🧩 composition-only",
        StrategyKind.NOT_USEFUL: "❌ not useful",
    }.get(kind, kind.value)


def _transform_bits(s: Strategy) -> str:
    t = s.transform
    bits: list[str] = []
    if t.text_column:
        bits.append(f"text=<code>{escape(t.text_column)}</code>")
    if t.label_column:
        bits.append(f"label=<code>{escape(t.label_column)}</code>")
    if t.label_value_map:
        kv = ", ".join(
            f"<code>{escape(k)}</code>→{escape(v)}" for k, v in t.label_value_map.items()
        )
        bits.append("values={" + kv + "}")
    if t.filter:
        bits.append(f"filter=<code>{escape(t.filter)}</code>")
    if t.take != "all":
        bits.append(f"take={escape(str(t.take))}")
    return " · ".join(bits)


# ─── badges (parallel to markdown_report._render_badges) ───────────


def _evidence_detail(sub: SubScore, kind: str) -> str | None:
    for e in sub.evidence:
        if e.kind == kind:
            return e.detail
    return None


def _render_badges(sc: Scorecard) -> list[tuple[str, str]]:
    """(css-class, text) tuples for the candidate's badges."""
    badges: list[tuple[str, str]] = []
    license_sub = sc.cheap_probes.get("license")
    if license_sub is not None:
        match = _evidence_detail(license_sub, "policy_match")
        spdx = _evidence_detail(license_sub, "license_spdx")
        if spdx and match == "allow":
            badges.append(("good", f"license: {spdx} ✅"))
        elif spdx and match == "warn_only":
            badges.append(("warn", f"license: {spdx} ⚠️"))
        elif spdx:
            badges.append(("bad", f"license: {spdx} ❗"))
        elif license_sub.status == "low_confidence":
            raw = _evidence_detail(license_sub, "license_raw") or "?"
            badges.append(("warn", f"license: {raw} (?)"))
        else:
            badges.append(("warn", "license: missing"))

    fresh_sub = sc.cheap_probes.get("freshness")
    if fresh_sub is not None and fresh_sub.status == "ok":
        bucket = _evidence_detail(fresh_sub, "bucket") or "?"
        badges.append(("", f"freshness: {bucket}"))

    size_sub = sc.cheap_probes.get("size")
    if size_sub is not None and size_sub.status == "ok":
        rows = _evidence_detail(size_sub, "rows")
        bytes_ = _evidence_detail(size_sub, "bytes")
        downloads = _evidence_detail(size_sub, "downloads")
        if rows:
            badges.append(("", f"rows: {rows}"))
        elif bytes_:
            badges.append(("", f"bytes: {bytes_}"))
        if downloads:
            badges.append(("", f"downloads: {downloads}"))

    lang_sub = sc.cheap_probes.get("languages")
    if lang_sub is not None and lang_sub.status == "ok":
        declared = _evidence_detail(lang_sub, "declared") or "?"
        badges.append(("", f"languages: {declared}"))

    if sc.label_intent_fit is not None and sc.label_intent_fit.value is not None:
        v = sc.label_intent_fit.value
        cls = "good" if v >= 0.6 else ("warn" if v >= 0.4 else "")
        badges.append((cls, f"semantic fit: {v:.2f}"))

    return badges


def _write_probe_signals(buf: StringIO, sc: Scorecard) -> None:
    """Compact per-probe details below badges. Kept brief; the report
    leads with strategies — probe signals are supporting evidence."""
    if not sc.cheap_probes and sc.label_intent_fit is None:
        return
    buf.write('<dl class="probe-grid">\n')
    for name, sub in sorted(sc.cheap_probes.items()):
        if sub.status not in ("ok", "low_confidence"):
            continue
        details = "; ".join(
            f"{e.kind}={escape(e.detail)}" for e in sub.evidence[:3]
        )
        if not details:
            continue
        buf.write(f"<dt>{escape(name)}</dt><dd>{details}</dd>\n")
    if sc.label_intent_fit is not None and sc.label_intent_fit.evidence:
        e = sc.label_intent_fit.evidence[0]
        buf.write(
            f"<dt>label_intent_fit</dt><dd>{escape(e.detail)}</dd>\n"
        )
    buf.write("</dl>\n")


def _write_footer(buf: StringIO) -> None:
    buf.write(
        '<footer class="disclaimer">'
        "License signals are an SPDX-best-effort guess. "
        "Always read the upstream card before redistributing. This is not legal advice."
        "</footer>\n"
    )


def _write_papers_section(
    buf: StringIO, result: ReconResult, ctx: ReconReportContext
) -> None:
    """Render the academic-paper discovery section.

    Mirrors the Markdown renderer's structure but with semantic HTML
    (article + dl) and color-coded venue / dataset-citation badges so
    a reviewer can scan the section without reading every paragraph.
    """
    buf.write("<h2>Related papers</h2>\n")
    citations = ctx.n_paper_dataset_citations
    if citations > 0:
        buf.write(
            f"<p>{ctx.n_papers} paper(s) from NeurIPS / ICML / ICLR / SaTML "
            f"with <b>{citations} dataset citation(s)</b> extracted from abstracts.</p>\n"
        )
    else:
        buf.write(
            f"<p>{ctx.n_papers} paper(s) from NeurIPS / ICML / ICLR / SaTML. "
            "No dataset URLs found in abstracts — read the paper to find "
            "the dataset directly.</p>\n"
        )
    for p in result.papers:
        _write_paper(buf, p)


def _write_paper(buf: StringIO, p: PaperReference) -> None:
    venue = p.venue or "?"
    cite_count = (
        f' · {p.citation_count} citation(s)' if p.citation_count else ""
    )
    buf.write('<article class="paper">\n')
    buf.write(f'<h3 class="paper__title"><a href="{escape(p.url)}">{escape(p.title)}</a></h3>\n')
    authors = ", ".join(p.authors[:5]) + ("…" if len(p.authors) > 5 else "")
    buf.write(
        f'<div class="paper__meta"><span class="badge">{escape(venue)} {p.year}</span>'
        f"{escape(cite_count)}</div>\n"
    )
    if authors:
        buf.write(f'<div class="paper__authors">{escape(authors)}</div>\n')
    if p.abstract:
        snippet = p.abstract.strip().replace("\n", " ")[:280]
        if len(p.abstract) > 280:
            snippet += "…"
        buf.write(f'<p class="paper__abstract">{escape(snippet)}</p>\n')
    if p.surfaced_by:
        buf.write(
            f'<div class="paper__surfaced">🧭 surfaced by: '
            f"{escape(', '.join(p.surfaced_by))}</div>\n"
        )
    if p.referenced_datasets:
        buf.write('<div class="paper__citations"><b>Datasets cited:</b><ul>\n')
        for d in p.referenced_datasets:
            buf.write(
                f'<li><code>{escape(d.source)}:{escape(d.identifier)}</code> — '
                f'<a href="{escape(d.url)}">{escape(d.url)}</a></li>\n'
            )
        buf.write("</ul></div>\n")
    buf.write("</article>\n")


__all__ = ["render_recon_report_html", "write_recon_report_html"]
