"""Output renderers.

Pure data-in, text/file-out. No business logic; renderers consume a
`ReconResult` (or future `CurateResult`) and write `results.json` /
`report.md`. The data layer never depends on rich/ANSI; rendering is
isolated here so the same `ReconResult` can also feed an HTTP response
or a notebook.
"""

from __future__ import annotations

from dataset_scout.render.html_report import render_recon_report_html, write_recon_report_html
from dataset_scout.render.inspect_panel import render_inspect
from dataset_scout.render.json_writer import write_results_json
from dataset_scout.render.judge_panel import render_eval_panel, render_judge_panel
from dataset_scout.render.markdown_report import render_recon_report, write_recon_report

__all__ = [
    "render_eval_panel",
    "render_inspect",
    "render_judge_panel",
    "render_recon_report",
    "render_recon_report_html",
    "write_recon_report",
    "write_recon_report_html",
    "write_results_json",
]
