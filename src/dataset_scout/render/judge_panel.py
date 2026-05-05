"""Rich console panels for the M10 judge / eval CLI verbs.

Pure data-in, ANSI-out. The library never depends on rich; rendering
lives here so the same :class:`JudgeResult` / :class:`EvalResult`
objects can drive an HTTP response or a notebook without taking the
CLI's display dep.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

if TYPE_CHECKING:
    from dataset_scout.eval_ import EvalResult
    from dataset_scout.judge import JudgeResult


def render_judge_panel(result: JudgeResult, *, console: Console | None = None) -> None:
    """Render a :class:`JudgeResult` summary to ``console`` (default stderr).

    Mirrors the lockfile ``judge`` block layout so the CLI output and
    the audit trail match. Soft-failure categories are surfaced under a
    "skipped" line; cache hit ratio is included so users notice when
    re-runs are mostly free.
    """
    cons = console if console is not None else Console(stderr=True)
    s = result.stats
    if result.dry_run:
        cons.print(
            Panel.fit(
                f"[bold]dry-run[/bold] — would issue [cyan]{result.estimated_calls}[/cyan] "
                f"judge call(s) across [cyan]{s.n_input}[/cyan] row(s)",
                title="datascout judge (dry-run)",
            )
        )
        return

    promoted_pct = (
        100.0 * (s.n_promoted_positive + s.n_promoted_negative) / s.n_judged if s.n_judged else 0.0
    )
    cache_pct = 100.0 * s.n_cache_hits / max(s.n_judged * result.n_judges, 1)
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(style="dim")
    table.add_column()
    table.add_row("axis", result.axis)
    table.add_row("model", result.model)
    table.add_row("template_version", result.template_version)
    table.add_row("threshold", f"{result.threshold:.2f}")
    table.add_row("judges / agreement", f"{result.n_judges} / {result.agreement}")
    table.add_row(
        "rows",
        f"[green]{s.n_promoted_positive}[/green] promoted positive · "
        f"[green]{s.n_promoted_negative}[/green] promoted negative · "
        f"[yellow]{s.n_left_unknown}[/yellow] left unknown · "
        f"[red]{s.n_skipped}[/red] skipped",
    )
    table.add_row(
        "totals",
        f"{s.n_judged}/{s.n_input} judged "
        f"({promoted_pct:.1f}% promoted, {cache_pct:.0f}% cache hits)",
    )
    if s.n_resumed:
        table.add_row("resumed", f"{s.n_resumed} previously-completed row(s)")
    if s.n_skipped:
        breakdown: list[str] = []
        if s.n_api_errors:
            breakdown.append(f"api={s.n_api_errors}")
        if s.n_parse_errors:
            breakdown.append(f"parse={s.n_parse_errors}")
        if s.n_content_filter_blocked:
            breakdown.append(f"content_filter={s.n_content_filter_blocked}")
        table.add_row("skips", " · ".join(breakdown) or "n/a")
    if result.calibration is not None:
        c = result.calibration
        table.add_row(
            "calibration",
            (
                f"P={c.get('precision', 0.0):.3f} · "
                f"R={c.get('recall', 0.0):.3f} · "
                f"F1={c.get('f1', 0.0):.3f} · "
                f"n={c.get('n_sampled', 0)}"
            ),
        )
    table.add_row("out_dir", str(result.out_dir))
    table.add_row("elapsed", f"{result.elapsed_seconds:.2f}s")
    cons.print(Panel(table, title="[bold]datascout judge[/bold]", title_align="left"))


def render_eval_panel(result: EvalResult, *, console: Console | None = None) -> None:
    """Render a :class:`EvalResult` summary to ``console`` (default stderr)."""
    cons = console if console is not None else Console(stderr=True)
    if not result.per_axis:
        cons.print("[yellow]eval produced no metrics — corpus had no joinable rows.[/yellow]")
        return
    table = Table(title="datascout eval", show_lines=False)
    table.add_column("axis", style="cyan")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("cov", justify="right")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("TN", justify="right")
    table.add_column("n_gold", justify="right")
    table.add_column("n_judged", justify="right")
    for axis, m in result.per_axis.items():
        cm = m.confusion
        table.add_row(
            axis,
            f"{m.precision:.3f}",
            f"{m.recall:.3f}",
            f"{m.f1:.3f}",
            f"{m.coverage:.3f}",
            str(cm.true_positive),
            str(cm.false_positive),
            str(cm.false_negative),
            str(cm.true_negative),
            str(m.n_gold),
            str(m.n_judged_seen),
        )
    cons.print(table)
    for n in result.notices:
        cons.print(f"[dim]note:[/dim] {n}")


__all__ = ["render_eval_panel", "render_judge_panel"]
