"""dataset-scout CLI — thin shell over the library.

Two equivalent entry points: `dataset-scout` and `datascout`. Same
command tree (see [docs/cli.md]). M0 ships the verb skeletons; later
milestones light up real work behind each.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from dataset_scout import __version__

app = typer.Typer(
    name="dataset-scout",
    help="Reconnaissance, reframing, and curation of public datasets for AI detection engineers.",
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

err = Console(stderr=True)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"dataset-scout {__version__}")
        raise typer.Exit()


def _load_dotenv() -> Path | None:
    """Load `.env` from the CWD if present. Returns the path loaded, or None.

    Uses `override=False` so explicitly-set environment variables always
    win over `.env` — explicit shell config trumps dotfiles.
    """
    from dotenv import load_dotenv

    candidate = Path.cwd() / ".env"
    if candidate.is_file():
        load_dotenv(candidate, override=False)
        return candidate
    return None


@app.callback()
def _root(
    version: Annotated[
        bool | None,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = None,
) -> None:
    """dataset-scout — find and curate public datasets for detection work."""
    import contextlib

    # Make stdout/stderr UTF-8 so emoji/unicode in reports render on
    # Windows consoles (cp1252 by default).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8")
    _load_dotenv()


# ─── recon ──────────────────────────────────────────────────────────


@app.command(help="Find candidates and assess how each could be used.")
def recon(
    brief: Annotated[
        str, typer.Argument(help="Natural-language brief describing the detection target.")
    ],
    detection_target: Annotated[
        str | None,
        typer.Option("--detection-target", help="Override the parsed detection target."),
    ] = None,
    deployment_context: Annotated[str | None, typer.Option("--deployment-context")] = None,
    language: Annotated[
        list[str] | None,
        typer.Option("--language", help="Languages to require (repeatable)."),
    ] = None,
    license: Annotated[
        list[str] | None,
        typer.Option("--license", help="License allowlist (repeatable)."),
    ] = None,
    min_strategy_confidence: Annotated[
        float, typer.Option("--min-strategy-confidence", min=0.0, max=1.0)
    ] = 0.5,
    out: Annotated[Path, typer.Option("--out", help="Output directory.")] = Path("datascout-out"),
    # Hidden expert escape hatches — not in --help. Useful for debugging
    # and forward compatibility; production users should let the brief +
    # decomposer carry the signal.
    threat_families: Annotated[
        list[str] | None,
        typer.Option("--threat-families", hidden=True),
    ] = None,
    no_explore: Annotated[
        bool,
        typer.Option("--no-explore", hidden=True, help="Skip LLM decomposition (debug)."),
    ] = False,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.pipeline import run_recon
    from dataset_scout.render import write_recon_report, write_results_json

    overrides: dict[str, object] = {"min_strategy_confidence": min_strategy_confidence}
    if detection_target:
        overrides["detection_target"] = detection_target
    if deployment_context:
        overrides["deployment_context"] = deployment_context
    if language:
        overrides["language"] = list(language)
    if license:
        overrides["license"] = list(license)
    if threat_families:
        overrides["threat_families"] = list(threat_families)

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())

    try:
        result = run_recon(
            brief,
            ctx=ctx,
            parser_overrides=overrides,
            explore=not no_explore,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    json_path = write_results_json(result, out)
    md_path = write_recon_report(result, out)

    err.print(
        f"[green]✔[/green] {len(result.candidates)} candidate(s) "
        f"from {', '.join(result.sources_searched) or '(no source)'} "
        f"in {result.elapsed_seconds:.2f}s"
    )
    err.print(f"  - results: {json_path}")
    err.print(f"  - report:  {md_path}")
    if result.notices:
        for n in result.notices:
            err.print(f"  [yellow]![/yellow] {n}")


# ─── inspect ────────────────────────────────────────────────────────


@app.command(help="Deep-dive on one candidate.")
def inspect(
    target: Annotated[
        str,
        typer.Argument(
            help="<source>:<id>[@revision], e.g. huggingface:deepset/prompt-injections."
        ),
    ],
    intent_from: Annotated[
        Path | None,
        typer.Option(
            "--intent-from",
            help="Re-use the most recent recon's Intent (point at results.json).",
        ),
    ] = None,
    brief: Annotated[
        str | None,
        typer.Option(
            "--brief",
            help="One-off brief for strategy assessment when no recon results.json exists.",
        ),
    ] = None,
    sample_size: Annotated[
        int,
        typer.Option(
            "--sample-size",
            help="Rows to stream for the schema/label-distribution/sample sections.",
            min=1,
            max=1000,
        ),
    ] = 50,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.inspect_ import make_intent, run_inspect
    from dataset_scout.render import render_inspect

    if intent_from is not None and brief is not None:
        err.print("[red]error:[/red] pass either --intent-from or --brief, not both.")
        raise typer.Exit(code=1)

    try:
        intent = make_intent(brief=brief, intent_from=intent_from)
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())

    try:
        result = run_inspect(
            target,
            ctx=ctx,
            intent=intent,
            sample_size=sample_size,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Inspect prints to stdout (so it works in pipes).
    typer.echo(render_inspect(result))
    if result.notices:
        for n in result.notices:
            err.print(f"[dim]note:[/dim] {n}")


# ─── curate ─────────────────────────────────────────────────────────


@app.command(help="Build a schema-normalized corpus from a recipe.")
def curate(
    from_: Annotated[Path, typer.Option("--from", help="Path to recipe.yaml.")],
    out: Annotated[Path, typer.Option("--out", help="Output corpus directory.")] = Path("mycorpus"),
    min_strategy_confidence: Annotated[
        float | None, typer.Option("--min-strategy-confidence", min=0.0, max=1.0)
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the recipe's seed.")] = None,
) -> None:
    from dataset_scout.context import ScoutContext
    from dataset_scout.curate import load_recipe, run_curate
    from dataset_scout.errors import DatasetScoutError

    try:
        recipe = load_recipe(from_)
    except Exception as e:
        err.print(f"[red]error:[/red] failed to load recipe: {e}")
        raise typer.Exit(code=1) from e

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())

    try:
        result = run_curate(
            recipe,
            out,
            ctx=ctx,
            seed_override=seed,
            min_strategy_confidence_override=min_strategy_confidence,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    err.print(
        f"[green]✔[/green] {result.total_rows} row(s) written to {result.out_dir} "
        f"({result.components_kept} component(s) kept, "
        f"{result.components_skipped} skipped) in {result.elapsed_seconds:.2f}s"
    )
    splits_str = " · ".join(f"{n}={c}" for n, c in result.rows_per_split.items())
    err.print(f"  - splits: {splits_str}")
    err.print(f"  - fingerprint: {result.fingerprint[:16]}...")
    err.print(
        "  [yellow]![/yellow] preview build — hash-mod split, no dedup. "
        "Audit-ready splitting + MinHash land in M4b."
    )


# ─── cache ──────────────────────────────────────────────────────────

cache_app = typer.Typer(help="Inspect and manage the dataset-scout cache.", no_args_is_help=True)
app.add_typer(cache_app, name="cache")


@cache_app.command("info")
def cache_info() -> None:
    err.print("[yellow]cache info is not implemented yet (lands with M1's cache).[/yellow]")
    raise typer.Exit(code=2)


@cache_app.command("prune")
def cache_prune() -> None:
    err.print("[yellow]cache prune is not implemented yet.[/yellow]")
    raise typer.Exit(code=2)


@cache_app.command("clear")
def cache_clear() -> None:
    err.print("[yellow]cache clear is not implemented yet.[/yellow]")
    raise typer.Exit(code=2)


# ─── sources ────────────────────────────────────────────────────────

sources_app = typer.Typer(help="List and toggle source plugins.", no_args_is_help=True)
app.add_typer(sources_app, name="sources")


@sources_app.command("list")
def sources_list() -> None:
    from dataset_scout.context import ScoutContext

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    for name, cfg in ctx.sources.items():
        state = "[green]enabled[/green]" if cfg.enabled else "[dim]disabled[/dim]"
        err.print(f"  {name:<14} {state}")


@sources_app.command("enable")
def sources_enable(name: str) -> None:
    err.print(
        f"[yellow]sources enable {name} is not implemented yet "
        f"(editing config.toml lands in M1).[/yellow]"
    )
    raise typer.Exit(code=2)


@sources_app.command("disable")
def sources_disable(name: str) -> None:
    err.print(
        f"[yellow]sources disable {name} is not implemented yet "
        f"(editing config.toml lands in M1).[/yellow]"
    )
    raise typer.Exit(code=2)


if __name__ == "__main__":  # pragma: no cover
    app()
