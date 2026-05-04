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
    decomposition_from: Annotated[
        Path | None,
        typer.Option(
            "--decomposition-from",
            help=(
                "Reuse a hand-edited decomposition.yaml instead of paying for "
                "a fresh LLM decompose call. Lets you iterate on directions "
                "without re-running the full decompose step."
            ),
        ),
    ] = None,
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
    from dataset_scout.decomposition_io import load_decomposition, write_decomposition
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.pipeline import run_recon
    from dataset_scout.recipe_draft import write_recipe_draft
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

    directions_override = None
    if decomposition_from is not None:
        try:
            directions_override = load_decomposition(decomposition_from)
        except Exception as e:
            err.print(f"[red]error:[/red] failed to load decomposition: {e}")
            raise typer.Exit(code=1) from e

    try:
        result = run_recon(
            brief,
            ctx=ctx,
            parser_overrides=overrides,
            explore=not no_explore,
            directions_override=directions_override,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    json_path = write_results_json(result, out)
    md_path = write_recon_report(result, out)
    recipe_path = write_recipe_draft(result, out)
    decomposition_path = (
        write_decomposition(result.coverage.decomposition, out)
        if result.coverage and result.coverage.decomposition
        else None
    )

    err.print(
        f"[green]✔[/green] {len(result.candidates)} candidate(s) "
        f"from {', '.join(result.sources_searched) or '(no source)'} "
        f"in {result.elapsed_seconds:.2f}s"
    )
    err.print(f"  - results:       {json_path}")
    err.print(f"  - report:        {md_path}")
    if decomposition_path is not None:
        err.print(f"  - decomposition: {decomposition_path}")
    if recipe_path is not None:
        err.print(f"  - recipe:        {recipe_path}")
    if result.notices:
        for n in result.notices:
            err.print(f"  [yellow]![/yellow] {n}")


# ─── decompose ──────────────────────────────────────────────────────


@app.command(help="Cheap brief-iteration: just decompose, don't search or assess.")
def decompose(
    brief: Annotated[str, typer.Argument(help="Natural-language brief.")],
    detection_target: Annotated[str | None, typer.Option("--detection-target")] = None,
    deployment_context: Annotated[str | None, typer.Option("--deployment-context")] = None,
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Optional path to write decomposition.yaml; otherwise stdout-only.",
        ),
    ] = None,
) -> None:
    """Run only the LLM decomposition step.

    ~5 seconds and one LLM call. Use this to iterate on a brief
    cheaply: see what directions the model proposes, refine, repeat.
    Once happy, pass --decomposition-from to `recon` to skip
    re-paying the decompose cost.
    """
    from dataset_scout.context import ScoutContext
    from dataset_scout.decompose import decompose_intent, llm_available
    from dataset_scout.decomposition_io import write_decomposition
    from dataset_scout.errors import DatasetScoutError, LLMError
    from dataset_scout.intent import HeuristicIntentParser

    ctx = ScoutContext.from_env(is_tty=sys.stderr.isatty())
    if not llm_available(ctx):
        err.print(
            "[red]error:[/red] Azure OpenAI is not configured. "
            "Set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT and "
            "run `az login`."
        )
        raise typer.Exit(code=1)

    overrides: dict[str, object] = {}
    if detection_target:
        overrides["detection_target"] = detection_target
    if deployment_context:
        overrides["deployment_context"] = deployment_context
    intent = HeuristicIntentParser().parse(brief, **overrides)

    try:
        directions = decompose_intent(intent, ctx=ctx)
    except (LLMError, DatasetScoutError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    if not directions:
        err.print("[yellow]The model returned no decomposition directions.[/yellow]")
        return

    # Print to stdout so it pipes cleanly.
    typer.echo(f"# Decomposition for: {brief}\n")
    for d in directions:
        typer.echo(f"## {d.name}\n")
        typer.echo(f"{d.rationale}\n")
        if d.keywords:
            typer.echo(f"- keywords: `{', '.join(d.keywords)}`")
        if d.expected_finds:
            typer.echo(f"- expected: {d.expected_finds}")
        typer.echo("")

    if out is not None:
        path = write_decomposition(directions, out)
        if path is not None:
            err.print(f"[green]✔[/green] {len(directions)} direction(s) written to {path}")
            err.print(
                "  Pass [cyan]--decomposition-from " + str(path) + "[/cyan] to "
                "[cyan]datascout recon[/cyan] to reuse."
            )


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


# ─── compose ────────────────────────────────────────────────────────


@app.command(help="Compose multiple recipes into one merged corpus blueprint.")
def compose(
    recipes: Annotated[
        list[Path],
        typer.Argument(
            help="Two or more recipe.yaml files to merge.",
            metavar="RECIPE...",
        ),
    ],
    out: Annotated[
        Path,
        typer.Option("--out", help="Path to write the merged recipe.yaml."),
    ],
    intent_brief: Annotated[
        str | None,
        typer.Option(
            "--intent-brief",
            help="Override the merged intent's brief (default: keep the first input's).",
        ),
    ] = None,
) -> None:
    """Merge multiple recipes (e.g. from multi-detection-program runs).

    Components dedupe by (source, source_id) — higher
    `strategy_confidence` wins on conflict. Output recipe is fed
    straight to `datascout curate`.
    """
    from dataset_scout.curate import load_recipe
    from dataset_scout.errors import DatasetScoutError
    from dataset_scout.recipe import RecipeIntent
    from dataset_scout.recipe_compose import compose_recipes, write_composed_recipe

    if len(recipes) < 2:
        err.print(
            f"[red]error:[/red] compose needs at least two input recipes (got {len(recipes)})."
        )
        raise typer.Exit(code=1)

    try:
        loaded = [load_recipe(p) for p in recipes]
    except Exception as e:
        err.print(f"[red]error:[/red] failed to load a recipe: {e}")
        raise typer.Exit(code=1) from e

    intent_override = None
    if intent_brief is not None:
        intent_override = RecipeIntent(brief=intent_brief)

    try:
        merged, notices = compose_recipes(loaded, intent_override=intent_override)
    except (ValueError, DatasetScoutError) as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    target = write_composed_recipe(merged, out)
    err.print(
        f"[green]✔[/green] merged {len(loaded)} recipe(s) into {target} — "
        f"{len(merged.components)} components, {len(merged.declined)} declined"
    )
    for n in notices:
        err.print(f"  [yellow]![/yellow] {n}")


# ─── curate ─────────────────────────────────────────────────────────


@app.command(help="Build a schema-normalized corpus from a recipe.")
def curate(
    from_: Annotated[Path, typer.Option("--from", help="Path to recipe.yaml.")],
    out: Annotated[Path, typer.Option("--out", help="Output corpus directory.")] = Path("mycorpus"),
    min_strategy_confidence: Annotated[
        float | None, typer.Option("--min-strategy-confidence", min=0.0, max=1.0)
    ] = None,
    seed: Annotated[int | None, typer.Option("--seed", help="Override the recipe's seed.")] = None,
    max_rows_per_component: Annotated[
        int | None,
        typer.Option(
            "--max-rows-per-component",
            min=1,
            help=(
                "Cap rows materialized per component for this run. Lowers but "
                "never raises the recipe's `take` value. Useful for fast "
                "iteration on heavy code/text corpora without hand-editing "
                "the recipe."
            ),
        ),
    ] = None,
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
            max_rows_per_component=max_rows_per_component,
        )
    except DatasetScoutError as e:
        err.print(f"[red]error:[/red] {e}")
        raise typer.Exit(code=1) from e

    err.print(
        f"[green]✔[/green] {result.total_rows} row(s) written to {result.out_dir} "
        f"({result.components_kept} component(s) kept, "
        f"{result.components_skipped} skipped"
        + (f", {result.components_failed} failed" if result.components_failed else "")
        + f") in {result.elapsed_seconds:.2f}s"
    )
    splits_str = " · ".join(f"{n}={c}" for n, c in result.rows_per_split.items())
    err.print(f"  - splits: {splits_str}")
    err.print(f"  - fingerprint: {result.fingerprint[:16]}...")
    if result.failures:
        err.print(
            f"  [yellow]![/yellow] {len(result.failures)} component(s) skipped "
            "due to upstream errors — see report.md / recipe.lock.yaml "
            "→ failed_components for hints:"
        )
        for f in result.failures[:5]:
            err.print(f"    - {f['id']} [{f['category']}]: {f['hint']}")
        if len(result.failures) > 5:
            err.print(f"    ... and {len(result.failures) - 5} more in the lockfile.")
    err.print("  [green]✔[/green] audit-ready: leakage-aware splits + filter DSL applied.")


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


# ─── tour ───────────────────────────────────────────────────────────


@app.command(help="30-second demo with canned data — no AOAI or HF access required.")
def tour(
    out: Annotated[
        Path | None,
        typer.Option(
            "--out",
            help="Optional path to also persist the demo artefacts (results.json / recipe.draft.yaml / decomposition.yaml).",
        ),
    ] = None,
) -> None:
    """Render a fully-populated tour report to stdout."""
    from dataset_scout.tour import render_tour

    typer.echo(render_tour(out_dir=out))
    if out is not None:
        err.print(
            f"[green]✔[/green] tour artefacts also written to {out} — "
            "open report.md / recipe.draft.yaml / decomposition.yaml to explore."
        )
