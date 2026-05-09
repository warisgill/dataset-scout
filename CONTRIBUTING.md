# Contributing

Thanks for the interest! `dataset-scout` is a small project and PRs
are very welcome — especially anything that **hardens the
experimental `curate` path** (see [Honest limits](README.md#honest-limits)
in the README).

## Quick start

```bash
git clone https://github.com/mdressman/dataset-scout
cd dataset-scout
uv sync                              # install deps
uv run pytest -m "unit or recorded"  # ~50s, gates every PR
uv run ruff check .                  # lint
uv run mypy                          # types (strict on src/)
```

Python 3.11 / 3.12 / 3.13. CI runs the same three commands across
ubuntu and windows runners — keep them green and you're golden.

## Conventions

- **`ruff format` is not enforced repo-wide.** Only run it on files
  you touch.
- **Tests are required for behaviour changes.** `unit` tests use
  pytest + mocks; `recorded` tests use `respx` against snapshots.
  No `live` tests on PRs (those are nightly).
- **Strict mypy** on `src/dataset_scout/` only; tests are unrestricted.
- **Briefs are dataset-shaped, not detector-shaped** — see
  [`docs/concepts.md` §9](docs/concepts.md#9-how-to-write-a-brief)
  if you're contributing to the brief-parsing or strategy layers.
- **Don't break the JSONL contract** in
  [`docs/judged-corpus-shape.md`](docs/judged-corpus-shape.md) without
  bumping its stability note. Downstream consumers depend on it.

## Where help is welcome

In rough priority order:

1. **`curate` validation** — train a model on a scout-curated corpus,
   compare to a hand-built reference, file an issue with what you
   found. (The README is honest that the author hasn't done this end
   to end.)
2. **Curate hardening** — soft-failure classifications, edge cases,
   throughput on large recipes.
3. **New `Source` plugins** — see
   [`docs/architecture.md` §4](docs/architecture.md) for the contract.
4. **Reframings the assessor misses** — surface them via an issue
   with a reproducible brief; we'll iterate on the prompt template.
5. **Doc clarity** — if anything in the README or `docs/` confused
   you on first read, a PR fixing it is gold.

## PR flow

- Branch off `main`, make focused commits, open a PR.
- Run the three commands above locally before pushing.
- CI must be green before merge.
- No CLA, no contributor agreement to sign.

## Issues

Bug reports and feature requests both fine. For bugs, include:
- the brief you ran, redacted as needed
- the command you used
- relevant lines from `report.md` or the stderr output
- python/uv versions (`python --version`, `uv --version`)

For sharp-edged user experiences ("this is technically working but
felt wrong"), the [Honest limits](README.md#honest-limits) section
of the README is fed by real friction — please add to it.
