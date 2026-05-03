# dataset-scout

> Reconnaissance, reframing, and curation of public datasets for AI
> detection engineers, forensic analysts, and incident responders.

Three CLI verbs:

- `dataset-scout recon "<brief>"` — find candidate datasets and assess
  how each could be used.
- `dataset-scout inspect <id>` — deep-dive on one candidate.
- `dataset-scout curate --from recipe.yaml` — build a schema-normalized,
  leakage-aware corpus with an audit trail.

Alias: `dscout`.

## Status

Early development. Not yet released.

## Configure

Copy `.env.example` to `.env` and fill in any keys you need. The CLI
auto-loads `.env` from the current working directory at startup. A HF
token isn't strictly required for public datasets, but it raises rate
limits and is the friendlier default.

## Install (development)

```bash
uv sync
uv run dataset-scout --help
```

## Develop

```bash
uv run pytest -m unit
uv run ruff check .
uv run mypy
```

## License

MIT.
