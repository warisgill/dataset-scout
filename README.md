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
