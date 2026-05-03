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

dataset-scout uses Azure OpenAI (Entra auth) for the LLM steps
(decomposition, strategy assessment, coverage). Local dev:

```bash
az login
cp .env.example .env   # set AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_DEPLOYMENT
```

The CLI auto-loads `.env` from the current working directory at
startup. With nothing configured the tool runs in metadata-only mode
(HuggingFace search + cheap probes only) and tells you what to set.

A HuggingFace token isn't strictly required for public datasets but
raises rate limits.

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
