# Copilot agent guidance for dataset-scout

Lightweight conventions for AI agents working in this repo. User
docs live in `README.md` and `docs/`; this file is what *agents*
need to know.

## What this repo is

`dataset-scout` is a CLI for AI-safety dataset discovery and
curation. The pipeline is **brief → recon → curate**, with optional
**inspect**, **judge**, **eval**. Discovery surfaces HuggingFace +
Kaggle candidates and academic papers (Semantic Scholar + arXiv),
cheaply probes them, then runs an LLM strategy assessor over the
top ~20 to recommend `direct_use` / `subset_extraction` /
`signal_proxy` / `benign_baseline` / `not_useful` per candidate.

## Run loop

```powershell
uv sync                               # install deps
uv run pytest tests/ -q               # 553 unit tests, ~37s
uv run ruff check src tests           # lint
uv run mypy src/dataset_scout         # strict mode, must be clean
uv run datascout recon "<brief>" --out datascout-out/<name>
```

Tests gate every change. **Always run unit tests + mypy before
committing.** `ruff format` is not enforced repo-wide; only run it
on files you touched (the team has not formatted everything yet).

## Conventions

- **Briefs are <250 chars and dataset-shaped, not detector-shaped.**
  See `docs/concepts.md` §9. If your construct has well-known named
  benchmarks (e.g. `INTIMA`, `AnthroBench`), include them in the
  brief — the decomposer turns them into proper-noun queries that
  also fire arXiv.
- **Windows CRLF.** This repo's working tree uses `\r\n`. Don't
  introduce LF-only files; the `edit` tool preserves whatever the
  file already uses.
- **Output goes to `datascout-out/`** (gitignored). Never commit
  recon output. Same for `scratch/` and `mycorpus/`.
- **AOAI auth is Entra-based.** Run `az login` before pipeline tests
  that hit the real LLM. `.env` carries `AZURE_OPENAI_ENDPOINT` +
  `AZURE_OPENAI_DEPLOYMENT`.

## Test-writing patterns

- `respx` mocks all HTTP. See `tests/unit/test_paper_search.py` and
  `test_arxiv_search.py` for the canonical pattern.
- The arXiv module has a 3-second global rate gate. Tests use an
  autouse fixture that resets `arxiv_search._LAST_CALL_AT[0] = 0.0`
  and monkeypatches `dataset_scout.arxiv_search.time.sleep` to a
  no-op. Same trick for the paper-search retry helper:
  `dataset_scout.paper_search.time.sleep`. Both `time` and `random`
  are imported at module level expressly so monkeypatch can find
  them.
- Pipeline-level tests inject `paper_search_fn=<callable | False>`
  to bypass real network calls — see
  `test_pipeline_promotes_paper_candidates_into_pool`.

## Honest limits to remember (don't oversell)

1. Discovery is HuggingFace-lexical-bound. A dataset whose card
   doesn't intersect brief keywords or `recalled_dataset_names`
   won't surface even if it's a perfect semantic fit. Document
   missed cases as gaps; don't quietly fix the brief.
2. Strategy assessor caps at ~20 of ~100 per axis (LLM-cost
   budget). When an axis returns empty, sweep the unassessed list
   before declaring a coverage gap.
3. Paper-only datasets (no HF/Kaggle home) won't auto-promote.
   Scout shows them as paper citations and stops there.
4. Paper search is rate-limited. Semantic Scholar throttles under
   parallel runs; arXiv falls back as a targeted second source for
   named-benchmark queries. If both fail, recon proceeds without
   the paper channel rather than blocking.

## Don'ts

- Don't add `## Roadmap` / `## Status` sections back. The README is
  for value-prop and honest limits; project status belongs in
  commits and changelogs.
- Don't add fuzzy title-similarity dedupe to paper search; cross-
  backend dedupe is keyed on `arxiv_id`. Tolerate small duplicate
  rates over false merges.
- Don't fire arXiv on every query. It's a targeted fallback for
  named-benchmark queries only; broader queries hit S2 only.
- Don't bypass `http_get_with_retry` when adding a new HTTP-backed
  source — the retry policy is the contract.
