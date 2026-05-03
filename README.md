# dataset-scout

> **Public-dataset reconnaissance for AI security work.**
> Find direct-fit corpora, surface defensible reframings of related work,
> and hand off audit-ready data to your detection workbench.

Detection engineers, forensic analysts, and incident responders spend
hours hand-searching HuggingFace and second-guessing whether the
creative reframings they have in mind are defensible. `dataset-scout`
automates that recon. It reads your brief, expands the search net into
adjacent directions an LLM proposes, and tells you — with receipts —
which datasets fit directly, which can be reframed (and how), and
which to skip.

It is **upstream of your detection workbench.** It doesn't label,
transform into your downstream's native schema, or train models. It
hands off a normalized, leakage-aware corpus and a defensible
`recipe.lock.yaml`, and gets out of the way.

```bash
# 30-second tour
uv tool install dataset-scout            # or `uv add` in your project
az login                                 # Entra auth for Azure OpenAI (optional)
echo "AZURE_OPENAI_ENDPOINT=https://my-aoai.openai.azure.com" > .env
echo "AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini" >> .env

datascout recon "find labeled prompt injection corpora for a RAG app"
# → datascout-out/{report.md, results.json, recipe.draft.yaml}
```

---

## What you get today

End-to-end working flow:

1. **`datascout recon "<brief>"`** — discovery + decomposition + per-candidate strategy assessment + coverage gaps → `report.md` + `results.json` + `recipe.draft.yaml`.
2. **`datascout inspect <source>:<id>`** — single-candidate deep-dive with schema, label distribution (Wilson 95% CIs), sample rows, license, and strategy assessment.
3. **`datascout curate --from recipe.yaml --out ./mycorpus`** — recipe → schema-normalized JSONL + lockfile + manifest + report + fingerprint + usage. *(M4a preview: hash-mod split, MinHash dedup + leakage-aware splitter land in M4b.)*

When AOAI isn't configured, recon and inspect **degrade cleanly** to metadata-only mode and tell you exactly what to set.

➡️ **[`docs/getting-started.md`](docs/getting-started.md)** is a step-by-step walkthrough from zero to a curated corpus.

## What's coming

- **M4b — audit-grade `curate`.** MinHash dedup + leakage-aware splitter + minimal filter expression DSL. Flips `recipe.lock.yaml`'s `audit_readiness` from `preview` to `ready`.
- **M1b** (deferred, parallel track): embedding label-intent fit, Kaggle and Papers-With-Code sources, sample-driven probes, SQLite cache.

See [`docs/architecture.md`](docs/architecture.md) for the full picture.

---

## Use cases

1. **Detection authoring.** *"Find labeled public datasets I can use
   to train and evaluate a [prompt-injection / jailbreak / unsafe-output]
   detector — direct fits where they exist, and creative reframings
   of related work so the corpus is rich and robust."*
2. **IR / forensic retrospective evals.** *"Assemble a labeled
   reference set so I can retrospectively grade a deployed
   detector."*
3. **Threat-family coverage.** *"I have one corpus. Find complementary
   datasets that cover attack styles mine misses, and blend them
   while preserving class balance."*

---

## Documentation

- **[Getting started](docs/getting-started.md)** — install, configure, run your first recon → inspect → curate end-to-end.
- **[Concepts](docs/concepts.md)** — the mental model: discovery vs ranking, strategy taxonomy, label kinds, why we never ship a single "quality score."
- **[Configuration](docs/configuration.md)** — `ScoutContext`, Azure OpenAI + Entra setup (`az login`), HuggingFace tokens, every recognised env var.
- **[CLI reference](docs/cli.md)** — every verb, every flag, with worked examples.
- **[Architecture](docs/architecture.md)** — pipeline diagram, source plugin contract, probe protocol, mode detection, milestone status.

---

## Install

Two equivalent CLI entry points: `dataset-scout` (formal) and
`datascout` (recommended short form).

```bash
# As a one-off:
uvx dataset-scout recon "your brief"

# Or as a project dependency:
uv add dataset-scout

# Or developer install:
git clone https://github.com/<your-org>/dataset-scout
cd dataset-scout
uv sync
uv run datascout --help
```

Python 3.11+. Heavy deps (litellm, azure-identity) are loaded lazily —
metadata-only runs cost no LLM-import overhead.

---

## Honest limits

- **Strategy assessment is LLM-judgment, not ground truth.** Always
  inspect samples before committing.
- **Card metadata is uneven.** License fields are sometimes missing
  or wrong; `language:` is often absent; many cards lack content
  dates. We surface what's there and clearly mark what isn't —
  no fabrication.
- **Recency alone means little.** "Uploaded yesterday" doesn't imply
  the data is current to today's threat landscape.
- **Reproducibility is contingent.** When `recipe.lock.yaml` lands
  (M4) it pins revisions and content hashes; if upstreams delete the
  data, only an archive (deferred) makes the blend reproducible
  standalone.
- **Not legal advice.** License signals are an SPDX best-effort
  guess; always read the upstream card before redistributing.

## License

MIT. See [LICENSE](LICENSE).
