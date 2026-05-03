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

When you run `datascout recon "<brief>"`:

1. **Brief parsing.** Heuristic parser pulls out languages, threat
   families, deployment context, license preferences. No LLM required
   for this step.
2. **Decomposition** *(when Azure OpenAI is configured)*. The LLM
   proposes 3–7 related search directions adjacent to your brief —
   reframings, hard-negative families, benign baselines, signal proxies
   you might have missed.
3. **Multi-direction search across HuggingFace.** Each direction
   becomes its own HF query; results dedupe into one pool with full
   provenance: every candidate keeps the list of directions that
   surfaced it.
4. **Cheap probes.** Six metadata-driven signals per candidate:
   license (with SPDX best-effort guess), size, recency, freshness
   bucket, declared languages, card-completeness. Each emits a
   `SubScore` with explicit evidence — no aggregate "quality score."
5. **Two-stage shortlist.** Top-k per direction for breadth, then a
   global re-rank by multi-direction hits + license sanity + card
   hygiene caps the assessor input at 15–20 candidates.
6. **Per-candidate strategy assessment** *(LLM)*. Each shortlisted
   candidate gets 1–4 ranked strategies from the 7-kind taxonomy
   (`direct_use`, `subset_extraction`, `label_remapping`,
   `cross_class_repurposing`, `signal_proxy`, `benign_baseline`,
   `not_useful`). Each carries a confidence, a rationale, caveats,
   and a concrete transform spec.
7. **Coverage report** *(LLM)*. After assessment, the model is asked
   what aspects of your target *no* candidate covers — even via
   reframing — and what concrete next steps would close each gap.
8. **Discovery report + structured results + draft recipe.**
   `datascout-out/report.md` leads with the strongest defensible fits;
   per-candidate sections show the chosen strategy, caveats, and
   transform. `datascout-out/results.json` is the same data as JSON.
   `datascout-out/recipe.draft.yaml` is hand-editable input for
   `datascout curate`.

When AOAI isn't configured the tool **degrades cleanly** to
metadata-only mode (search + cheap probes only) and tells you exactly
what to set.

## What's coming

The end-to-end vision (per
[`docs/architecture.md`](docs/architecture.md#milestones)):

- **M3 — `datascout inspect`.** One-candidate deep-dive with full
  strategy assessment.
- **M4b — full audit-grade curate.** The current `curate` is a
  **preview**: deterministic hash-mod split + multimodal coercion +
  filter-DSL hard-fail. M4b adds MinHash dedup, leakage-aware
  splitting, and a minimal filter expression DSL — turning the
  output into the defensible audit artefact reviewers can ask about.
- **M1b** (deferred, parallel track): embedding label-intent fit,
  Kaggle and Papers-With-Code sources, sample-driven probes,
  SQLite cache.

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

- **[Concepts](docs/concepts.md)** — the mental model: discovery vs
  ranking, strategy taxonomy, label kinds, why we never ship a single
  "quality score."
- **[Configuration](docs/configuration.md)** — `ScoutContext`, Azure
  OpenAI + Entra setup (`az login`), HuggingFace tokens, every
  recognised env var.
- **[CLI reference](docs/cli.md)** — every verb, every flag, with
  worked examples.
- **[Architecture](docs/architecture.md)** — pipeline diagram, source
  plugin contract, probe protocol, mode detection, milestone status.

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
