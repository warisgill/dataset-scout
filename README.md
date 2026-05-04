# dataset-scout

> **Brief in. Curated, audit-ready corpus out.**
> Public-dataset reconnaissance for AI security work — direct-fit
> corpora when they exist, defensible reframings of related work when
> they don't, and a hand-off-able JSONL with full provenance.

Detection engineers, forensic analysts, and incident responders spend
hours hand-searching HuggingFace for datasets that fit their threat
model — then second-guessing whether the creative reframings they
have in mind would survive an audit. `dataset-scout` automates that
loop. It reads your brief, expands the search net into adjacent
directions an LLM proposes, fetches real rows from each candidate,
tells you (with receipts) which datasets fit directly, which can be
reframed and how, and what your candidate set is missing — then
materializes a normalized, leakage-aware corpus with a defensible
`recipe.lock.yaml`.

It's **upstream of your detection workbench.** No labeling,
no transformation into your downstream's native schema, no model
training. Hand-off and get out of the way.

---

## See it in 30 seconds

No HuggingFace token. No Azure OpenAI key. Zero setup:

```bash
uvx dataset-scout tour
```

A complete recon report — decomposition, strategies, coverage gaps,
candidates ranked by reframing strength — for an over-refusal
detection program, rendered to your terminal in under a second from
canned demo data. `--out scratch/` also persists the full set of
artefacts (`results.json`, `recipe.draft.yaml`, `decomposition.yaml`,
`report.md`) so you can poke them.

When you're ready, give it a real brief.

---

## End-to-end in 5 minutes

```bash
az login                                                     # Entra auth for AOAI
echo "AZURE_OPENAI_ENDPOINT=https://my-aoai.openai.azure.com" > .env
echo "AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini"                  >> .env

# 1. Iterate on the brief cheaply (~5 s, one LLM call)
datascout decompose "your brief here" --out scratch/

# 2. Once the directions look right, run the full recon
#    reusing the decomposition (skips re-paying the decompose call)
datascout recon "your brief here" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/

# 3. Curate straight to JSONL — recipe.draft.yaml has real
#    column names because the assessor sampled rows during recon
datascout curate --from scratch/recon/recipe.draft.yaml --out ./mycorpus
```

The output is a six-file corpus with leakage-aware splits, a
`recipe.lock.yaml` audit record, a 5-second scorecard report, a
fingerprint, and snippet-pasteable usage instructions for HF
`datasets`, pandas, and raw JSONL.

> **Tip:** `datascout decompose` is the cheap brief-iteration loop.
> Use it to refine briefs before you pay for the full ~2-minute recon.

---

## What makes it different

**The wow moments after a real run:**

1. **Coverage-gap report = sourcing roadmap.** When a brief explores
   novel territory, the LLM tells you what `s missing across all the
   candidates and where to find it ("nothing covers Markdown-specific
   cloaking; nothing has DOM-level localisation; consider Common
   Crawl + Wayback for cloaking-temporal data"). This is the report's
   lead section, not buried at the bottom.

2. **Recipes are curate-ready straight from recon.** The strategy
   assessor fetches 8 real rows per candidate before producing its
   transform spec, so `recipe.draft.yaml` has actual column names and
   actual label values — not `prompt_and_response_or_equivalent`
   placeholders. End-to-end demo: brief → `recon` → `curate` →
   200K rows materialized in ~5 minutes with no hand-editing.

3. **Sparse coverage is a first-class outcome, not a failure.** When
   HF returns 1 candidate (because the brief explores frontier
   territory) the report explicitly says "this is sparse-coverage
   territory; the decomposition + gaps are your real deliverable" —
   not "no candidates returned, try broadening." Honest framing.

4. **The strategy taxonomy makes reframings legible.** Each candidate
   gets 1–4 ranked strategies from a 7-kind taxonomy (`direct_use`,
   `subset_extraction`, `label_remapping`, `cross_class_repurposing`,
   `signal_proxy`, `benign_baseline`, `not_useful`) with confidence,
   rationale, caveats, and a concrete transform spec. For novel
   threats it's normal to see zero `direct_use` results — every
   candidate is a reframing — and the report flags this honestly.

5. **`label_kind` is load-bearing.** Output JSONL marks each row's
   provenance: `ground_truth`, `subset_extracted`, `remapped`, or
   `proxy`. Train on everything; **eval excludes proxies by
   contract.** No "we trained on synthetic data" surprises.

6. **`recipe.lock.yaml` is the defensible record.** Pinned
   revisions, realised counts, content hashes, label-kind
   distributions per component, override sources for every CLI flag.
   The single file a reviewer can ask about.

---

## Use cases

1. **Detection authoring.** *"Find labeled public datasets I can use
   to train and evaluate a [prompt-injection / over-refusal /
   unsafe-output] detector — direct fits where they exist, and
   creative reframings of related work so the corpus is rich and
   robust."*
2. **IR / forensic retrospective evals.** *"Assemble a labeled
   reference set so I can retrospectively grade a deployed detector."*
3. **Threat-family coverage.** *"I have one corpus. Find
   complementary datasets that cover attack styles mine misses, and
   blend them while preserving class balance."*
4. **Multi-detection programs.** *"I have three threat-intel
   detection sub-programs sharing a backbone — produce one corpus
   that backs all three."*  (See `datascout compose`.)

---

## Documentation

- **[Getting started](docs/getting-started.md)** — install, configure,
  run the demo, run the loop end-to-end.
- **[Concepts](docs/concepts.md)** — the mental model: discovery vs
  ranking, **how to write a brief**, the strategy taxonomy, label
  kinds, modes, recipes / lockfiles.
- **[Configuration](docs/configuration.md)** — `ScoutContext`, Azure
  OpenAI + Entra setup (`az login`), HuggingFace tokens, every
  recognised env var.
- **[CLI reference](docs/cli.md)** — every verb (`tour`, `decompose`,
  `recon`, `inspect`, `curate`, `compose`, `cache`, `sources`),
  every flag, with worked examples.
- **[Architecture](docs/architecture.md)** — pipeline diagram, source
  plugin contract, probe protocol, mode detection, milestone status.

---

## Install

Two equivalent CLI entry points: `dataset-scout` (formal) and
`datascout` (recommended short form).

```bash
# As a one-off:
uvx dataset-scout tour

# Or as a project dependency:
uv add dataset-scout

# Or developer install:
git clone https://github.com/<your-org>/dataset-scout
cd dataset-scout
uv sync
uv run datascout --help
```

Python 3.11+. Heavy deps (litellm, azure-identity, datasets) are
loaded lazily — `tour` and metadata-only runs cost no LLM-import
overhead.

---

## Honest limits

- **Strategy assessment is LLM-judgment, not ground truth.** Always
  inspect samples before committing.
- **Card metadata is uneven.** License fields are sometimes missing
  or wrong; `language:` is often absent; many cards lack content
  dates. We surface what's there and clearly mark what isn't —
  no fabrication.
- **Recency alone means little.** "Uploaded yesterday" doesn't imply
  the data reflects today's threat surface.
- **Reproducibility is contingent.** `recipe.lock.yaml` pins
  revisions and content hashes. If upstreams delete the data, only an
  archive (a future feature) makes the blend reproducible standalone.
- **`curate` is currently `audit_readiness: preview`.** Hash-mod
  splits + no MinHash dedup + filter expressions hard-fail until M4b
  ships them. The lockfile says so explicitly. Don't ship a defended
  detector trained on a preview corpus without acknowledging it.
- **Not legal advice.** License signals are an SPDX best-effort
  guess; read the upstream card before redistributing.

---

## License

MIT. See [LICENSE](LICENSE).
