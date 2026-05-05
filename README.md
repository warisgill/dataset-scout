<h1 align="center">dataset-scout</h1>

<p align="center">
  <em>Your dataset-discovery teammate.</em><br>
  Tell it what you need. Get back a curated corpus — with receipts — in minutes.
</p>

<p align="center">
  <a href="docs/getting-started.md">Getting started</a> ·
  <a href="docs/concepts.md">Concepts</a> ·
  <a href="docs/cli.md">CLI</a> ·
  <a href="docs/architecture.md">Architecture</a>
</p>

---

ML practitioners spend hours hand-searching HuggingFace for datasets
that fit their problem — and then second-guessing whether the creative
reframings they have in mind would actually hold up. `dataset-scout`
automates that loop:

1. Reads your brief and **expands it into adjacent search directions**
   an LLM proposes.
2. Pulls real candidates and **samples actual rows** before judging fit.
3. Tells you, with rationale, **which datasets fit directly**, **which
   can be reframed and how**, and **what's missing**.
4. Hands you a normalized JSONL corpus with leakage-aware splits and a
   `recipe.lock.yaml` you can show a reviewer.

Every claim points back at a card, a column, a sample row.
You stay in the loop.

---

## See it in 30 seconds

No HuggingFace token. No Azure OpenAI key. Zero setup:

```bash
uvx dataset-scout tour
```

A complete recon report — decomposition, reframings, coverage gaps,
candidates ranked by fit — for an over-refusal detection program,
rendered to your terminal in under a second from canned demo data.
Add `--out scratch/` to also drop the full set of artefacts
(`results.json`, `recipe.draft.yaml`, `decomposition.yaml`, `report.md`)
on disk so you can poke them.

---

## End-to-end in 5 minutes

```bash
az login                                                     # Entra auth for AOAI
echo "AZURE_OPENAI_ENDPOINT=https://my-aoai.openai.azure.com" > .env
echo "AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini"                  >> .env

# 1. Iterate on the brief cheaply (~5s, one LLM call)
datascout decompose "your brief here" --out scratch/

# 2. When the directions look right, run the full recon
#    Reusing the decomposition skips re-paying for it.
datascout recon "your brief here" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/

# 3. Materialize the corpus straight from the draft recipe.
#    Recipes ship with REAL column names — no hand-editing required.
datascout curate --from scratch/recon/recipe.draft.yaml \
    --out ./mycorpus
```

You end up with a six-file corpus: leakage-aware train / val / test
splits, a `recipe.lock.yaml` audit trail, a 5-second scorecard report,
a deterministic fingerprint, and ready-to-paste snippets for HF
`datasets`, pandas, and raw JSONL.

> **Tip:** `decompose` is the cheap brief-iteration loop. Use it to
> refine your brief before paying for the full ~2-minute recon.

---

## What makes it click

**Three things that turn a 4-hour HF safari into a coffee break:**

1. **Reframings, not just matches.** The LLM doesn't just return
   keyword hits. For each candidate it proposes 1–4 ranked strategies
   — *direct fit*, *reframe these labels*, *use as a hard-negative
   distribution*, *signal proxy* — with confidence, written rationale,
   caveats, and a concrete transform spec. For a novel brief it's
   normal to see *zero* direct fits and a stack of defensible
   reframings instead. That's the value.

2. **Recipes are real, straight from recon.** The strategy assessor
   streams 8 actual rows per candidate before producing its transform,
   so `recipe.draft.yaml` references **real column names** and **real
   label values** — not `prompt_and_response_or_equivalent`
   placeholders. Brief → recon → curate → tens of thousands of rows
   on disk in minutes, no column-name whack-a-mole.

3. **A coverage-gap report = a sourcing roadmap.** When your brief
   pushes into novel territory, the report leads with what's *missing*
   across all the candidates and where to go look (*"nothing covers
   Markdown-specific cloaking; consider Common Crawl + Wayback for
   cloaking-temporal data"*). Sparse coverage is a first-class
   outcome, not a failure — and the deliverable, when it happens.

**Plus academic paper mining** as a fourth discovery channel: relevant
papers from **NeurIPS, ICML, ICLR, and SaTML** are surfaced in their
own report section, and any HuggingFace or Kaggle dataset URLs found
in their abstracts are *promoted into the candidate pool* with paper
provenance — so a dataset cited in a recent NeurIPS paper shows up
beside the HF search hits with the citation as its `surfaced_by`.

The output is upstream of your training and eval workbenches. No
transformation into anyone's downstream schema, no model training.
Hand-off and get out of the way.

---

## Who this is for

Any practitioner who needs to **assemble a labeled corpus from public
data** — quickly, defensibly, and without playing column-name
whack-a-mole. Examples:

- **Detection engineers** building a prompt-injection / over-refusal /
  unsafe-output classifier. *"Find me direct fits where they exist
  and creative reframings of related work so the corpus is robust."*
- **Eval engineers** assembling a labeled reference set for a
  retrospective grade of a deployed system.
- **ML researchers** doing fast triage across HuggingFace for a new
  problem area — *"what's even out there?"*
- **Data scientists** stitching multiple narrow corpora into one
  cohesive blend that preserves class balance.

Sweet spot: AI-security and detection work, where reframings of
adjacent data are how you survive frontier-territory briefs. But the
loop generalizes.

---

## Documentation

- **[Getting started](docs/getting-started.md)** — install, configure,
  run the demo, run the loop end-to-end, with the over-refusal
  detection scenario as the working example.
- **[Hero demo script](docs/demo.md)** — the 7-minute, three-beat
  walkthrough for showing dataset-scout to a team for the first time.
- **[Concepts](docs/concepts.md)** — the mental model: **how to write
  a good brief**, the strategy taxonomy, label kinds, modes, recipes
  and lockfiles.
- **[CLI reference](docs/cli.md)** — every verb, every flag, with
  worked examples.
- **[Architecture](docs/architecture.md)** — pipeline diagram,
  module layout, source plugin contract, milestone status.
- **[Configuration](docs/configuration.md)** — `ScoutContext`, AOAI +
  Entra setup, HuggingFace tokens, every recognised env var.

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

Python 3.11+. Heavy deps (litellm, azure-identity, datasets) load
lazily — `tour` and metadata-only runs cost no LLM-import overhead.

---

## Advanced features

The three-verb core is `tour` / `recon` / `curate` (plus `decompose`
as the cheap-iter helper). The rest of the surface is for specific
needs:

- **`datascout inspect <source>:<id>`** — single-candidate deep-dive:
  schema from a streamed sample, label distribution with Wilson 95% CIs,
  text-length stats, license, and a strategy assessment against your
  current Intent. Pipes cleanly to a file.
- **`datascout compose r1.yaml r2.yaml ...`** — merge multiple
  recipes into one. For multi-detection programs that share a corpus.
- **`datascout judge ./mycorpus --axis X`** — opt-in LLM-as-judge label
  rescue for proxy / weakly-labeled rows. Multi-judge agreement
  (single / majority-of-3 / unanimous-of-5), explicit-gap promotion at
  a configurable threshold, sha256-keyed disk cache, per-batch resumable,
  and `--calibrate-against` reports P/R/F1 vs gold *before* the full
  pass with an optional precision floor that aborts a too-low run.
  Strictly audited; never the default.
- **`datascout eval ./judged --against ./gold`** — score any judged
  corpus against any gold corpus.

See [`docs/cli.md`](docs/cli.md) for the full surface.

---

## Roadmap

| | Status |
|---|---|
| Discovery + cheap probes (HuggingFace) | ✅ shipped |
| LLM decomposition + multi-direction search | ✅ shipped |
| Row-aware strategy assessor + coverage gaps | ✅ shipped |
| `inspect` deep-dive | ✅ shipped |
| `curate` (audit-ready: dedup + leakage-aware splits + filter DSL + soft failures) | ✅ shipped |
| `judge` + `eval` (opt-in LLM-as-judge label rescue) | ✅ shipped |
| **Cache** (SQLite WAL, age-based eviction, namespace-scoped TTLs) | ✅ shipped |
| **Embedding label-intent fit** (Azure OpenAI embeddings, cached) | ✅ shipped |
| **Kaggle source plugin** (discovery-only) | ✅ shipped |
| **HTML report alongside markdown** | ✅ shipped |
| **Academic paper discovery** (NeurIPS / ICML / ICLR / SaTML via Semantic Scholar; abstract URL extraction; promotes cited HF/Kaggle datasets to candidates with paper provenance) | ✅ shipped |
| `--watch` / re-validate mode against upstream revisions | considering |
| Archive option for offline-reproducible corpora | considering |
| Lineage DAG, resumable-operation envelope, multi-candidate portfolio assessor | deferred until requested |

The core loop — **brief → recon → curate** — is feature-complete with
caching, semantic-fit signal, multi-source discovery, paper-citation
mining, and reviewer-friendly output.

---

## Related projects

dataset-scout is one piece of a small toolkit, but it ships and runs
standalone. Each project below interoperates only via published JSONL
contracts — no shared package, no Python-level dependencies.

- **[protozoa-gym](https://github.com/mdressman/protozoa-gym)** — eval
  orchestration for AISP detection enrichments. Reads scout-produced
  corpora via its `scout_corpus` adapter; see
  [`docs/judged-corpus-shape.md`](docs/judged-corpus-shape.md) for the
  spec.
- **[tribunal](https://github.com/mdressman/tribunal)** — multi-agent
  LLM-as-judge framework for *output* evaluation (different question
  shape than scout's row-classification judge). A natural future
  engine for gym's eval-judge step.

---

## Honest limits

- **Strategy assessment is LLM judgment, not ground truth.** Inspect
  samples before committing.
- **Card metadata is uneven.** License fields are sometimes missing or
  wrong; `language:` is often absent. We surface what's there and mark
  what isn't — no fabrication.
- **Recency alone means little.** "Uploaded yesterday" doesn't imply
  the data reflects today's threat surface.
- **Reproducibility is contingent.** `recipe.lock.yaml` pins revisions
  and content hashes; if upstreams delete the data, only an archive
  (a future feature) makes the blend reproducible standalone.
- **Judge verdicts are LLM-generated too.** Conservative promotion
  rules, calibration mode, and precision floors mitigate but don't
  eliminate the risk. A judged label is only as good as the rubric
  you wrote.
- **Not legal advice.** License signals are an SPDX best-effort guess;
  read the upstream card before redistributing.

---

## License

MIT. See [LICENSE](LICENSE).
