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
that fit a problem — then second-guessing whether the creative
reframings they have in mind actually hold up. `dataset-scout` runs
that loop for you and hands back a curated corpus with receipts:
every claim ties back to a card, a column, and a sample row.

---

## See it in 30 seconds

No HuggingFace token. No Azure OpenAI key. Zero setup:

```bash
uvx --from git+https://github.com/mdressman/dataset-scout dataset-scout tour
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

## Why scout

Three things you don't get from a tab full of HuggingFace search
results:

1. **Reframings, not just matches.** Each candidate gets ranked
   strategies — *direct fit*, *subset extraction*, *signal proxy*,
   *benign baseline* — with rationale, caveats, and a transform
   spec. On novel briefs the typical shape is zero direct fits plus
   a defensible stack of reframings, and that's the deliverable.

2. **Recipes ship `curate`-ready.** Strategy assessment reads actual
   rows before writing the transform, so `recipe.draft.yaml` carries
   real column names and label values — no placeholder
   whack-a-mole between recon and corpus.

3. **Coverage gaps are first-class output.** When the data isn't
   there, scout leads with what's missing and where to look next.
   Sparse coverage is a deliverable, not a failure.

Discovery spans HuggingFace, Kaggle, and academic papers from
NeurIPS / ICML / ICLR / SaTML and other venues — HF/Kaggle URLs
found in paper abstracts are promoted into the candidate pool with
paper provenance.

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
# As a one-off, straight from GitHub (no clone, no PyPI):
uvx --from git+https://github.com/mdressman/dataset-scout dataset-scout tour

# Or install as a uv tool, alias-friendly:
uv tool install git+https://github.com/mdressman/dataset-scout
datascout tour

# Or developer install:
git clone https://github.com/mdressman/dataset-scout
cd dataset-scout
uv sync
uv run datascout --help
```

Python 3.11+. Not on PyPI yet — `uv add dataset-scout` will work
once published; until then use the `git+` URL above. Heavy deps
(litellm, azure-identity, datasets) load lazily — `tour` and
metadata-only runs cost no LLM-import overhead.

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
- **`datascout render <out_dir>`** — re-emit `report.html` / `report.md`
  from an existing `results.json`. No recon, no API calls — useful for
  iterating on report styling. `--html-only` / `--md-only` to skip one.

See [`docs/cli.md`](docs/cli.md) for the full surface.

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
- **Discovery is HuggingFace-lexical-bound.** A dataset whose card
  text doesn't intersect the brief's keywords (or a decomposition
  direction's recalled names) won't surface, even if it's a
  semantically perfect fit. If your construct has well-known named
  benchmarks, list them in the brief — the decomposer turns them
  into proper-noun queries.
- **Recon assesses the top ~35 of ~100 candidates per axis.** That
  cap protects an LLM-cost budget; the rest stay listed but
  unassessed. When an axis comes back empty, sweep the unassessed
  list before declaring a coverage gap.
- **Paper-only datasets (no HF/Kaggle home) won't auto-promote to
  candidates.** Scout extracts dataset references from paper
  abstracts but only promotes ones with a HuggingFace or Kaggle URL.
  Datasets hosted on author sites or generic GitHub repos surface
  as paper citations but stop there — verify those manually.
- **Paper search is rate-limited.** Semantic Scholar throttles under
  parallel runs; arXiv falls back as a targeted second source for
  named-benchmark queries. If both fail, recon proceeds without the
  paper channel rather than blocking.
- **Card metadata is uneven.** License fields are sometimes missing
  or wrong; `language:` is often absent. We surface what's there and
  mark what isn't — no fabrication.
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
