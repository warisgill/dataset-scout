# Getting started

A walkthrough from zero to a curated corpus, end to end.

## The scenario

You're the detection lead at a B2B SaaS that ships an AI customer-support
agent. Production traffic shows a new failure mode: the model has
started **over-refusing** — declining benign, legitimate requests by
citing safety concerns. ("I can't help with that" when the user asks
about how to use a product feature.) Support tickets are stacking up,
the trust-and-safety team is unhappy, and you need a labeled corpus to
train an over-refusal detector that can flag these in real time.

This is **not** prompt injection. It's not a jailbreak. It's the
opposite — your model is being *too* cautious. There's no obvious
public dataset called "over-refusal in customer support," but there is
relevant adjacent work: refusal benchmarks, safety-tuning datasets,
exaggerated-caution test suites. `dataset-scout` is built for exactly
this shape of problem.

We'll go from a one-line brief to a JSONL corpus you can hand to your
training pipeline.

---

## 1. Install

`dataset-scout` is a Python 3.11+ package. The fastest path is `uv`:

```bash
# As a one-off CLI tool (recommended for evaluation)
uv tool install dataset-scout

# Or in a project
uv add dataset-scout
```

Verify:

```bash
datascout --version
# dataset-scout 0.0.1
```

(`dataset-scout` and `datascout` are equivalent — pick whichever you
prefer.)

---

## 2. Configure

`dataset-scout` reads a `.env` file from the current working
directory. Create one — start from the bundled example:

```bash
cp .env.example .env
```

The minimum useful configuration:

```bash
# .env
HUGGINGFACE_HUB_TOKEN=hf_...                              # raises HF rate limits
AZURE_OPENAI_ENDPOINT=https://your-aoai.openai.azure.com  # for LLM steps
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
```

Auth for Azure OpenAI is **Entra**, not API keys. Locally, `az login`
is enough — `DefaultAzureCredential` chains through `az login`,
managed identity, and service-principal env vars in CI.

```bash
az login
```

If you skip the AOAI block, `dataset-scout` runs in **metadata-only
mode** and tells you what to set. We'll see both modes below.

See [`configuration.md`](configuration.md) for every recognised
variable.

---

## 3. First run — metadata-only

Let's start without the LLM, to feel the metadata-only fallback:

```bash
datascout recon "labeled corpora for detecting over-refusal in customer support agents"
```

You'll see something like:

```
✔ 50 candidate(s) from huggingface in 3.2s
  - results: datascout-out/results.json
  - report:  datascout-out/report.md
  ! Running in metadata-only mode: Azure OpenAI is not configured, so
    decomposition, strategy assessment, and coverage gaps were skipped.
    To enable them, copy .env.example to .env, set AZURE_OPENAI_ENDPOINT
    and AZURE_OPENAI_DEPLOYMENT, and run `az login`.
```

Open `datascout-out/report.md`. The header is honest:

> ⚠️ **Metadata-only mode.**
> Azure OpenAI is not configured, so decomposition, strategy assessment,
> and coverage gaps were skipped.

Below that, 50 HuggingFace candidates in source-relevance order with
license badges, freshness buckets, declared languages, card
completeness — **annotations, not a ranking score**. This is real
discovery output: useful for triage, not yet a defensible
recommendation.

---

## 4. Light up the LLM

Once your `.env` has `AZURE_OPENAI_ENDPOINT` + `AZURE_OPENAI_DEPLOYMENT`
and `az login` has been run, re-run the same command:

```bash
datascout recon "labeled corpora for detecting over-refusal in customer support agents"
```

Now the pipeline does the full thing:

1. **Decomposes** the brief into 3–7 search directions. The LLM might
   propose: `safety_tuning_corpora`, `refusal_benchmarks`,
   `helpful_assistant_baselines`, `exaggerated_caution_examples`,
   `customer_support_dialogue`. Each becomes its own HF query.
2. **Searches across all directions**, dedupes, merges
   surfacing-direction provenance.
3. **Runs the cheap probes** (license, freshness, languages, etc.).
4. **Two-stage shortlist** picks ~15–20 candidates worth spending LLM
   budget on.
5. **Per-candidate strategy assessor** rates each shortlisted
   candidate against the seven-strategy taxonomy. Some come back as
   `direct_use` (an over-refusal benchmark like `bench-llm/or-bench`),
   others as `signal_proxy` (general safety-tuning data, useful for
   training but not eval), or `benign_baseline` (helpful-assistant
   conversations).
6. **Coverage gaps** — what aspects of your problem nothing addresses,
   and what to do about it ("nothing covers customer-support tone
   specifically; consider augmenting with internal logs").
7. Writes `report.md`, `results.json`, and `recipe.draft.yaml`.

The new report leads with the strongest defensible fits, each one
shows its strategy badge (✅ direct use, 📡 signal proxy, etc.),
confidence, rationale, caveats, and a transform spec the curate step
will use.

Skim `datascout-out/report.md`. Note which candidates the LLM tagged
`direct_use` versus reframed as proxies — that's the value-add over
naive HF search.

---

## 5. Inspect a candidate

A strategy badge is a hypothesis. Before you commit, **inspect** the
top candidate yourself:

```bash
datascout inspect huggingface:bench-llm/or-bench --intent-from datascout-out/results.json
```

Output (to stdout, pipeable):

- **Identity** — the card URL, revision, gating posture.
- **License** — raw + SPDX guess.
- **Card-declared metadata** — languages, tags, dates, downloads.
- **Schema** — column names + inferred types from a 50-row sample.
- **Label distribution** — counts and **Wilson 95% CIs** so you don't
  over-interpret 50 rows.
- **Text length** — min / median / max chars in the picked text column.
- **Sample rows** — first five, so you can read what's actually in there.
- **Strategy assessment** — same logic as recon, run on this one
  candidate against the Intent recon used (`--intent-from` carries the
  Intent over).

Pipe it into your favourite Markdown viewer, or `> inspect.md` and
review later:

```bash
datascout inspect huggingface:walledai/XSTest --intent-from datascout-out/results.json > xstest-inspect.md
```

Repeat for two or three candidates that recon shortlisted.

---

## 6. Hand-edit the recipe

`datascout-out/recipe.draft.yaml` is your starting point. Open it. The
top section captures your intent verbatim and pins the
`min_strategy_confidence` threshold. Below that, each kept candidate
is one `components` entry:

```yaml
components:
  - id: huggingface_bench-llm_or-bench
    source: huggingface
    source_id: bench-llm/or-bench
    revision: 4f61ec...
    surfaced_by: [refusal_benchmarks]
    strategy: direct_use
    strategy_confidence: 0.85
    rationale: |
      Direct fit: prompts paired with whether a strict-tuned model
      over-refuses. Maps cleanly to a binary over-refusal classifier.
    transform:
      text_column: prompt
      label_column: category
      label_value_map:
        toxic: positive
        hate: positive
        ...
      label_kind_map:
        all: ground_truth
      filter: null
      take: all
```

Common edits:

- **Drop weak components** — delete entries you don't trust. They land
  in the `declined` list with a reason.
- **Tighten `label_value_map`** — the assessor's mapping is a
  proposal. Read the source's label values and adjust.
- **Cap `take`** — for proxies, `take: 2000` keeps the corpus
  balanced so a large signal-proxy doesn't drown the direct fits.
- **Move the file** — copy `recipe.draft.yaml` to `recipe.yaml` once
  you're happy. The draft can be regenerated; the edited recipe is
  yours.

```bash
cp datascout-out/recipe.draft.yaml recipe.yaml
$EDITOR recipe.yaml
```

The recipe is the seam between recon and curate. Anything you put in
here, the corpus reflects.

---

## 7. Curate the corpus

```bash
datascout curate --from recipe.yaml --out ./over-refusal-corpus
```

Output:

```
✔ 4,231 row(s) written to over-refusal-corpus (5 component(s) kept,
  2 skipped) in 47.8s
  - splits: train=3385 · val=419 · test=427
  - fingerprint: 8f3a40b1c2d4e007...
  ! preview build — hash-mod split, no dedup. Audit-ready splitting +
    MinHash land in M4b.
```

Inside `over-refusal-corpus/`:

```
├── train.jsonl / val.jsonl / test.jsonl   schema-normalized records
├── recipe.yaml                             your edited recipe (verbatim)
├── recipe.lock.yaml                        pinned revisions, realized counts, hashes
├── manifest.json                           machine-readable lock equivalent
├── report.md                               5-second scorecard + provenance
├── fingerprint.txt                         one-line content hash
└── usage.md                                3-line snippets for HF datasets / pandas
```

A sample row:

```json
{
  "text": "How do I remove the password from my account?",
  "label": "positive",
  "label_kind": "ground_truth",
  "strategy": "direct_use",
  "strategy_confidence": 0.85,
  "source": "huggingface:bench-llm/or-bench",
  "source_row_id": "742",
  "source_revision": "4f61ec...",
  "source_config": null,
  "source_split": "train",
  "threat_family": null,
  "extras": { "/* every original column verbatim */" },
  "extras_coercion": false
}
```

Two fields to internalise:

- **`label_kind`** — `ground_truth` rows are safe to train AND eval on.
  `proxy` rows are train-only — *exclude them from your eval set*. The
  field is load-bearing for honest measurement.
- **`extras`** — the original source row is preserved verbatim (with
  multimodal coercion when needed). Nothing is dropped.

---

## 8. The audit trail

Open `over-refusal-corpus/recipe.lock.yaml`. This is the file a
reviewer asks about: which corpus did this detector train on, and how
did proxies factor in?

```yaml
recipe_version: '1'
audit_readiness: preview
audit_readiness_notes:
  - Hash-mod split is deterministic but NOT leakage-aware.
  - MinHash dedup is deferred to M4b; near-duplicate rows may cross splits.
  - Filter DSL is deferred to M4b; recipes with non-null filter are rejected.
intent: { ... }
min_strategy_confidence:
  recipe: 0.5
  effective: 0.5
  overridden_by_cli: false
seed: { recipe: 42, effective: 42, overridden_by_cli: false }
splits:
  method: hash_mod
  proportions_recipe: { train: 0.8, val: 0.1, test: 0.1 }
  realized: { train: 3385, val: 419, test: 427 }
components:
  - id: huggingface_bench-llm_or-bench
    source: huggingface
    source_id: bench-llm/or-bench
    revision: 4f61ec...
    strategy: direct_use
    strategy_confidence: 0.85
    realized:
      rows_taken: 1247
      label_kind_counts: { ground_truth: 1247 }
      row_identity_method: column:id
fingerprint: 8f3a40b1c2d4e007...
scout_version: 0.0.1
```

`audit_readiness: preview` is honest. M4b will flip it to `ready` once
MinHash dedup + leakage-aware splitting land. Until then, this is a
working corpus, not yet a defensible one — and the lockfile says so.

---

## 9. Use it downstream

`usage.md` has the three snippets you'll most often want:

```python
# huggingface_hub datasets
from datasets import load_dataset
ds = load_dataset("json", data_files={
    "train": "over-refusal-corpus/train.jsonl",
    "val":   "over-refusal-corpus/val.jsonl",
    "test":  "over-refusal-corpus/test.jsonl",
})

# pandas
import pandas as pd
train = pd.read_json("over-refusal-corpus/train.jsonl", lines=True)

# raw jsonl
import json
rows = [json.loads(l) for l in open("over-refusal-corpus/train.jsonl")]
```

A reasonable training-time filter:

```python
# Train on everything; eval on ground-truth only.
train = ds["train"]
eval_ = ds["val"].filter(lambda r: r["label_kind"] == "ground_truth")
```

---

## 10. Honest limits

- **Strategy assessment is LLM-judgment**, not ground truth. Read the
  rationale and inspect samples before committing.
- **Card metadata is uneven.** Licenses are sometimes wrong; languages
  are often missing. We surface what's there and clearly mark what
  isn't.
- **Recency alone means little.** "Uploaded yesterday" doesn't imply
  the data reflects today's threat surface.
- **Reproducibility is contingent.** `recipe.lock.yaml` pins
  revisions and content hashes. If upstreams delete the data, only an
  archive (a future feature) makes the blend reproducible standalone.
- **Not legal advice.** License signals are an SPDX best-effort guess;
  read the upstream card before redistributing.

---

## What's next

- **[CLI reference](cli.md)** — every flag, every verb.
- **[Concepts](concepts.md)** — the strategy taxonomy, label kinds,
  why discovery vs ranking matters.
- **[Configuration](configuration.md)** — env vars, paths, programmatic
  `ScoutContext`.
- **[Architecture](architecture.md)** — pipeline, sources, probes,
  milestone status.

Found a sharp edge? Open an issue. The honest limits list is fed by
real friction; please add to it.
