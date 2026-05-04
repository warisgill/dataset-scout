# Getting started

A walkthrough from zero to a curated corpus, end to end.

## 0. See it in 30 seconds (no setup)

Before you configure anything, see what dataset-scout produces:

```bash
uvx dataset-scout tour
```

A complete recon report — decomposition, strategies, coverage gaps,
candidates ranked by reframing strength — for an over-refusal
detection program, rendered in <1s from canned demo data. No
HuggingFace token, no Azure OpenAI, no waiting.

```bash
uvx dataset-scout tour --out scratch/
```

…also persists `results.json`, `recipe.draft.yaml`,
`decomposition.yaml`, and `report.md` so you can poke at the full
output shape.

When you're ready, give it a real brief.

---

## The scenario

You're the detection lead at a B2B SaaS that ships an AI customer-
support agent. Production traffic shows a new failure mode: the
model has started **over-refusing** — declining benign, legitimate
requests by citing safety concerns. ("I can't help with that" when
the user asks about how to use a product feature.) Support tickets
are stacking up, the trust-and-safety team is unhappy, and you need
a labeled corpus to train an over-refusal detector.

This is **not** prompt injection. It's not a jailbreak. It's the
opposite — your model is being *too* cautious. There's no obvious
public dataset called "over-refusal in customer support," but there
is relevant adjacent work: refusal benchmarks, safety-tuning datasets,
exaggerated-caution test suites. dataset-scout is built for exactly
this shape of problem.

---

## 1. Install + configure

Three minutes of one-time setup:

```bash
# Install
uv tool install dataset-scout

# Or in a project: uv add dataset-scout

# Configure Azure OpenAI (Entra auth — no API keys)
az login
cp .env.example .env
# Edit .env to set:
#   AZURE_OPENAI_ENDPOINT=https://my-aoai.openai.azure.com
#   AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
#   HUGGINGFACE_HUB_TOKEN=hf_...        (optional, raises HF rate limits)
```

Without the AOAI block dataset-scout runs in metadata-only mode
(HuggingFace search + cheap probes only). It tells you what to set.

See [`configuration.md`](configuration.md) for every recognised
variable and CI / managed-identity options.

---

## 2. Write a good brief (this is the real input)

The brief is the only input that drives everything downstream. **A
good brief describes the dataset you want, not the detector you'll
build on top.**

| ❌ Conflated detector spec | ✅ Crisp dataset request |
|---|---|
| _"Find labeled corpora for detecting over-refusal — inputs are dialogue, outputs are positive vs benign with hard-negatives. We'll train and evaluate a transformer."_ | _"Refusal-labeled corpora for customer-support agents — over-refused benign prompts plus correctly-refused harmful prompts."_ |

What belongs: labels, content shape, domain context (English,
customer-support, agent-mediated). What doesn't: input/output
schemas, model architecture, "we'll train and evaluate." Aim for
**under 250 characters**.

dataset-scout will surface a hint in the report when it detects a
detector-spec pattern. Listen to it.

See [`concepts.md` §9](concepts.md#9-how-to-write-a-brief) for the
fuller treatment.

---

## 3. Iterate cheaply on the brief

Before paying for a full recon, validate the directions:

```bash
datascout decompose "refusal-labeled corpora for customer-support agents — over-refused benign prompts plus correctly-refused harmful prompts" --out scratch/
```

~5 seconds, one LLM call. The model proposes 3–7 search directions
adjacent to your brief and prints them to stdout. Read them.

- If the directions look wrong, refine the brief and re-run.
- If they look right, the saved `scratch/decomposition.yaml` is
  ready to feed into a full recon — skipping the
  decomposition step the next time so you don't re-pay for it.

---

## 4. Run the full recon

```bash
datascout recon "<your refined brief>" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/
```

What runs:

1. **Brief parsing.** The heuristic parser extracts languages,
   threat families, deployment context. No LLM required.
2. **Multi-direction search.** Each direction's keywords become
   their own HF query; results dedupe into one pool with full
   provenance.
3. **Cheap probes.** Six metadata-driven signals per candidate:
   license (with SPDX guess), size, recency, freshness bucket,
   declared languages, card-completeness.
4. **Two-stage shortlist.** Top-k per direction for breadth, then
   global re-rank by multi-direction hits + license sanity + card
   hygiene caps the assessor input at 15–20 candidates.
5. **Row-aware strategy assessment.** **For each shortlisted
   candidate, the assessor fetches 8 real rows** before the LLM call.
   Returns 1–4 ranked strategies from the 7-kind taxonomy with
   actual column names and label values in the transform spec.
6. **Coverage report.** What aspects of your brief no candidate
   covers, and what to do about each.
7. **Re-rank** by best strategy + kind bonus.

Outputs:

```
scratch/recon/
├── report.md            human-readable; LEADS WITH coverage gaps when notable
├── results.json         structured ReconResult
├── decomposition.yaml   stand-alone direction list (hand-editable)
└── recipe.draft.yaml    REAL column names + label values, curate-ready
```

Open `scratch/recon/report.md`. Note which candidates the LLM
tagged `direct_use` versus `signal_proxy` — that's the value-add
over naive HF search. **If you see "Sourcing roadmap" leading the
report, your brief is exploring frontier territory and HF coverage
is sparse — the gaps + decomposition are the actual deliverable.**

---

## 5. Inspect a candidate (optional but useful)

A strategy badge is a hypothesis. Before committing, **inspect** the
top candidate yourself:

```bash
datascout inspect huggingface:bench-llm/or-bench --intent-from scratch/recon/results.json
```

Outputs (to stdout, pipeable):

- Identity, license, card-declared metadata
- Inferred schema from a 50-row sample
- Label distribution with **Wilson 95% CIs** so you don't
  over-interpret 50 rows
- Min / median / max text length
- First five sample rows
- Strategy assessment against the same Intent recon used (`--intent-from`)

```bash
# Deep-dive multiple candidates
datascout inspect huggingface:walledai/XSTest --intent-from scratch/recon/results.json > xstest.md
```

---

## 6. Hand-edit the recipe (lightly — recipes are real now)

Open `scratch/recon/recipe.draft.yaml`. Because the assessor sampled
real rows during recon, the transforms reference **actual column
names**:

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
      Direct fit: prompts paired with refusal labels. Maps cleanly to
      a binary over-refusal classifier.
    transform:
      text_column: prompt              # ← real column name
      label_column: category           # ← real column name
      label_value_map:
        over-refusal: positive         # ← real label values from the data
        ok: benign
      label_kind_map:
        all: ground_truth
      filter: null                     # filter DSL lands in M4b
      take: all
```

Common edits:

- **Drop weak components** — delete entries you don't trust. They
  land in the `declined` list with a reason.
- **Cap `take`** — for proxies, `take: 2000` keeps the corpus
  balanced so a large signal-proxy doesn't drown the direct fits.
- **Null out filters** — until M4b ships the filter DSL, any
  non-null filter causes curate to hard-fail. Edit them to `null`.

```bash
cp scratch/recon/recipe.draft.yaml recipe.yaml
$EDITOR recipe.yaml
```

The recipe is the seam between recon and curate. Anything you put
in here, the corpus reflects.

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
  "extras": { "/* every original column verbatim */" },
  "extras_coercion": false
}
```

Two fields to internalise:

- **`label_kind`** — `ground_truth` rows are safe to train AND eval
  on. `proxy` rows are train-only — *exclude them from your eval
  set*. The field is load-bearing for honest measurement.
- **`extras`** — the original source row is preserved verbatim
  (with multimodal coercion when needed). Nothing is dropped.

---

## 8. The audit trail

Open `over-refusal-corpus/recipe.lock.yaml`. This is the file a
reviewer asks about: which corpus did this detector train on, and
how did proxies factor in?

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

`audit_readiness: preview` is honest. M4b will flip it to `ready`
once MinHash dedup + leakage-aware splitting + the filter DSL land.
Until then, this is a working corpus, not yet a defensible one — and
the lockfile says so.

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

## 10. Multi-detection programs (`compose`)

If your threat model spans several sub-detections — e.g., one
threat-intel report yielded three sub-programs that share a corpus —
run recon for each, then compose:

```bash
# One recon per sub-detection
datascout recon "<brief 1>" --out detection-1/
datascout recon "<brief 2>" --out detection-2/
datascout recon "<brief 3>" --out detection-3/

# Merge the recipes — components dedupe by (source, source_id),
# higher confidence wins on conflict
datascout compose \
    detection-1/recipe.draft.yaml \
    detection-2/recipe.draft.yaml \
    detection-3/recipe.draft.yaml \
    --out programs/merged-recipe.yaml \
    --intent-brief "Combined detection corpus"

# Materialise the merged corpus
datascout curate --from programs/merged-recipe.yaml --out programs/corpus/
```

---

## 11. Honest limits

- **Strategy assessment is LLM-judgment**, not ground truth. Read
  the rationale and inspect samples before committing.
- **Card metadata is uneven.** Licenses are sometimes wrong;
  languages are often missing. We surface what's there and clearly
  mark what isn't.
- **Recency alone means little.** "Uploaded yesterday" doesn't
  imply the data reflects today's threat surface.
- **Reproducibility is contingent.** `recipe.lock.yaml` pins
  revisions and content hashes. If upstreams delete the data, only
  an archive (a future feature) makes the blend reproducible
  standalone.
- **`curate` is currently `audit_readiness: preview`.** Hash-mod
  splits, no MinHash dedup, filter DSL hard-fails. M4b is the next
  upgrade.
- **Not legal advice.** License signals are an SPDX best-effort
  guess; read the upstream card before redistributing.

---

## What's next

- **[CLI reference](cli.md)** — every flag, every verb.
- **[Concepts](concepts.md)** — the strategy taxonomy, label kinds,
  why discovery vs ranking matters, **how to write a brief**.
- **[Configuration](configuration.md)** — env vars, paths,
  programmatic `ScoutContext`.
- **[Architecture](architecture.md)** — pipeline, sources, probes,
  milestone status.

Found a sharp edge? Open an issue. The honest limits list is fed by
real friction; please add to it.
