# Concepts

The mental model behind `dataset-scout`. Read this once and the CLI
output and library types should feel obvious.

---

## 1. Discovery, not ranking

`datascout recon` is **discovery tooling**, not a ranked-quality
scoreboard. Candidates come back in the source's own search-relevance
order; probe outputs are **annotations** beside each candidate, never
folded into a single "quality" number.

The reason is honesty. License + freshness + card-completeness say a
lot about a dataset's polish but very little about whether it actually
fits your detection task. Without a semantic-fit signal (embedding fit
+ LLM strategy assessment, both later milestones), aggregating those
into a fitness score would imply confidence we don't have.

When the LLM strategy assessor runs (full mode) the framing changes:
candidates carry per-strategy confidence, the report leads with the
strongest defensible reframings, and `recipe.draft.yaml` lands in the
output directory. In metadata-only mode it stays pure discovery, with
receipts.

---

## 2. Modes: full vs metadata-only

The pipeline picks one of two modes at start-up.

| Mode | Trigger | What runs |
|---|---|---|
| **Full** | Azure OpenAI configured (`AZURE_OPENAI_ENDPOINT` + `_DEPLOYMENT`) | Brief parsing → LLM decomposition → multi-direction HF search → cheap probes → two-stage shortlist → per-candidate strategy assessor → coverage gaps → ranked report + `recipe.draft.yaml` |
| **Metadata-only** | AOAI not configured | Brief parsing → single-query HF search → cheap probes (no decomposition, no strategies, no recipe draft) |

The fallback is **explicit and noisy** — you'll see a notice on stderr
and a prominent header in `report.md` telling you what to set. No
silent weakening.

See [`configuration.md`](configuration.md) for the env-var setup.

---

## 3. Sources, candidates, and metadata

A `Source` plugin (HuggingFace today; Kaggle and Papers-With-Code in
M1b) yields `Candidate` objects in response to an `Intent` plus a list
of `DecompositionDirection`s. Each candidate carries:

- **Identity** — `source`, `id`, `revision`.
- **`metadata`** — a typed `CandidateMetadata` envelope: license raw +
  SPDX guess, dates, declared languages, size signals, columns
  (when cheap), card-fields-present set, gating posture, tags.
- **`surfaced_by: list[str]`** — the names of every decomposition
  direction that found this candidate. A candidate hit from the
  original brief carries an empty list. Multi-direction hits keep
  all their provenance, which lets future ranking weight them
  appropriately.

The metadata envelope is **source-agnostic** — Kaggle and PWC plugins
will populate the same shape, so probes never read source-specific keys.

---

## 4. Probes

Each probe is a small object emitting a typed `SubScore` per candidate:

```python
class Probe(Protocol):
    name: str
    version: str
    def applies(self, candidate, intent) -> bool: ...
    def run(self, candidate, intent) -> SubScore: ...
```

Six metadata-driven probes ship today:

| Probe | What it tells you |
|---|---|
| `license` | SPDX best-effort guess and how it sits relative to your `LicensePolicy` (allow / warn / outside policy / unknown). |
| `size` | Row count / byte count / downloads / likes when declared. No 0–1 score; size doesn't map cleanly. |
| `recency` | Days since upload and last-modified. Raw evidence only. |
| `freshness` | Bucketed signal: fresh `<6mo` / current `6–18mo` / aging `>18mo`. |
| `languages` | Overlap fraction between declared languages and your intent's language list. |
| `card_completeness` | Fraction of expected YAML fields actually declared. Documentation-hygiene only — never used as a major rank component. |

**Sample-driven probes** — `label_structure`, `schema_fingerprint`,
plus the **embedding label-intent fit** — land in M1b once `Source`
plugins implement `stream_sample()`.

Every `SubScore` carries:

- a `value` (when a 0-1 signal makes sense, otherwise `None`),
- an `n` (rows sampled / count when applicable),
- a `status` (`ok` / `not_applicable` / `low_confidence` / `skipped`),
- a list of `Evidence` items — every claim links to specific facts.

The status field is load-bearing: probes that genuinely don't apply to
a candidate self-report `not_applicable` instead of fabricating a value.

---

## 5. The strategy taxonomy

Per-candidate, the LLM strategy assessor returns 1–4 ranked `Strategy`
objects from this 7-kind taxonomy:

| Kind | Meaning |
|---|---|
| `direct_use` | Labels and content map cleanly to your task. |
| `subset_extraction` | Only some rows are relevant; filter to subset. |
| `label_remapping` | Same data, different label semantics. |
| `cross_class_repurposing` | Original positives → hard-negatives, or vice versa. |
| `signal_proxy` | Adjacent threat used as proxy positive during cold start. |
| `benign_baseline` | No relevant positives, but useful as benign distribution. |
| `not_useful` | The honest answer when nothing fits. |

(An eighth kind — `composition_only` — is reserved in the enum for a
future portfolio-level pass, when the assessor evaluates pairs/triples
rather than single candidates. Per-candidate assessments today never
emit it.)

Each strategy carries a `confidence ∈ [0, 1]`, a written rationale,
caveats, and a concrete `transform` spec (column maps, label
value-maps, filters). The assessor is **conservative-but-creative**:
stretches get low confidence, and `--min-strategy-confidence` filters
aggressive reframings out of the draft recipe.

---

## 6. `label_kind` — proxies are honest by default

The output JSONL (M4) marks every row with one of:

| `label_kind` | When to train | When to eval |
|---|---|---|
| `ground_truth` | ✅ | ✅ |
| `subset_extracted` | ✅ | ✅ (within the documented subset) |
| `remapped` | ✅ | ✅ *if you accept the remapped definition* |
| `proxy` | ✅ (often weighted) | ❌ — exclude proxies from eval |

This is the load-bearing field for downstream training. Downstream
tools should filter to `label_kind != "proxy"` for evaluation.

---

## 7. Recipes and lockfiles

The output of `recon` includes a `recipe.draft.yaml`. Hand-edit it
(adjust strategies, drop candidates, add explicit components), then
hand it to `datascout curate --from recipe.yaml` *(M4 — coming next)*.

`curate` will materialise the recipe into:

```
mycorpus/
├── train.jsonl / val.jsonl / test.jsonl   leakage-aware splits
├── recipe.yaml                             input, copied verbatim (audit trail)
├── recipe.lock.yaml                        pinned revisions + realized counts + hashes
├── manifest.json                           machine-readable lock equivalent
├── report.md                               5-second scorecard + provenance
├── fingerprint.txt                         one-line content hash for commits
└── usage.md                                3-line snippets for HF datasets / pandas
```

`recipe.lock.yaml` is **the defensible record** — the single file a
reviewer can ask about: which corpus did this detector train on, and
how did proxies factor in?

---

## 8. Voice

The audience is detection engineers under audit pressure. Output is
designed accordingly:

- **No aggregate "quality" or "risk" headline.** Per-signal evidence,
  per-candidate strategy assessment, written rationale.
- **Receipts everywhere.** Every claim links back to a card URL,
  sample row, prompt, or response.
- **`recipe.lock.yaml` is the defensible record.**
- **Proxies are honest by default.** Output JSONL marks them; eval
  must exclude them.
- **"This is not legal advice"** footer on any report touching
  licensing.


## 9. How to write a brief

The brief is the only input that drives everything downstream — the
heuristic parser, the LLM decomposer, the multi-direction search, the
strategy assessor. **A good brief is a crisp dataset request, not a
detector design.** Two specs jammed together (the dataset I want AND
the detector I'll build) confuses every step.

### Bad → good

| ❌ Conflated detector spec | ✅ Crisp dataset request |
|---|---|
| _"Find labeled corpora for detecting X — inputs are HTML, outputs are positive vs benign with hard-negatives for Y. We'll train and evaluate a transformer..."_ | _"HTML and Markdown corpora with labelled hidden text — phishing pages, indirect prompt-injection payloads, accessibility-style benigns."_ |
| _"Find datasets to train a classifier for over-refusal where the model declines benign requests citing safety. Output schema: positive vs benign vs hard-negative."_ | _"Refusal-labeled corpora for customer-support agents — over-refused benign prompts plus correctly-refused harmful prompts."_ |

### What belongs in a brief

- **Labels** you want (positive / benign / hard-negative).
- **Content shape** (HTML, dialogue, code, image, multi-turn, …).
- **Domain context** that affects matching (English, customer-support,
  agent-mediated, code-switched, …).

### What does NOT belong

- Input/output schemas of your downstream model.
- "We'll train and evaluate" / "the classifier should…"
- Architecture choices (transformer, LLM-judge, etc.).
- Long lists of every sub-case — the LLM decomposer's job is to
  enumerate adjacencies. Don't pre-empt it.

### Length

Aim for **under 250 characters**. The HF lexical search and the LLM
decomposer both work better on crisp briefs. If you're at 400+
characters, you've probably described the detector instead of the
dataset.

### Iterate cheaply

Run `datascout decompose "<brief>"` first — ~5 seconds, one LLM call.
It prints the directions the model would explore. If the directions
are wrong, refine the brief and re-run. Once the directions look
right, run full `datascout recon` (and pass `--decomposition-from
decomposition.yaml` to skip re-paying for decomposition).

```bash
# Cheap iteration loop
datascout decompose "your brief" --out scratch/

# When happy, full recon reusing the directions
datascout recon "your brief" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/
```

### When the parser nudges you

`recon` will surface a hint when the brief looks like a detector
spec ("describes detector inputs", "describes detector outputs",
"describes the train/eval plan"). When you see it, tighten the brief.

See [`architecture.md`](architecture.md) for the pipeline detail.
