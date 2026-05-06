# Hero demo — 7 minutes, three beats

A scripted walkthrough for showing dataset-scout to a team for the
first time. Designed to land the value prop without going down
implementation rabbit-holes.

**Total time:** ~7 minutes (5 talking, 2 watching).
**Setup needed beforehand:** `az login`; a `.env` with
`AZURE_OPENAI_ENDPOINT` and `AZURE_OPENAI_DEPLOYMENT`.

---

## The framing (30 seconds)

> *"Picture the last time you needed a labeled dataset for a problem
> that didn't have an obvious public corpus. Three hours of HF
> searches, ten different schemas, and you ended up writing your own
> column-renaming code anyway. dataset-scout is built for exactly
> that loop. It treats dataset triage as a first-class engineering
> problem."*

---

## Beat 1 — the no-setup wow (~30 seconds)

```bash
uvx dataset-scout tour
```

> *"This is the full output shape — decomposition, reframings,
> coverage gaps, candidates ranked by fit — for a real-feeling brief
> ('over-refusal in customer support'). No tokens, no config. The
> point: this is what you get on disk after every recon."*

**Things to point at on screen:**

- The decomposition: *"the LLM expanded the brief into 3 search
  directions before searching anything."*
- One `direct_use` candidate, one `signal_proxy`, one `benign_baseline`:
  *"three different relationships to the same brief — that's the
  strategy taxonomy at work."*
- The "Sourcing roadmap" / coverage-gap section: *"this part —
  what's missing — is the deliverable when your brief is in frontier
  territory and HF doesn't have a clean fit."*

---

## Beat 2 — the real recon (~3 minutes including coffee)

Pick a brief you actually care about. The over-refusal scenario is
fine; a fresher one for the demo:

```bash
datascout decompose "labeled corpora for detecting prompts that try
to extract a model's hidden system prompt — including paraphrases,
roleplay-jailbreak vectors, and benign prompt-engineering questions
as hard negatives" --out scratch/
```

> *"Five seconds, one LLM call. This is the cheap-iter loop —
> tighten the brief until the directions look right, then pay for
> the full recon."*

Point at the directions printed to stdout. They should be
recognizable, non-trivial, and scoped.

```bash
datascout recon "<same brief>" \
    --decomposition-from scratch/decomposition.yaml \
    --out scratch/recon/
```

> *"~2 minutes. While it runs: it's pulling candidates per direction,
> running cheap probes, sampling 8 real rows from each shortlisted
> candidate, then asking the LLM to propose 1–4 reframing strategies
> per candidate with rationale. Coffee."*

When it finishes, open `scratch/recon/report.md`:

- **The lead section** — coverage gaps if any, otherwise the top
  reframings. *"Notice: no aggregate quality score. Per-candidate
  rationale, not a leaderboard."*
- **A reframing rationale** — find a candidate tagged
  `cross_class_repurposing` or `signal_proxy` and read the rationale
  out loud. *"This is the magic — it's not pattern-matching, it's a
  defensible reframing of related work."*
- **`recipe.draft.yaml`** — open it. *"Look — real column names. Real
  label values. The assessor sampled actual rows, so the transform
  spec is curate-ready."*

---

## Beat 3 — the corpus on disk (~2 minutes)

```bash
datascout curate --from scratch/recon/recipe.draft.yaml \
    --out ./demo-corpus
```

While it runs:

> *"4 components materialising in parallel. MinHash + LSH near-dup
> detection clusters similar rows, then leakage-aware splitting routes
> whole clusters to the same side of the train/eval boundary. Every
> parameter goes into the lockfile."*

When it finishes:

```bash
ls demo-corpus/
cat demo-corpus/usage.md         # 3-line snippets for HF / pandas / raw
head -1 demo-corpus/train.jsonl  # a normalized row
```

Point at:

- **`label_kind`** in the row: *"`ground_truth` rows train and eval
  on. `proxy` rows train only — exclude from eval. The field is
  load-bearing."*
- **`extras`**: *"every original column preserved verbatim, nothing
  dropped."*
- **`recipe.lock.yaml`**: *"this is the file a reviewer asks about.
  Pinned revisions, dedup parameters, every kept and skipped
  component with rationale."*

---

## Closer (30 seconds)

> *"The takeaway: dataset triage isn't 'spend an afternoon on
> HuggingFace and hope.' It's a tractable engineering problem with
> reframings as the core abstraction. We get from a fuzzy brief to a
> defensible corpus in under ten minutes — with the audit trail to
> show the reviewer."*

**If asked "what about X downstream?":**
> *"Out of scope on purpose. Scout writes JSONL with full provenance.
> Anything that consumes JSONL — a training pipeline, a workbench, an
> eval orchestrator — wires up in a small adapter."*

**If asked "is this AI-security-only?":**
> *"AI-security is the lead persona because reframings matter most
> there. But the loop is general: any time you need a labeled corpus
> from public data and the off-the-shelf datasets are the wrong shape,
> scout helps."*

---

## Backup material

If you have extra time or someone digs in:

- **`datascout inspect huggingface:bench-llm/or-bench --intent-from
  scratch/recon/results.json`** — single-candidate deep-dive with
  Wilson-CI label distribution. Pipes cleanly to a markdown file.
- **`datascout judge ./demo-corpus --axis <axis> --calibrate-against
  ./gold`** — opt-in label rescue under audit, with a calibration
  pass that reports P/R/F1 against gold *before* the full run and
  optionally aborts a too-low precision pass.
- **The "honest limits" section** of the README — read it out loud.
  *"The product is built by someone who actually has to defend this
  to a reviewer."*

---

## What NOT to demo (yet)

- **`compose`** — niche; only useful if your audience explicitly
  has a multi-detection program. Mention as available.
- **The full M10 judge / eval flow** — it's powerful but it's *not
  the wow*. Show only if asked.
- **`cache` sub-commands** — available but not interesting for a
  demo. Mention as available if asked.
- **The 7-kind strategy enum** — say "direct fit / reframing /
  proxy" three times and stop. The wire format has more nuance; the
  pitch doesn't need it.
