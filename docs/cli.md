# CLI reference

Two equivalent entry points. Pick whichever you prefer:

```bash
dataset-scout ...      # formal
datascout      ...     # recommended short form
```

All commands route to the same library functions. The CLI is a thin
wrapper — anything you can do from the CLI is also exposed in
`dataset_scout.recon()` / `inspect()` / `curate()` (see the library
section of [architecture.md](architecture.md)).

## Verb cheat-sheet

| Verb | What it does | Cost |
|---|---|---|
| [`tour`](#tour) | Render a fully-populated demo report from canned data | none — no AOAI/HF |
| [`decompose`](#decompose) | LLM brief decomposition only (cheap iteration loop) | ~5s, 1 LLM call |
| [`recon`](#recon) | Full pipeline → `report.md` + `results.json` + `recipe.draft.yaml` + `decomposition.yaml` | ~2 min, ~16 LLM calls |
| [`inspect`](#inspect) | One-candidate deep-dive | ~5s + optional 1 LLM call |
| [`curate`](#curate) | Recipe → JSONL + lockfile | depends on row count |
| [`compose`](#compose) | Merge multiple recipes into one | none — local |
| [`cache`](#cache) | Inspect / prune / clear the cache | M1b — not yet implemented |
| [`sources`](#sources) | List / toggle source plugins | local |

---

## `tour`

A 30-second demo with **no external services**: no HuggingFace token,
no Azure OpenAI configuration. The command renders a fully-populated
recon report from canned data so a new user can see the value
immediately.

```bash
datascout tour                # → stdout
datascout tour --out scratch  # → also persist artefacts
```

| Option | Description |
|---|---|
| `--out PATH` | Persist the demo's `results.json`, `recipe.draft.yaml`, `decomposition.yaml`, `report.md`. |

Use this in PRs, slack messages, or first-time onboarding.

---

## `decompose`

The cheap brief-iteration verb. Just runs the LLM decomposition step
— ~5 seconds and one LLM call. Prints the proposed directions to
stdout; `--out` optionally writes them to `decomposition.yaml`.

```bash
datascout decompose "<brief>" [options]
```

| Option | Description |
|---|---|
| `--detection-target TEXT` | Override the parsed detection target. |
| `--deployment-context TEXT` | Production / RAG / agent / etc. |
| `--out PATH` | Write `decomposition.yaml` to this directory; otherwise stdout-only. |

### The cheap-iteration loop

```bash
# Try it
datascout decompose "<v1 brief>"
# Don't like the directions? Refine and retry.
datascout decompose "<v2 brief>"
# Once happy, persist
datascout decompose "<v2 brief>" --out scratch/
# Run the full recon, skipping the (now-paid-for) decompose step
datascout recon "<v2 brief>" --decomposition-from scratch/decomposition.yaml --out scratch/recon/
```

You'd otherwise burn ~2 minutes per iteration paying for full recons
just to see the directions.

---

## `recon`

Find candidate datasets, run probes, assess strategies, identify
coverage gaps, emit a draft recipe.

```bash
datascout recon "<brief>" [options]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `BRIEF` | yes | Natural-language brief describing the **dataset you want** (not the detector you'll build — see `concepts.md` §9). |

### Options

| Option | Default | Description |
|---|---|---|
| `--detection-target TEXT` | parser-derived | Override the parsed detection target. |
| `--deployment-context TEXT` | parser-derived | Production / RAG / agent / etc. |
| `--language TEXT` | `en` | Required language(s); repeat for multiple. |
| `--license TEXT` | permissive set | License allowlist; repeat for multiple. |
| `--min-strategy-confidence FLOAT` | `0.5` | Strategies below this confidence are filtered out of the draft recipe. Recipe-authoritative — `curate` defaults to the value baked into the recipe. |
| `--decomposition-from PATH` | — | Reuse a hand-edited `decomposition.yaml` instead of paying for a fresh LLM decompose call. |
| `--out PATH` | `datascout-out/` | Output directory. |

### Outputs

```
<out>/
├── report.md            human-readable, LEADS WITH coverage gaps when notable
├── results.json         structured ReconResult (Pydantic dump)
├── decomposition.yaml   stand-alone direction list (hand-editable)
└── recipe.draft.yaml    hand-editable input for `datascout curate` —
                          contains REAL column names from row sampling
                          when the LLM strategy assessor runs
```

### Examples

```bash
# Most common — natural-language brief
datascout recon "HTML and Markdown corpora with labelled hidden text — phishing pages and accessibility benigns"

# Multi-language
datascout recon "refusal-labeled customer-support corpora" --language en --language es

# Reuse a prior decomposition (cheap iteration)
datascout recon "<refined brief>" --decomposition-from scratch/decomposition.yaml

# Force a license restriction
datascout recon "..." --license MIT --license Apache-2.0
```

### Hidden expert flags

These exist for debugging and forward compatibility but are
deliberately **not in `--help`**, because they tend to mask weaknesses
the brief should express:

- `--threat-families` — comma-separated families to inject into the
  parsed Intent. Use only when the brief is genuinely ambiguous.
- `--no-explore` — skip LLM decomposition. Useful for reproducing
  metadata-only behavior when AOAI is configured.

---

## `inspect`

One-candidate deep-dive. Renders to **stdout** so it pipes cleanly:
`datascout inspect huggingface:org/x > inspect.md`.

```bash
datascout inspect <source>:<id>[@revision] [options]
```

| Option | Default | Description |
|---|---|---|
| `--intent-from PATH` | — | Re-use the most recent recon's Intent (point at `results.json`). The strategy assessor sees the same Intent recon used. |
| `--brief TEXT` | — | One-off brief for strategy assessment when no recon `results.json` exists. Mutually exclusive with `--intent-from`. |
| `--sample-size INT` | `50` | Rows to stream for the schema / label-distribution / sample sections. |

### What it shows

1. **Identity** — `source:id`, revision, card URL, description, gating posture.
2. **License** — raw card declaration plus a best-effort SPDX guess.
3. **Card-declared metadata** — languages, task categories, tags, dates, popularity.
4. **Inferred schema** — column names + types from the streamed sample.
5. **Label distribution** — count and Wilson 95% CI per value when a label column is detected.
6. **Text-length stats** — min / median / max characters for the heuristically-picked text column.
7. **Sample rows** — first five.
8. **Strategy assessment** — when `--intent-from` or `--brief` is supplied AND AOAI is configured.

```bash
# Quick metadata-only deep-dive
datascout inspect huggingface:bench-llm/or-bench

# With a fresh brief — adds LLM strategy assessment if AOAI is set
datascout inspect huggingface:bench-llm/or-bench --brief "refusal-labeled corpora for support agents"

# Re-use the Intent from your latest recon
datascout inspect huggingface:walledai/XSTest --intent-from datascout-out/results.json
```

---

## `curate`

Build a schema-normalized corpus from a recipe.

```bash
datascout curate --from recipe.yaml --out ./mycorpus
```

| Option | Default | Description |
|---|---|---|
| `--from PATH` | required | Path to `recipe.yaml`. |
| `--out PATH` | `mycorpus/` | Output directory. |
| `--min-strategy-confidence FLOAT` | recipe-defined | Override the recipe's threshold. Recorded as an override in `recipe.lock.yaml`. |
| `--seed INT` | recipe-defined | Override the recipe's split seed. Recorded as an override. |

### Status: audit-ready

The pipeline ships a defensible end-to-end build:

- ✅ Recipe loader, materialisation from HuggingFace, normalized
  JSONL output, lockfile, manifest, report, fingerprint, usage.
- ✅ Multimodal-safe `extras` (bytes/arrays coerced to strings with
  a flag).
- ✅ **Filter DSL** — `transform.filter` accepts a sandboxed
  expression (booleans, comparisons, `len`, `contains_pattern`,
  `lower`, `startswith`, `endswith`, `int`, `str`, `in`).
  Disallowed nodes (attribute access, comprehensions, lambdas,
  unknown functions) are rejected at compile time.
- ✅ **MinHash dedup + leakage-aware split.** Near-duplicate rows
  (Jaccard ≥ 0.8 over char 5-grams) cluster together; whole
  clusters are assigned to a single split. Cluster stats are
  written to the lockfile.
- ✅ **Resilient to per-component upstream errors.** Gated
  datasets, multi-config datasets without a `source_config`,
  non-standard split names, parse errors — each is classified
  (categories: `gated_dataset`, `missing_config`, `bad_split`,
  `no_data_files`, `parse_error`, `not_found`, `network`,
  `unknown`) and recorded under `failed_components` in the
  lockfile with an actionable hint. The corpus continues to build
  from what does succeed; the run only fails hard if every
  component errors out.

`recipe.lock.yaml` records `audit_readiness: ready` with the
MinHash params (`num_perm`, `threshold`, `shingle_size`,
`dedup_version`) + filter expressions in scope, and a
`failed_components` block when applicable. `report.md` carries
an "audit-ready" banner with cluster stats and a "Components
skipped due to upstream errors" section so the corpus's
provenance is visible at a glance.

---

## `compose`

Merge multiple recipes into one — for multi-detection programs that
share a corpus.

```bash
datascout compose recipe-a.yaml recipe-b.yaml [recipe-c.yaml ...] --out merged.yaml
```

| Option | Description |
|---|---|
| `--out PATH` | Required. Path to write the merged recipe.yaml. |
| `--intent-brief TEXT` | Override the merged intent's brief (default: keep the first input's). |

### Semantics

- Components dedupe by `(source, source_id)`. Higher
  `strategy_confidence` wins on conflict; the loser becomes a notice.
- `min_strategy_confidence` takes the **maximum** of the inputs (most
  conservative wins).
- Declined entries are unioned and deduped.
- Intent comes from the first input unless `--intent-brief` overrides.
  When inputs have different briefs, a notice is recorded.

```bash
# Three sub-detection programs into one corpus
datascout compose \
    detection-1/recipe.draft.yaml \
    detection-2/recipe.draft.yaml \
    detection-3/recipe.draft.yaml \
    --out programs/merged.yaml \
    --intent-brief "Content Injection Traps — combined corpus"

# Then materialise
datascout curate --from programs/merged.yaml --out programs/corpus/
```

---

## `cache` *(M1b — not yet implemented)*

Inspect and manage the SQLite cache.

```bash
datascout cache info
datascout cache prune
datascout cache clear
```

---

## `sources`

List and toggle source plugins.

```bash
datascout sources list             # works today: reflects current ScoutContext
datascout sources enable <name>    # M1b: writes to ~/.config/dataset-scout/config.toml
datascout sources disable <name>   # M1b
```

---

## Global options

| Option | Description |
|---|---|
| `--version` | Print version and exit. |
| `--help` | Standard Typer help. |

---

## Output streams and exit codes

- **`stdout`** — Markdown reports (when piped) and `inspect` deep-dives. Otherwise empty.
- **`stderr`** — Progress events, notices, errors, completion summary.
- **Exit codes** — `0` success · `1` runtime error · `2` not-yet-implemented or invalid usage.

---

## See also

- [Concepts](concepts.md) — how to write a brief, what the report's framing language means.
- [Configuration](configuration.md) — `.env`, Azure OpenAI, HF.
- [Architecture](architecture.md) — what runs under each verb.
