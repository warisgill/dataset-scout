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

---

## `recon`

Find candidate datasets and emit a discovery report.

```bash
datascout recon "<brief>" [options]
```

### Arguments

| Argument | Required | Description |
|---|---|---|
| `BRIEF` | yes | Natural-language description of what you're looking for. The brief carries the signal; flags are nudges. |

### Options

| Option | Default | Description |
|---|---|---|
| `--detection-target TEXT` | parser-derived | Override the detection target inferred from the brief. |
| `--deployment-context TEXT` | parser-derived | Production / RAG / agent / etc. |
| `--language TEXT` | `en` | Required language(s); repeat for multiple. |
| `--license TEXT` | permissive set | License allowlist; repeat for multiple. |
| `--min-strategy-confidence FLOAT` | `0.5` | Strategies below this confidence are filtered out of the draft recipe. Recipe-authoritative — `curate` defaults to the value baked into the recipe. |
| `--out PATH` | `datascout-out/` | Output directory. |

### Outputs

```
<out>/
├── report.md          discovery + decomposition + per-candidate strategies + coverage gaps
├── results.json       structured ReconResult (Pydantic dump)
└── recipe.draft.yaml  hand-editable input for `datascout curate` (M4)
                       — emitted only when LLM strategies were assessed
```

### Examples

```bash
# Most common
datascout recon "find labeled prompt injection corpora for our RAG service"

# Force a license restriction
datascout recon "jailbreak detector training data" --license MIT --license Apache-2.0

# Multi-language
datascout recon "unsafe output detection" --language en --language ja

# Steer detection target explicitly
datascout recon "data for LLM safety filter" \
    --detection-target "harmful response classification" \
    --deployment-context "consumer chatbot"
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

## `inspect` *(M3 — not yet implemented)*

Deep-dive on one candidate. Will print a single candidate's full
metadata envelope, label distribution (with Wilson 95% CIs), schema,
sample rows, license summary, and full strategy assessment against
the most recent intent.

```bash
datascout inspect <source>:<id>[@revision]
datascout inspect huggingface:deepset/prompt-injections@4f61ecb038e9
```

| Option | Description |
|---|---|
| `--intent-from PATH` | Re-use the most recent recon's Intent (point at `results.json`). |

---

## `curate` *(M4 — not yet implemented)*

Build a schema-normalized, leakage-aware corpus from a recipe.

```bash
datascout curate --from recipe.yaml --out ./mycorpus
```

| Option | Default | Description |
|---|---|---|
| `--from PATH` | required | Path to `recipe.yaml`. |
| `--out PATH` | `mycorpus/` | Output directory. |
| `--min-strategy-confidence FLOAT` | recipe-defined | Override the recipe's confidence threshold. |

Output layout described in
[concepts.md](concepts.md#7-recipes-and-lockfiles-m4--preview).

---

## `cache` *(M1b — not yet implemented)*

Inspect and manage the SQLite cache.

```bash
datascout cache info       # size, hit counts, oldest / newest entries
datascout cache prune      # evict to a target size
datascout cache clear      # wipe everything
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

- [Concepts](concepts.md) — what the report's framing language means.
- [Configuration](configuration.md) — `.env`, Azure OpenAI, HF.
- [Architecture](architecture.md) — what runs under each verb.
