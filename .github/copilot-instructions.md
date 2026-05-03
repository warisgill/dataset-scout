# Session handoff — dataset-scout

> Pick this up in a new session to start implementation. Everything
> the next agent (or future-you) needs to know is here. The
> authoritative specs are in the repo; this file is the meta-context
> that explains what was decided, what was rejected, and why.

---

## 1. Where things are

**Repo:** `C:\Users\mdressman\dev\dataset-scout`

```
dataset-scout/
├── plan.md              vision, scope, scenarios, milestones (16 KB)
├── TECH_DESIGN.md       buildable v1 spec — start coding from here (42 KB)
├── ROADMAP.md           explicitly deferred work (eval, etc.)
└── examples/
    └── scenario-malware.md   dual-use stress test reference
```

There is also an empty husk `C:\Users\mdressman\dev\dataset-curation-agent`
that couldn't be removed because the agent process held it as a cwd.
Safe to `rmdir` after the next CLI restart.

**Naming locked in:**

| Layer | Name |
|---|---|
| Repo / directory | `dataset-scout` |
| PyPI distribution | `dataset-scout` (verified available May 2026) |
| Python import | `dataset_scout` (forced; hyphens not allowed) |
| Primary CLI | `dataset-scout` |
| Convenience alias CLI | `dscout` |
| Cache dir | `~/.cache/dataset-scout/` |
| Config dir | `~/.config/dataset-scout/` |
| Default output dir | `./dscout-out/` |
| pip extras | `pip install dataset-scout[langid]` |
| Entry-points group | `dataset_scout.sources` |

GitHub: `<your-account>/dataset-scout` is free under your namespace.
The `dscout` org name is taken globally (a UX-research company) but
that doesn't affect repo names.

---

## 2. Current state

**Specs are complete. Implementation has not started.**

No `pyproject.toml`, no `src/`, no tests. Next concrete step is M0.

---

## 3. The audience

Detection engineers, forensic analysts, incident responders. Not
generic ML researchers. The voice and design choices are tuned for
people who write prompt-injection / jailbreak / malware /
unsafe-output detectors and have to defend their dataset choices to
audit-conscious reviewers.

The user has a sibling project `protozoa-gym` (a detection-authoring
workbench at `C:\Users\mdressman\dev\protozoa-gym`) that's a
representative downstream consumer — it does its own labeling,
schema transformation, and inspection. **dataset-scout slots in
upstream of that, never duplicating its functionality.** Don't
couple the two; the relationship is "dataset-scout produces
normalized JSONL; protozoa-gym (or any consumer) takes it from
there."

---

## 4. What v1 ships (the wedge)

Three CLI verbs:

- **`recon`** — brief → ranked shortlist + draft recipe. The unified
  pipeline always decomposes the brief into 3–7 related search
  directions, searches across all of them on HF + Kaggle + PWC,
  scores with cheap probes + embedding label-intent fit, then runs
  an LLM strategy assessor on the top-15–20 candidates that returns
  ranked reframing strategies per candidate.
- **`inspect`** — deep-dive on one candidate.
- **`curate`** — recipe → schema-normalized JSONL train/val/test +
  audit-ready lockfile + manifest + report + fingerprint + usage
  snippet.

The killer ideas:

1. **Always explore.** Even when direct fits exist, decomposing the
   brief and surfacing reframings produces richer corpora.
2. **Strategy taxonomy.** 8 strategy kinds (`direct_use`,
   `subset_extraction`, `label_remapping`, `cross_class_repurposing`,
   `signal_proxy`, `benign_baseline`, `composition_only`,
   `not_useful`). Per-candidate, the LLM picks the strongest few and
   marks confidence + caveats + transform spec.
3. **`label_kind` per row in output JSONL** (`ground_truth`,
   `subset_extracted`, `remapped`, `proxy`). Downstream knows what
   to train on vs eval on; proxies are honest by default.
4. **Composition pairs and coverage gaps.** The assessor flags
   candidates that only pair with another (`composes_with: [id]`)
   and produces a semantic coverage-gap report describing what
   nothing addresses.
5. **`--min-strategy-confidence` is the diversity dial.** Flows
   end-to-end: `recon`'s value filters the draft recipe; `curate`
   honors the same default. Aggressive reframings filter out at low
   thresholds, in at high.

---

## 5. Decisions log (what's IN, what's OUT, why)

### IN for v1

- HuggingFace + Kaggle + PWC as sources, behind a `Source` plugin
  interface.
- Heuristic brief parser by default; optional litellm fallback
  (`--use-llm-parser`).
- Cheap probes: license, size, recency, languages, label structure
  (with class balance + Wilson 95% CIs), schema fingerprint, card
  completeness, freshness.
- Embedding label-intent fit (sentence-transformers MiniLM) as
  upfront ranker.
- LLM strategy assessor + decomposition (litellm; user-supplied
  API key).
- Recipes carry `composes_with` references; `curate` validates
  them.
- Schema-normalized JSONL with full `extras` passthrough — **no
  source data ever dropped**.
- Cache (SQLite WAL) keyed by `(revision, probe_version,
  intent.stable_hash())`.
- Three CI tiers (`pytest -m unit`, `-m recorded`, `-m live`),
  with `respx` snapshot freshness checks.
- Latency + API-call budgets tracked in `pytest -m perf`.

### OUT of v1 (in `ROADMAP.md`)

- **Evaluation framework.** Frozen benchmarks, recall@K,
  calibration tracking, head-to-head head-to-head comparisons.
  Deferred until we have real-use results to evaluate against.
- **Cost tracking** ($ per run). Removed because we have no
  visibility into the user's account / model pricing.
- **Embedding-space coverage analysis.** When we add it, simpler
  framing: embed brief + decomposition into composite intent
  cloud, measure how much candidates cover. No production data
  assumption.
- **`--existing-corpus` overlap probe.** Deferred to v1.5.
- **`--archive` flag.** Placeholder accepted but no-op in v1.
- arXiv / Zenodo / GitHub-hosted releases.
- PII / toxicity / embedding-diversity / label-noise probes.
- LLM-planned multi-step chains; deterministic-template
  repurposing.
- MCP server / Copilot CLI / Claude Code adapters.
- Croissant / Parquet provenance.
- Modality plugins beyond text.
- Synthetic-data generation. (We *flag* candidates that could
  serve as synthesis seeds; we don't generate.)
- Logprob-based contamination probes.
- `scout eval --quickeval` and `scout why <id>` commands.

### Things tried during this session and rejected

These came up earlier and were explicitly rolled back. Don't
re-introduce without a real reason:

- **Judge-based labeling/enrichment.** Tried as a v1 feature.
  Rejected: scope creep; downstream tools (protozoa-gym, etc.)
  already do labeling; we're upstream of that.
- **Adequacy check / two-phase pipeline.** Tried gating exploration
  on whether direct fits existed. Rejected: always-explore is
  simpler and produces richer corpora regardless.
- **Modes (`llm_eval` / `classifier_train` / etc.).** Tried as a
  user-declared mode that drives detector selection. Rejected:
  audience is firmly security/IR; one shape covers all three
  scenarios in §0.
- **`recipe_version` field.** Versioning rides on the package
  version. Adding a recipe-schema knob now is overengineering.
- **Embedding-space coverage with `--production-sample`.**
  Rejected: we won't realistically have production data access.
  When we add embedding coverage, we measure against the
  decomposition's intent cloud (see ROADMAP).
- **MCP / agent-runtime first-class support.** Was earlier
  positioning. Rejected: positioned as an upstream CLI tool, not
  an "agent." MCP is a future surface.
- **Croissant export, Parquet provenance, `--archive` in v1.**
  Trimmed for simplicity. Add when real users ask.

---

## 6. Stack and tooling conventions

Modern Python, **uv everywhere**:

- Python 3.11+
- **uv** for env / install / lock / run. No `pip install` in docs.
- **hatchling** for build backend.
- **ruff** for both lint and format.
- **mypy** strict for type-checking; pre-commit + CI.
- **pydantic v2** for typed structured outputs and JSON Schema.
- **typer** for CLI.
- **rich** for terminal rendering (tables, progress, panels) — most
  of the "magical UX" wins live here.
- **structlog** for structured logging; JSON to stderr at INFO,
  rich formatter when stderr is a TTY.
- **pytest** with `-m unit / -m recorded / -m live / -m perf`
  markers.
- **`uvx`** in user-facing docs for one-off invocations.

`pyproject.toml` will have `[project]`, `[project.scripts]` (with
both `dataset-scout` and `dscout` entries), `[tool.uv]`,
`[tool.ruff]`, `[tool.mypy]`, `[tool.pytest.ini_options]`,
`[tool.hatch.build.targets.wheel]`.

Dependency footprint commitments:

- Heavy: `torch + sentence-transformers + MiniLM` (~280MB cached)
  — earned by the label-intent fit probe.
- No torch in any test marker that runs without the embedding probe.
- No `transformers`, no `sklearn`, no `MCP` SDK in v1 base install.

---

## 7. Library API design (future-aware)

Even though v1 ships only a CLI, the library is designed so a future
HTTP API / web UX is a small wrapper. Key choices:

- **Library is the source of truth**; CLI is a thin wrapper. All
  business logic in `dataset_scout.recon(...)`,
  `dataset_scout.curate(...)`, etc.
- **No global state.** Config/auth flows through an explicit
  `ScoutContext` object (API keys, cache path, source enablement).
  CLI populates it from env+config.toml; an API server populates
  it from request context.
- **Pipeline emits events.** Long-running ops yield
  `ProgressEvent`s alongside the final result, so CLI renders to
  rich progress bars and a future SSE/WebSocket forwards JSON.
  Same code path.
- **All output types are Pydantic v2.** Free JSON serialization,
  free OpenAPI schema generation, free deserialization.
- **No ANSI/rich codes leak into core data.** Rendering is its own
  layer (`dataset_scout.render`).

---

## 8. Where to start (M0)

```bash
cd C:\Users\mdressman\dev\dataset-scout
uv init --name dataset-scout
# Add to pyproject.toml:
#   [project.scripts]
#     dataset-scout = "dataset_scout.cli:app"
#     dscout         = "dataset_scout.cli:app"
# Set up:
#   src/dataset_scout/{__init__,core,cli,context}.py
#   tests/{__init__,test_core_types.py}
#   .github/workflows/ci.yml
#   .pre-commit-config.yaml
# Tooling:
#   uv add pydantic typer rich structlog pyyaml tomli
#   uv add --dev pytest pytest-cov ruff mypy respx
```

M0 is just scaffolding: empty CLI command stubs, the core dataclasses
from `TECH_DESIGN.md` §3, fixture data files, and a green
`pytest -m unit` (probably just schema-validation tests on the core
types). No real source plugins, no LLM calls, no real probes.

M1 lights up the HF source plugin and the cheap probes; M2 brings
the LLM. Detailed milestones are in `plan.md` §8 and `TECH_DESIGN.md`
§18.

---

## 9. Things to think about during M0

These are open at the implementation-detail level; they don't change
the design but they need a tasteful answer when wiring up the code:

- **Filter expression DSL** (`transform.filter` field in recipes —
  e.g., `column == value`, `contains_pattern(name)`, `len(text) > N`).
  Likely candidate: `simpleeval` or a tiny hand-rolled visitor.
  Decide when wiring up the first real recipe.
- **Logging defaults.** `structlog` JSON to stderr at INFO is the
  obvious default; verify rich-formatter swap-on-TTY works cleanly.
- **`ScoutContext` shape.** Define explicitly in M0 even if only the
  CLI populates it — sets the API/server-friendly precedent.
- **`ProgressEvent` types.** Sketch the protocol in M0; populate as
  later milestones add stages.
- **Snapshot-test scaffolding for prompts.** Both the decomposition
  prompt and the strategy assessor prompt should be string snapshots
  with `pytest --snapshot-update` machinery. Don't build this lazily.

---

## 10. Pointer files

- **`plan.md`** — vision, scenarios, scope, voice, milestones.
- **`TECH_DESIGN.md`** — architecture, types, plugin contracts,
  pipeline, probes, prompts, recipe schema, normalized record schema,
  CLI surface, cache, deps, tests. Implementation-ready.
- **`ROADMAP.md`** — explicitly deferred features and the design
  notes for them (eval framework, embedding coverage, etc.). Pick
  these up only when v1 has been used on real briefs.
- **`examples/scenario-malware.md`** — dual-use stress-test
  walkthrough from earlier in the design process. Useful background
  for the policy/refusal flow if it ever needs implementing
  (sensitive-domain check is currently in-scope but minimal).
