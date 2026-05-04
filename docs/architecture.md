# Architecture

The shape of the codebase, the pipeline, and what's wired in each
milestone.

---

## 1. Pipeline

```
brief + flags
    │
    ▼
HeuristicIntentParser  ────▶  Intent
    │
    ▼ (brief_smell_warnings → notices)
    │
    ▼
llm_available(ctx)?  ──── No ──▶  metadata-only mode
    │ Yes
    ▼
decompose_intent (Azure OpenAI / Entra)        ←──  --decomposition-from
    │                                                 reuses a saved file
    ▼                                                 and skips this call
DecompositionDirection × N
    │
    ▼
Source.search(intent, directions, …)  → Candidate pool (deduped, surfaced_by)
    │      (HF today; Kaggle / PWC in M1b)
    ▼
Cheap probes (license, size, recency, freshness, languages, card_completeness)
    │
    ▼
Scorecard per candidate
    │
    ▼
select_top_for_assessor (stage-1 per-direction, stage-2 quality re-rank)
    │
    ▼
LLM strategy assessor on top-15-20  ─────► Strategy[] per candidate
    │       │
    │       └─── source.stream_rows(candidate, take=8)  ←── ROW-AWARE
    │                                                       transforms get
    ▼                                                       REAL columns
LLM coverage report  ──────────────────► CoverageGap[]
    │
    ▼
re-rank scorecards by best_strategy + kind bonus
    │
    ▼
ReconResult ─── render ──▶  report.md         (gaps lead when notable)
                         │
                         └▶  results.json
                         │
                         └▶  decomposition.yaml  (stand-alone, hand-editable)
                         │
                         └▶  recipe.draft.yaml   (real columns ready for curate)
```

Plain Python iterator pipeline. No DAG framework, no Celery, no Ray.

---

## 2. Module layout

```
src/dataset_scout/
├── __init__.py             public API: recon(), inspect(), curate(), types
├── context.py              ScoutContext (frozen Pydantic, no global state)
├── core.py                 Intent, Candidate, CandidateMetadata, Scorecard,
│                           Strategy, ReconResult, InspectResult,
│                           NormalizedRecord, … — the typed vocabulary
├── errors.py               DatasetScoutError, LLMError, SourceUnavailableError, …
├── events.py               ProgressEvent / ProgressEventKind
├── intent.py               HeuristicIntentParser + brief_smell_warnings
├── llm_client.py           shared AOAI/Entra plumbing for LLM call sites
├── decompose.py            LLM decomposition (Azure OpenAI + Entra)
├── decomposition_io.py     decomposition.yaml read/write (--decomposition-from)
├── strategy.py             LLM per-candidate strategy assessor (row-aware)
├── coverage.py             LLM coverage-gap report
├── shortlist.py            two-stage selector for the assessor
├── recipe.py               typed Recipe / RecipeComponent / RecipeTransform models
├── recipe_draft.py         recipe.draft.yaml emission
├── recipe_compose.py       merge multiple recipes (datascout compose)
├── inspect_.py             single-candidate deep-dive
├── curate.py               recipe → corpus orchestrator (M4a preview)
├── tour.py                 canned demo for `datascout tour`
├── stats.py                Wilson score CI helper
├── pipeline.py             run_recon orchestrator
├── licenses.py             tiny SPDX guesser
├── cli.py                  Typer app (thin wrapper)
├── sources/
│   ├── base.py             Source Protocol, Obligation, Budget
│   └── huggingface.py      HuggingFaceSource
├── probes/
│   ├── base.py             Probe Protocol, ProbeRegistry
│   └── cheap.py            6 metadata-driven probes
└── render/
    ├── json_writer.py      results.json
    ├── markdown_report.py  report.md (4 framings + sourcing-roadmap lead)
    └── inspect_panel.py    inspect deep-dive markdown
```

---

## 3. Library API surface

The library is the source of truth; the CLI is a thin wrapper.

```python
from dataset_scout import (
    ScoutContext,
    recon,                  # M2a (works in metadata-only when AOAI absent)
    inspect,                # M3 — NotImplementedError
    curate,                 # M4 — NotImplementedError
)

ctx = ScoutContext.from_env()
result = recon("your brief here", ctx=ctx)

# result is a ReconResult with: intent, candidates (Scorecards),
# sources_searched, coverage (when LLM ran), elapsed_seconds, notices.
```

Everything in the public surface is **Pydantic v2**, so JSON
serialization, JSON Schema export, and round-trip validation are free.
A future HTTP API is a thin wrapper over the library.

---

## 4. Source plugin contract

```python
class Source(Protocol):
    name: str

    def search(
        self,
        intent: Intent,
        directions: list[DecompositionDirection],
        *,
        budget: Budget,
    ) -> Iterator[Candidate]: ...

    def fetch_metadata(self, candidate: Candidate) -> dict[str, Any]: ...

    def stream_sample(
        self, candidate: Candidate, n: int, seed: int,
    ) -> Iterator[dict[str, Any]]: ...

    def card_url(self, candidate: Candidate) -> str: ...

    def terms_check(self, intent: Intent) -> list[Obligation]: ...
```

Sources are registered via `pyproject.toml` `entry_points` in the
`dataset_scout.sources` group. The HF source is wired today; Kaggle
and PWC entries exist commented out and light up in M1b.

Crucially, every source plugin populates the **same**
`CandidateMetadata` envelope. Probes never read source-specific keys.

---

## 5. Probe protocol

```python
class Probe(Protocol):
    name: str
    version: str
    def applies(self, candidate: Candidate, intent: Intent) -> bool: ...
    def run(self, candidate: Candidate, intent: Intent) -> SubScore: ...
```

Probes are stateless and parallelizable. The version field anchors
future cache keys: `(candidate.revision, probe.version, intent.stable_hash())`.

The six **cheap** probes shipping today consume only
`CandidateMetadata` (no row sampling). **Sample-driven** probes
(`label_structure`, `schema_fingerprint`, embedding label-intent fit)
land with `Source.stream_sample()` in M1b.

---

## 6. Mode detection

```python
def llm_available(ctx: ScoutContext) -> bool:
    return ctx.aoai_configured
```

Cheap one-liner. Importantly, it does **not** import `litellm` or
`azure-identity` (~10 s import cost on first use). Users with no AOAI
configured pay nothing.

When `llm_available` returns False, the pipeline:

- Skips decomposition entirely.
- Emits the `METADATA_ONLY_NOTICE` on stderr and as a prominent header
  in `report.md`, naming the env vars to set.
- Still runs HF search + cheap probes — discovery still works.

When `llm_available` returns True but the call fails at runtime
(expired token, network, rate limit), the same fallback kicks in
with a more specific notice.

---

## 7. Events

The pipeline emits `ProgressEvent`s as it works:

```python
class ProgressEventKind(StrEnum):
    STAGE_STARTED, STAGE_FINISHED,
    CANDIDATE_FOUND, CANDIDATE_SCORED,
    DIRECTION_PROPOSED, STRATEGY_ASSESSED,
    NOTICE, WARNING
```

Tests collect events into a list to assert on. The CLI passes a sink
that renders to rich progress bars (lands properly in M2b). A future
HTTP API forwards them as SSE / WebSocket frames. Same code path.

---

## 8. Cache *(M1b)*

SQLite WAL at `~/.cache/dataset-scout/cache.db`, single-writer file
lock, LRU eviction at 2 GB default cap. Key namespaces:

```
hf_meta:{id}:{revision}                          (TTL 7d)
hf_sample:{id}:{revision}:{seed}                 (infinite)
probe:{source}:{id}:{revision}:{name}:{version}
embed_fit:{source}:{id}:{revision}:{intent_hash}
strategy:{source}:{id}:{revision}:{intent_hash}:{assessor_v}
decompose:{intent_hash}:{decomposer_v}
coverage:{intent_hash}:{candidate_set_hash}:{coverage_v}
```

`Intent.stable_hash()` already exists today; cache wrapping is
mechanical when M1b lands.

---

## 9. Tests

Three tiers:

| Marker | Network? | Gates? |
|---|---|---|
| `pytest -m unit` | No | every PR (currently 154 tests) |
| `pytest -m recorded` | respx-replayed | every PR |
| `pytest -m live` | Real HF + AOAI | nightly only |

LLM calls in tests are mocked at the `litellm.completion` boundary
with canned JSON responses. Decomposition + (future) strategy
prompts are **snapshot-tested** — drift surfaces as a PR diff.

---

## 10. Milestones

| Milestone | Status | What's in |
|---|---|---|
| M0 | ✅ done | scaffolding: pyproject, src layout, core types, CLI stubs, 30 tests |
| M1a | ✅ done | discovery slice — HF, 6 cheap probes, report.md / results.json, FakeSource, recorded harness |
| M1b | ⏳ deferred | sample-driven probes, embedding fit, Kaggle, PWC, cache |
| M2a | ✅ done | Azure OpenAI Entra, LLM decomposition, multi-direction search, surfaced_by, mode detection |
| M2b | ✅ done | strategy assessor, coverage report, recipe.draft.yaml, two-stage shortlist, ranked report |
| **Wow loop** | ✅ done | `tour` (no-setup demo), `decompose` (cheap brief iteration), `--decomposition-from` reuse, **row-aware strategy assessor** (recipes ship with REAL columns), `compose` (multi-recipe merge), sourcing-roadmap-led report, brief-smell hints |
| M3 | ✅ done | `inspect` deep-dive — schema + Wilson-CI label distribution + length stats + license + strategy assessment |
| M4a | ✅ done | `curate` preview slice — recipe → JSONL + lockfile + manifest + report + fingerprint + usage. Hash-mod split (NOT leakage-aware), filter DSL hard-fails. |
| M4b | 🔄 next | MinHash dedup + leakage-aware splitter + filter DSL — flips `recipe.lock.yaml` `audit_readiness: preview → ready` |
| M5 | ⏳ | real-brief validation, ship 1.0 |

Detail in [`docs/concepts.md`](concepts.md) and the (archived)
TECH_DESIGN spec.

---

## 11. Voice for output

The audience is detection engineers under audit pressure. **No
aggregate "quality" headlines.** Receipts everywhere. Proxies are
honest by default (see `label_kind` in [concepts.md](concepts.md)).
"This is not legal advice" footer on any report touching licensing.
