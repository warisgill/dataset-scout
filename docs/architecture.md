# Architecture

The shape of the codebase, the pipeline, and what's wired in each
milestone.

---

## 1. Pipeline

The pre-eval data flow, end to end. Solid arrows are required;
dashed arrows are M10 opt-ins.

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
                                          │
                                          ▼
                              datascout curate  ─── parallel materialiser ──▶
                                          │       (per-component soft failures,
                                          │        MinHash dedup, leakage-aware
                                          │        splits, filter DSL)
                                          ▼
                              Corpus directory:
                                   train.jsonl  /  eval.jsonl
                                   recipe.lock.yaml   (audit_readiness: ready)
                                   report.md          (5-second scorecard)
                                   manifest.json      (fingerprint)
                                          │
                                          ▼  - - (M10 opt-in: rescue weak labels)
                              datascout judge  ─── per-row LLM verdict ──▶
                                          │       (axis question, optional rubric,
                                          │        single / majority-of-3 /
                                          │        unanimous-of-5, sha256-cached,
                                          │        per-batch resumable)
                                          ▼
                              <corpus>/judged/   judged.jsonl
                                                 judge.lock.yaml
                                                 judge.report.md
                                          │
                                          ▼  - - (M10 standalone: any time)
                              datascout eval  ─── precision/recall/F1 ──▶
                                                  confusion + coverage
                                                  vs --against gold
```

Plain Python iterator pipeline. No DAG framework, no Celery, no Ray.
Library is the source of truth — every verb is a `run_<verb>(ctx, ...)`
function; CLI is a thin wrapper.

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
├── curate.py               recipe → corpus orchestrator (audit-ready)
├── filter_dsl.py           sandboxed filter expression compiler
├── dedup.py                MinHash + LSH + leakage-aware splitter
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

## 10. Status & roadmap

The product surface is feature-complete for its core loop: **brief →
recon → curate**. Below is the honest state.

### Shipped

| Capability | What's in |
|---|---|
| **Discovery (dataset platforms)** | HuggingFace + Kaggle source plugins, 6 metadata-driven probes, deduplicated multi-source candidate pool, `surfaced_by` provenance. Kaggle is discovery-only — `stream_sample`/`stream_rows` raise `SourceUnsupportedError` and curate classifies the candidate under `unsupported_source` with a hint. |
| **Discovery (academic papers)** | Pipeline stage (not a Source plugin — papers ≠ datasets) querying Semantic Scholar across NeurIPS / ICML / ICLR / SaTML. Round-robin per-direction queries, regex extraction of HF/Kaggle/GitHub dataset URLs from abstracts, deduped + capped at 12 papers per recon. Cited HF / Kaggle datasets are promoted into the candidate pool with `surfaced_by` carrying the paper id; existing strategy / coverage flow then runs over them. CLI: `--no-papers` opts out. Cached per `(venue-set, year-range, query)`. |
| **Cache** | SQLite WAL at `<ctx.cache_dir>/cache.db`. Namespace-scoped TTL defaults with per-call override; age-based eviction at a 2 GB cap (`DATASET_SCOUT_CACHE_MAX_BYTES`); read paths never write. Wraps decompose, strategy, embedding, and paper-search calls. CLI: `datascout cache info\|prune\|clear`. |
| **LLM decomposition** | Azure OpenAI / Entra. Brief → 3–7 adjacent search directions. Reusable via `--decomposition-from` for cheap iteration. Cached. Mode-detection falls back to metadata-only with an explicit notice when AOAI is absent. |
| **Row-aware strategy assessor** | For each shortlisted candidate: stream 8 real rows → LLM call → 1–4 ranked strategies from the 7-kind taxonomy with rationale, caveats, and a transform spec referencing **actual** columns and label values. Cached. |
| **Embedding label-intent fit** | Dedicated pipeline stage between probes and the assessor. Embeds intent text + a deterministic candidate text (description + canonical row sample) via Azure OpenAI embeddings (`AZURE_OPENAI_EMBEDDING_DEPLOYMENT`). Writes `Scorecard.label_intent_fit`. Cached per text hash; the intent embedding is reused across candidates. |
| **Coverage-gap report** | What no candidate covers and where to source it. Leads `report.md`/`report.html` when notable. |
| **Reports** | `report.md` (audit-friendly Markdown) and `report.html` (self-contained HTML, embedded CSS, color-coded strategy badges, no JS) rendered from a shared `ReconReportContext` view-model so the two can't drift. |
| **`tour`** | Canned no-setup demo. Writes both Markdown and HTML when `--out` is given. |
| **`inspect`** | Single-candidate deep-dive: schema, Wilson 95% CI label distribution, length stats, license, strategy assessment. |
| **`curate`** | Audit-ready: parallel materialisation with deterministic reassembly, MinHash + LSH dedup, leakage-aware splits, sandboxed filter DSL, per-component soft-failure classification (now including `unsupported_source` for Kaggle), full `recipe.lock.yaml` audit trail. |
| **`judge` + `eval`** | Opt-in LLM-as-judge label rescue with sha256-cached verdicts, per-batch resumable checkpoint, multi-judge agreement, explicit-gap promotion, calibration mode with precision floor. Standalone `eval` against any gold corpus. |
| **`compose`** | Merge multiple recipes for shared multi-detection corpora. |

### Considering

- **`--watch`/re-validate mode.** Re-check a `recipe.lock.yaml`
  against current upstream revisions: *"has anything I depend on
  moved?"*
- **Archive option.** Pin contents (not just revisions) so a corpus
  is reproducible even if upstream deletes the data.

### Deferred

- **Lineage DAG + resumable-operation envelope.** Engineering
  cleanup with no user request behind it. Revisit when somebody asks.
- **Multi-candidate portfolio assessor + the `composition_only` 8th
  strategy kind.** Clever, not yet earned.
- **Papers-with-Code source plugin.** Entry-point hook reserved; not
  yet wired.

---

## 11. Voice for output

The audience is detection engineers under audit pressure. **No
aggregate "quality" headlines.** Receipts everywhere. Proxies are
honest by default (see `label_kind` in [concepts.md](concepts.md)).
"This is not legal advice" footer on any report touching licensing.
