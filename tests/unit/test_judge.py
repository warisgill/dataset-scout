"""Unit tests for the M10 LLM-as-judge module.

Covers single-judge core (cache + strict promotion + soft failures),
checkpoint resumability, and multi-judge agreement aggregation.
Reference: ``M10-judge-design.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dataset_scout import (
    JudgeBlock,
    LabelKind,
    NormalizedRecord,
    ScoutContext,
    StrategyKind,
)
from dataset_scout.errors import LLMError
from dataset_scout.judge import (
    _CHECKPOINT_NAME,
    ContentFilterError,
    _cache_key,
    _ChatClient,
    _promote,
    judge_cache_dir,
    render_judge_prompt,
    run_judge,
)

pytestmark = pytest.mark.unit


# ─── helpers ────────────────────────────────────────────────────────


def _ctx(tmp_path: Path) -> ScoutContext:
    return ScoutContext(cache_dir=tmp_path / "cache", out_dir=tmp_path / "out")


def _row(
    *,
    text: str,
    label: str = "benign",
    label_kind: LabelKind = LabelKind.PROXY,
    rid: str | None = None,
) -> NormalizedRecord:
    return NormalizedRecord(
        text=text,
        label=label,  # type: ignore[arg-type]
        label_kind=label_kind,
        strategy=StrategyKind.SIGNAL_PROXY,
        strategy_confidence=0.7,
        source="huggingface:fake/ds",
        source_row_id=rid or text[:8],
    )


def _write_corpus(dir_path: Path, rows: list[NormalizedRecord]) -> Path:
    dir_path.mkdir(parents=True, exist_ok=True)
    target = dir_path / "train.jsonl"
    target.write_text("\n".join(r.model_dump_json() for r in rows) + "\n", encoding="utf-8")
    return dir_path


class _ScriptedChat(_ChatClient):
    """A stub chat client returning canned content per call.

    Pass either:
      - ``responses``: a list cycled through one entry per call, or
      - ``by_text``: a dict keyed by the row's text containing the
        response payload(s) to emit for that row.
    """

    def __init__(
        self,
        *,
        responses: list[Any] | None = None,
        by_text: dict[str, Any] | None = None,
        raise_on: dict[str, Exception] | None = None,
    ) -> None:
        self.responses = list(responses or [])
        self.by_text = dict(by_text or {})
        self.raise_on = dict(raise_on or {})
        self.calls: list[list[dict[str, str]]] = []
        self._call_idx = 0
        self._per_text_idx: dict[str, int] = {}

    def call(self, *, messages: list[dict[str, str]], timeout_s: float) -> str:
        self.calls.append(list(messages))
        text = messages[0]["content"]
        for needle, exc in self.raise_on.items():
            if needle in text:
                raise exc
        if self.by_text:
            for needle, payload in self.by_text.items():
                if needle in text:
                    if isinstance(payload, list):
                        idx = self._per_text_idx.get(needle, 0)
                        self._per_text_idx[needle] = idx + 1
                        chosen = payload[idx % len(payload)]
                    else:
                        chosen = payload
                    return chosen if isinstance(chosen, str) else json.dumps(chosen)
        if self.responses:
            chosen = self.responses[self._call_idx % len(self.responses)]
            self._call_idx += 1
            return chosen if isinstance(chosen, str) else json.dumps(chosen)
        return json.dumps(
            {
                "verdict": "ambiguous",
                "subcategory": "default",
                "confidence": 0.5,
                "rationale": "default",
            }
        )


def _payload(
    verdict: str = "positive",
    confidence: float = 0.9,
    subcategory: str = "test-cat",
    rationale: str = "ok",
) -> dict[str, Any]:
    return {
        "verdict": verdict,
        "subcategory": subcategory,
        "confidence": confidence,
        "rationale": rationale,
    }


# ─── prompt rendering ───────────────────────────────────────────────


def test_render_includes_axis_and_text() -> None:
    p = render_judge_prompt(axis="psych_harm", rubric="be strict", text="hello")
    assert "psych_harm" in p
    assert "be strict" in p
    assert "hello" in p
    assert "STRICT JSON" in p


def test_render_uses_placeholder_when_no_rubric() -> None:
    p = render_judge_prompt(axis="psych_harm", rubric=None, text="hi")
    assert "(no rubric supplied" in p


def test_cache_key_changes_with_template_version() -> None:
    a = _cache_key(prompt="x", axis="y", model="m", template_version="1")
    b = _cache_key(prompt="x", axis="y", model="m", template_version="2")
    assert a != b


# ─── core run_judge ─────────────────────────────────────────────────


def test_run_judge_promotes_high_confidence_positive(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="self-harm content here", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="psych_harm", chat_client=chat, threshold=0.8)
    assert result.stats.n_input == 1
    assert result.stats.n_judged == 1
    assert result.stats.n_promoted_positive == 1
    assert result.stats.n_left_unknown == 0
    out_file = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out_file.read_text().splitlines()[0]))
    assert rec.label == "positive"
    assert rec.label_kind == LabelKind.JUDGED
    assert rec.label_confidence == pytest.approx(0.95)
    assert rec.judge is not None
    assert rec.judge.axis == "psych_harm"
    assert rec.judge.verdict == "positive"


def test_run_judge_does_not_promote_below_threshold(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="borderline", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.6)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, threshold=0.8)
    assert result.stats.n_promoted_positive == 0
    assert result.stats.n_left_unknown == 1
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    # Original label preserved, but JudgeBlock attached.
    assert rec.label == "benign"
    assert rec.label_kind == LabelKind.PROXY
    assert rec.judge is not None
    assert rec.judge.verdict == "positive"


def test_run_judge_skips_ground_truth_by_default(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [
            _row(text="t1", label_kind=LabelKind.GROUND_TRUTH, rid="r1"),
            _row(text="t2", label_kind=LabelKind.PROXY, rid="r2"),
        ],
    )
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_judged == 1
    assert len(chat.calls) == 1


def test_run_judge_re_judge_all_overrides_skip(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="t1", label_kind=LabelKind.GROUND_TRUTH, rid="r1")],
    )
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, re_judge_all=True)
    assert result.stats.n_judged == 1


def test_run_judge_uses_disk_cache_on_second_run(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    chat1 = _ScriptedChat(responses=[_payload("positive", 0.95)])
    run_judge(ctx, target, axis="x", chat_client=chat1)
    assert len(chat1.calls) == 1

    # Second run with a different out_dir but same workspace cache.
    chat2 = _ScriptedChat(raise_on={"hi": LLMError("must not be called — cache should hit")})
    result = run_judge(ctx, target, axis="x", chat_client=chat2, out_dir=tmp_path / "out2")
    assert result.stats.n_cache_hits == 1
    assert result.stats.n_judged == 1
    assert len(chat2.calls) == 0


def test_run_judge_recovers_from_corrupt_cache_file(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    # Plant a corrupt cache entry.
    cache_dir = judge_cache_dir(ctx)
    cache_dir.mkdir(parents=True, exist_ok=True)
    # We don't know the key without rendering the prompt; brute-force a
    # corrupt file in every two-char shard the run might use.
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_judged == 1
    # Now corrupt the resulting cache file and re-run.
    corrupted = 0
    for f in cache_dir.rglob("*.json"):
        f.write_text("{not json", encoding="utf-8")
        corrupted += 1
    assert corrupted >= 1
    chat2 = _ScriptedChat(responses=[_payload("positive", 0.9)])
    res2 = run_judge(ctx, target, axis="x", chat_client=chat2, out_dir=tmp_path / "out2")
    # Re-judged because the cache file was corrupt and got deleted.
    assert res2.stats.n_judged == 1
    assert len(chat2.calls) == 1


def test_run_judge_soft_fails_on_api_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="bad", rid="r1"), _row(text="good", rid="r2")],
    )
    chat = _ScriptedChat(
        by_text={"good": _payload("positive", 0.95)},
        raise_on={"bad": LLMError("transient")},
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_input == 2
    assert result.stats.n_judged == 1
    assert result.stats.n_skipped == 1
    assert result.stats.n_api_errors == 1
    assert len(result.failures) == 1


def test_run_judge_soft_fails_on_content_filter(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    chat = _ScriptedChat(raise_on={"x": ContentFilterError("blocked")})
    result = run_judge(ctx, target, axis="psych_harm", chat_client=chat)
    assert result.stats.n_content_filter_blocked == 1
    assert result.stats.n_skipped == 1


def test_run_judge_retries_once_on_invalid_json(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    chat = _ScriptedChat(responses=["not json", _payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_judged == 1
    assert len(chat.calls) == 2
    # Retry message is sent as a user follow-up.
    assert any("STRICT JSON ONLY" in m["content"] for m in chat.calls[1])


def test_run_judge_soft_fails_after_two_invalid_json(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="hi", rid="r1")])
    chat = _ScriptedChat(responses=["not json", "still bad"])
    result = run_judge(ctx, target, axis="x", chat_client=chat)
    assert result.stats.n_parse_errors == 1
    assert result.stats.n_skipped == 1


def test_run_judge_dry_run_makes_no_calls(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="a", rid="r1"), _row(text="b", rid="r2")],
    )
    chat = _ScriptedChat(raise_on={"a": LLMError("nope"), "b": LLMError("nope")})
    result = run_judge(
        ctx, target, axis="x", chat_client=chat, dry_run=True, judges=3, agreement="majority"
    )
    assert result.dry_run is True
    assert result.estimated_calls == 6
    assert len(chat.calls) == 0
    # No output JSONL files written on dry-run.
    assert not (result.out_dir / "train.jsonl").exists()


# ─── checkpoint / resumability ──────────────────────────────────────


def test_run_judge_writes_checkpoint(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text=f"t{i}", rid=f"r{i}") for i in range(3)])
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    out = tmp_path / "out"
    run_judge(ctx, target, axis="x", chat_client=chat, out_dir=out, batch_size=1)
    cp = json.loads((out / _CHECKPOINT_NAME).read_text(encoding="utf-8"))
    assert cp["axis"] == "x"
    assert len(cp["completed_row_ids"]) == 3


def test_run_judge_resume_skips_completed_rows(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(
        tmp_path / "corpus",
        [_row(text="t1", rid="r1"), _row(text="t2", rid="r2")],
    )
    out = tmp_path / "out"
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    run_judge(ctx, target, axis="x", chat_client=chat, out_dir=out)
    # Second run: cache hits keep call count at zero; checkpoint
    # records the rows as already completed.
    chat2 = _ScriptedChat(
        raise_on={"t1": LLMError("must not be called"), "t2": LLMError("must not be called")}
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat2, out_dir=out)
    assert result.stats.n_resumed == 2
    assert len(chat2.calls) == 0


def test_run_judge_corrupt_checkpoint_starts_fresh(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    out = tmp_path / "out"
    out.mkdir(parents=True, exist_ok=True)
    (out / _CHECKPOINT_NAME).write_text("{not json", encoding="utf-8")
    chat = _ScriptedChat(responses=[_payload("positive", 0.95)])
    result = run_judge(ctx, target, axis="x", chat_client=chat, out_dir=out)
    assert result.stats.n_resumed == 0
    assert result.stats.n_judged == 1


# ─── multi-judge agreement ──────────────────────────────────────────


def test_majority_agreement_picks_majority_verdict(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    # 3 judges: positive (0.9), positive (0.8), negative (0.7).
    chat = _ScriptedChat(
        responses=[
            _payload("positive", 0.9, subcategory="cat-a"),
            _payload("positive", 0.8, subcategory="cat-a"),
            _payload("negative", 0.7),
        ]
    )
    result = run_judge(
        ctx,
        target,
        axis="x",
        chat_client=chat,
        judges=3,
        agreement="majority",
        threshold=0.5,
    )
    assert len(chat.calls) == 3
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "positive"
    assert rec.judge.n_judges == 3
    assert rec.judge.agreement == "majority"
    # Derived label_confidence = (2/3) * mean(0.9, 0.8) = 0.5666...
    assert rec.label_confidence == pytest.approx(0.566667, rel=1e-3)
    assert rec.label_kind == LabelKind.JUDGED


def test_majority_no_majority_marks_ambiguous(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    chat = _ScriptedChat(
        responses=[
            _payload("positive", 0.9),
            _payload("negative", 0.9),
            _payload("ambiguous", 0.9),
        ]
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat, judges=3, agreement="majority")
    assert result.stats.n_left_unknown == 1
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "ambiguous"


def test_unanimous_requires_all_to_agree(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    # 5 judges, one disagrees → ambiguous.
    chat = _ScriptedChat(
        responses=[
            _payload("positive", 0.9),
            _payload("positive", 0.9),
            _payload("positive", 0.9),
            _payload("positive", 0.9),
            _payload("negative", 0.9),
        ]
    )
    result = run_judge(ctx, target, axis="x", chat_client=chat, judges=5, agreement="unanimous")
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "ambiguous"


def test_unanimous_promotes_when_all_agree(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="t1", rid="r1")])
    chat = _ScriptedChat(responses=[_payload("positive", 0.85)] * 5)
    result = run_judge(ctx, target, axis="x", chat_client=chat, judges=5, agreement="unanimous")
    out = result.out_dir / "train.jsonl"
    rec = NormalizedRecord.model_validate(json.loads(out.read_text().splitlines()[0]))
    assert rec.judge is not None
    assert rec.judge.verdict == "positive"
    assert rec.label_confidence == pytest.approx(0.85)
    assert rec.label_kind == LabelKind.JUDGED


def test_majority_requires_judges_ge_3(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    with pytest.raises(Exception, match="majority"):
        run_judge(
            ctx,
            target,
            axis="x",
            judges=2,
            agreement="majority",
            chat_client=_ScriptedChat(),
        )


def test_unanimous_requires_judges_ge_3(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    with pytest.raises(Exception, match="unanimous"):
        run_judge(
            ctx,
            target,
            axis="x",
            judges=2,
            agreement="unanimous",
            chat_client=_ScriptedChat(),
        )


def test_single_with_judges_gt_1_rejected(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    target = _write_corpus(tmp_path / "corpus", [_row(text="x", rid="r1")])
    with pytest.raises(Exception, match="agreement"):
        run_judge(
            ctx,
            target,
            axis="x",
            judges=3,
            agreement="single",
            chat_client=_ScriptedChat(),
        )


# ─── promotion rule directly ────────────────────────────────────────


def test_promote_negative_above_threshold_writes_benign() -> None:
    rec = _row(text="t", label="positive", label_kind=LabelKind.PROXY, rid="r1")
    block = JudgeBlock(
        axis="x",
        verdict="negative",
        subcategory="benign-talk",
        confidence=0.95,
        rationale="ok",
        model="m",
        template_version="1",
        n_judges=1,
        agreement="single",
    )
    out = _promote(rec, block, 0.95, threshold=0.8)
    assert out.label == "benign"
    assert out.label_kind == LabelKind.JUDGED
    assert out.label_confidence == pytest.approx(0.95)


def test_promote_ambiguous_keeps_label() -> None:
    rec = _row(text="t", rid="r1")
    block = JudgeBlock(
        axis="x",
        verdict="ambiguous",
        subcategory="unclear",
        confidence=0.99,
        rationale="ok",
        model="m",
        template_version="1",
        n_judges=1,
        agreement="single",
    )
    out = _promote(rec, block, 0.99, threshold=0.5)
    assert out.label_kind == LabelKind.PROXY
    assert out.judge is not None
    assert out.label_confidence == pytest.approx(0.99)
