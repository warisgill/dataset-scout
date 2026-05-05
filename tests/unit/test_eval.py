"""Unit tests for the M10 ``datascout eval`` module."""

from __future__ import annotations

from pathlib import Path

import pytest

from dataset_scout import (
    JudgeBlock,
    LabelKind,
    NormalizedRecord,
    ScoutContext,
    StrategyKind,
)
from dataset_scout.eval_ import run_eval

pytestmark = pytest.mark.unit


def _row(
    *,
    text: str,
    label: str,
    label_kind: LabelKind,
    rid: str,
    axis: str | None = None,
    label_confidence: float | None = None,
) -> NormalizedRecord:
    block: JudgeBlock | None = None
    if axis is not None:
        block = JudgeBlock(
            axis=axis,
            verdict="positive" if label == "positive" else "negative",
            subcategory="cat",
            confidence=label_confidence or 0.9,
            rationale="ok",
            model="m",
            template_version="1",
            n_judges=1,
            agreement="single",
        )
    return NormalizedRecord(
        text=text,
        label=label,  # type: ignore[arg-type]
        label_kind=label_kind,
        strategy=StrategyKind.SIGNAL_PROXY,
        strategy_confidence=0.7,
        source="huggingface:fake/ds",
        source_row_id=rid,
        label_confidence=label_confidence,
        judge=block,
    )


def _write(path: Path, recs: list[NormalizedRecord]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(r.model_dump_json() for r in recs) + "\n", encoding="utf-8")
    return path


def test_eval_perfect_agreement(tmp_path: Path) -> None:
    gold = _write(
        tmp_path / "gold" / "train.jsonl",
        [
            _row(text="a", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="r1"),
            _row(text="b", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="r2"),
        ],
    )
    judged = _write(
        tmp_path / "judged" / "train.jsonl",
        [
            _row(
                text="a",
                label="positive",
                label_kind=LabelKind.JUDGED,
                rid="r1",
                axis="x",
                label_confidence=0.9,
            ),
            _row(
                text="b",
                label="benign",
                label_kind=LabelKind.JUDGED,
                rid="r2",
                axis="x",
                label_confidence=0.9,
            ),
        ],
    )
    result = run_eval(ScoutContext(), judged.parent, gold=gold.parent, axis="x")
    m = result.overall()
    assert m.precision == 1.0
    assert m.recall == 1.0
    assert m.f1 == 1.0
    assert m.coverage == 1.0
    assert m.confusion.true_positive == 1
    assert m.confusion.true_negative == 1


def test_eval_false_positive_lowers_precision(tmp_path: Path) -> None:
    gold = _write(
        tmp_path / "gold" / "train.jsonl",
        [
            _row(text="a", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="r1"),
            _row(text="b", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="r2"),
        ],
    )
    judged = _write(
        tmp_path / "judged" / "train.jsonl",
        [
            _row(
                text="a",
                label="positive",
                label_kind=LabelKind.JUDGED,
                rid="r1",
                axis="x",
                label_confidence=0.9,
            ),
            _row(
                text="b",
                label="positive",  # false positive
                label_kind=LabelKind.JUDGED,
                rid="r2",
                axis="x",
                label_confidence=0.9,
            ),
        ],
    )
    result = run_eval(ScoutContext(), judged.parent, gold=gold.parent, axis="x")
    m = result.overall()
    assert m.precision == pytest.approx(0.5)
    assert m.recall == 1.0
    assert m.confusion.false_positive == 1


def test_eval_unjudged_rows_excluded_from_pr(tmp_path: Path) -> None:
    gold = _write(
        tmp_path / "gold" / "train.jsonl",
        [
            _row(text="a", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="r1"),
        ],
    )
    judged = _write(
        tmp_path / "judged" / "train.jsonl",
        [
            # Below threshold: not promoted, judge block but label_kind = PROXY.
            _row(
                text="a",
                label="benign",
                label_kind=LabelKind.PROXY,
                rid="r1",
                axis="x",
                label_confidence=0.4,
            ),
        ],
    )
    result = run_eval(ScoutContext(), judged.parent, gold=gold.parent, axis="x")
    m = result.overall()
    assert m.n_judged_unknown == 1
    assert m.confusion.true_positive == 0
    assert m.confusion.false_negative == 0


def test_eval_skips_non_ground_truth_gold(tmp_path: Path) -> None:
    gold = _write(
        tmp_path / "gold" / "train.jsonl",
        [
            _row(text="a", label="positive", label_kind=LabelKind.PROXY, rid="r1"),
        ],
    )
    judged = _write(
        tmp_path / "judged" / "train.jsonl",
        [
            _row(
                text="a",
                label="positive",
                label_kind=LabelKind.JUDGED,
                rid="r1",
                axis="x",
                label_confidence=0.9,
            ),
        ],
    )
    result = run_eval(ScoutContext(), judged.parent, gold=gold.parent, axis="x")
    assert any("ground_truth" in n for n in result.notices)


def test_eval_coverage_partial(tmp_path: Path) -> None:
    gold = _write(
        tmp_path / "gold" / "train.jsonl",
        [
            _row(text="a", label="positive", label_kind=LabelKind.GROUND_TRUTH, rid="r1"),
            _row(text="b", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="r2"),
            _row(text="c", label="benign", label_kind=LabelKind.GROUND_TRUTH, rid="r3"),
        ],
    )
    judged = _write(
        tmp_path / "judged" / "train.jsonl",
        [
            _row(
                text="a",
                label="positive",
                label_kind=LabelKind.JUDGED,
                rid="r1",
                axis="x",
                label_confidence=0.9,
            ),
        ],
    )
    result = run_eval(ScoutContext(), judged.parent, gold=gold.parent, axis="x")
    m = result.overall()
    assert m.coverage == pytest.approx(1 / 3)
