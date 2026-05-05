"""``datascout eval`` — precision/recall/F1 of a judged corpus vs gold.

Generic comparator: takes two scout corpora — a "judged or any-label"
corpus and a "gold" corpus — and computes per-axis precision / recall /
F1, a confusion matrix, and coverage (fraction of gold rows that have a
corresponding row in the judged corpus, joined on
:attr:`NormalizedRecord.stable_id`).

Used by:

1. The internal calibration step in :func:`dataset_scout.judge.run_judge`
   when ``--calibrate-against`` is passed.
2. Post-hoc comparison from the CLI (``datascout eval``).

Same library/CLI separation as the rest of scout. Reference:
``M10-judge-design.md`` §8.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from dataset_scout.core import LabelKind, NormalizedRecord
from dataset_scout.errors import DatasetScoutError

if TYPE_CHECKING:
    from dataset_scout.context import ScoutContext


# Labels we score over. The judge maps positive → "positive" and
# negative → "benign"; "hard_negative" is a recon-side category that
# the judge never produces, so it joins the negative pool when
# computing precision/recall against gold.
_NEGATIVE_LABELS: frozenset[str] = frozenset({"benign", "hard_negative"})
_POSITIVE_LABEL: str = "positive"


# ─── result types ───────────────────────────────────────────────────


@dataclass
class ConfusionMatrix:
    """2x2 confusion matrix (positive vs negative).

    ``true_positive``: judged positive AND gold positive.
    ``false_positive``: judged positive AND gold negative.
    ``false_negative``: judged negative AND gold positive.
    ``true_negative``: judged negative AND gold negative.
    """

    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0
    true_negative: int = 0

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass
class AxisMetrics:
    """Per-axis P/R/F1 + confusion + coverage.

    For corpora with a single (implicit) axis the consumer can call
    :meth:`EvalResult.overall` to get the same values without keying.
    """

    axis: str
    n_gold: int = 0
    n_judged_seen: int = 0
    n_joined: int = 0
    n_judged_positive: int = 0
    n_judged_negative: int = 0
    n_judged_unknown: int = 0
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    @property
    def precision(self) -> float:
        cm = self.confusion
        denom = cm.true_positive + cm.false_positive
        return cm.true_positive / denom if denom else 0.0

    @property
    def recall(self) -> float:
        cm = self.confusion
        denom = cm.true_positive + cm.false_negative
        return cm.true_positive / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.recall
        return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0

    @property
    def coverage(self) -> float:
        return self.n_joined / self.n_gold if self.n_gold else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "axis": self.axis,
            "n_gold": self.n_gold,
            "n_judged_seen": self.n_judged_seen,
            "n_joined": self.n_joined,
            "n_judged_positive": self.n_judged_positive,
            "n_judged_negative": self.n_judged_negative,
            "n_judged_unknown": self.n_judged_unknown,
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "coverage": round(self.coverage, 6),
            "confusion": self.confusion.as_dict(),
        }


@dataclass
class EvalResult:
    """Top-level eval output.

    ``per_axis`` is keyed by axis name. When ``axis`` is filtered (or
    only one axis appears in the data) :meth:`overall` returns that
    single AxisMetrics for ergonomic access.
    """

    per_axis: dict[str, AxisMetrics] = field(default_factory=dict)
    notices: list[str] = field(default_factory=list)

    def overall(self) -> AxisMetrics:
        """Convenience accessor when only one axis was scored."""
        if len(self.per_axis) != 1:
            raise DatasetScoutError(
                f"EvalResult.overall() requires exactly one axis; got {list(self.per_axis)}"
            )
        return next(iter(self.per_axis.values()))

    def as_dict(self) -> dict[str, object]:
        return {
            "per_axis": {k: v.as_dict() for k, v in self.per_axis.items()},
            "notices": list(self.notices),
        }


# ─── corpus loading ─────────────────────────────────────────────────


_CORPUS_FILES: tuple[str, ...] = ("train.jsonl", "val.jsonl", "test.jsonl")


def _resolve_corpus_files(target: Path) -> list[Path]:
    if target.is_file() and target.suffix == ".jsonl":
        return [target]
    if not target.is_dir():
        raise DatasetScoutError(f"eval target {target} is neither a directory nor a .jsonl file")
    files = [target / name for name in _CORPUS_FILES if (target / name).is_file()]
    if not files:
        files = sorted(target.glob("*.jsonl"))
    if not files:
        raise DatasetScoutError(f"no JSONL files found under {target}")
    return files


def _iter_records(target: Path) -> Iterator[NormalizedRecord]:
    for path in _resolve_corpus_files(target):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield NormalizedRecord.model_validate(json.loads(line))


# ─── public entry point ─────────────────────────────────────────────


def _binary_class(label: str) -> str:
    """Collapse the three-class scout label to {positive, negative}."""
    if label == _POSITIVE_LABEL:
        return "positive"
    if label in _NEGATIVE_LABELS:
        return "negative"
    raise DatasetScoutError(f"unknown corpus label: {label!r}")


def _eligible_gold(rec: NormalizedRecord) -> bool:
    """Gold rows are those carrying authoritative labels.

    We accept ``label_kind == GROUND_TRUTH`` only — that's the audit
    contract. Other label_kinds (``proxy``, ``judged``, ``remapped``,
    ``subset_extracted``) are too noisy to use as eval gold.
    """
    return rec.label_kind == LabelKind.GROUND_TRUTH


def run_eval(
    ctx: ScoutContext,
    judged: Path,
    *,
    gold: Path,
    axis: str | None = None,
) -> EvalResult:
    """Compute precision / recall / F1 / coverage for ``judged`` vs ``gold``.

    Rows are joined on :attr:`NormalizedRecord.stable_id`. Gold rows
    must carry ``label_kind == GROUND_TRUTH`` (the eval contract);
    other gold-side label kinds are skipped with a notice.

    When ``axis`` is provided, only judged rows whose
    :attr:`NormalizedRecord.judge` block matches (or whose
    ``label_kind == JUDGED`` for that axis) participate. Without
    ``axis``, all joinable rows participate under the bucket
    ``"<unspecified>"``.
    """
    notices: list[str] = []
    gold_index: dict[str, NormalizedRecord] = {}
    n_gold_skipped = 0
    for rec in _iter_records(gold):
        if not _eligible_gold(rec):
            n_gold_skipped += 1
            continue
        gold_index[rec.stable_id] = rec
    if n_gold_skipped:
        notices.append(f"{n_gold_skipped} gold row(s) skipped (label_kind != ground_truth)")

    per_axis: dict[str, AxisMetrics] = {}

    def _bucket(name: str) -> AxisMetrics:
        if name not in per_axis:
            per_axis[name] = AxisMetrics(axis=name)
        return per_axis[name]

    seen_judged_ids: set[str] = set()
    for rec in _iter_records(judged):
        rec_axis: str | None = None
        if rec.judge is not None:
            rec_axis = rec.judge.axis
        if axis is not None and rec_axis is not None and rec_axis != axis:
            continue
        bucket = _bucket(rec_axis if rec_axis is not None else (axis or "<unspecified>"))
        bucket.n_judged_seen += 1
        seen_judged_ids.add(rec.stable_id)
        gold_rec = gold_index.get(rec.stable_id)
        if gold_rec is None:
            continue
        bucket.n_joined += 1
        if rec.label_kind == LabelKind.JUDGED:
            judged_class = _binary_class(rec.label)
            if judged_class == "positive":
                bucket.n_judged_positive += 1
            else:
                bucket.n_judged_negative += 1
        else:
            bucket.n_judged_unknown += 1
            # Unpromoted rows don't contribute to precision/recall —
            # they're "no decision". Track for coverage clarity only.
            continue
        gold_class = _binary_class(gold_rec.label)
        cm = bucket.confusion
        if judged_class == "positive" and gold_class == "positive":
            cm.true_positive += 1
        elif judged_class == "positive":
            cm.false_positive += 1
        elif gold_class == "positive":
            cm.false_negative += 1
        else:
            cm.true_negative += 1

    # n_gold for each axis = number of gold rows (axis filtering on the
    # gold side is identity since gold rows don't carry an axis).
    for bucket in per_axis.values():
        bucket.n_gold = len(gold_index)

    if not per_axis and axis is not None:
        per_axis[axis] = AxisMetrics(axis=axis, n_gold=len(gold_index))
        notices.append(f"no judged rows matched axis={axis!r}")

    return EvalResult(per_axis=per_axis, notices=notices)


__all__ = [
    "AxisMetrics",
    "ConfusionMatrix",
    "EvalResult",
    "run_eval",
]
