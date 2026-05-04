"""Tests for brief-smell warnings (recommendation H)."""

from __future__ import annotations

import pytest

from dataset_scout.intent import brief_smell_warnings

pytestmark = pytest.mark.unit


def test_clean_brief_no_warnings():
    assert brief_smell_warnings("HTML corpora with hidden text") == []


def test_inputs_outputs_pattern_caught():
    warnings = brief_smell_warnings(
        "find labeled corpora for X. Inputs are HTML. Outputs are positive vs benign."
    )
    assert warnings, "expected a smell warning"
    assert "detector spec" in warnings[0]
    assert "describes detector inputs" in warnings[0]
    assert "describes detector outputs" in warnings[0]


def test_train_and_evaluate_pattern_caught():
    warnings = brief_smell_warnings(
        "find datasets to train and evaluate a transformer on prompt injection"
    )
    assert any("describes the train/eval plan" in w for w in warnings)


def test_classifier_pattern_caught():
    warnings = brief_smell_warnings("labeled data for a classifier for jailbreak detection")
    assert any("describes the model architecture" in w for w in warnings)


def test_long_brief_triggers_length_warning():
    long_brief = "a " * 200  # 400 chars
    warnings = brief_smell_warnings(long_brief)
    assert any("400 characters" in w or "characters" in w for w in warnings)


def test_short_clean_brief_passes():
    assert brief_smell_warnings("refusal-labeled corpora for support agents") == []
