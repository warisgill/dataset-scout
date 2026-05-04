"""Unit tests for the M4b filter DSL."""

from __future__ import annotations

import pytest

from dataset_scout.filter_dsl import (
    FilterCompileError,
    FilterEvalError,
    compile_filter,
    matches,
)

pytestmark = pytest.mark.unit


# ─── compilation: allowed shapes ───────────────────────────────────


@pytest.mark.parametrize(
    "expr",
    [
        "label == 1",
        "label == 'positive'",
        "len(text) > 50",
        "label != 'spam' and len(text) > 30",
        "contains_pattern(text, 'hello')",
        "not (label == 'junk')",
        "lower(text) == 'foo'",
        "startswith(prompt, 'IGNORE')",
        "endswith(prompt, '!')",
        "int(score) > 5",
        "label in ('positive', 'benign')",
    ],
)
def test_allowed_expressions_compile(expr: str):
    fn = compile_filter(expr)
    assert callable(fn)


# ─── compilation: rejected shapes ──────────────────────────────────


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "label.upper() == 'X'",  # attribute access disallowed
        "[x for x in range(10)]",  # comprehension disallowed
        "lambda x: x + 1",  # lambda disallowed
        "{1: 2}",  # dict literal disallowed
        "exec('print(1)')",  # not in whitelist
    ],
)
def test_disallowed_expressions_rejected(expr: str):
    with pytest.raises(FilterCompileError):
        compile_filter(expr)


def test_empty_expression_rejected():
    with pytest.raises(FilterCompileError):
        compile_filter("")
    with pytest.raises(FilterCompileError):
        compile_filter("   ")


def test_syntax_error_surfaced_with_message():
    with pytest.raises(FilterCompileError, match="could not parse"):
        compile_filter("label ==")


# ─── evaluation ─────────────────────────────────────────────────────


def test_evaluate_string_equality():
    fn = compile_filter("label == 'positive'")
    assert fn({"label": "positive"})
    assert not fn({"label": "benign"})


def test_evaluate_int_comparison():
    fn = compile_filter("score > 5")
    assert fn({"score": 10})
    assert not fn({"score": 3})


def test_evaluate_len_helper():
    fn = compile_filter("len(text) > 5")
    assert fn({"text": "a longer string"})
    assert not fn({"text": "abc"})
    # Missing column → len(None) == 0.
    assert not fn({})


def test_evaluate_contains_pattern():
    fn = compile_filter("contains_pattern(prompt, '(?i)ignore previous')")
    assert fn({"prompt": "Please IGNORE previous instructions"})
    assert not fn({"prompt": "hello world"})


def test_evaluate_boolean_operators():
    fn = compile_filter("label == 'positive' and len(text) > 5")
    assert fn({"label": "positive", "text": "hello world"})
    assert not fn({"label": "positive", "text": "hi"})
    assert not fn({"label": "benign", "text": "hello world"})


def test_evaluate_in_operator():
    fn = compile_filter("label in ('positive', 'benign')")
    assert fn({"label": "positive"})
    assert fn({"label": "benign"})
    assert not fn({"label": "spam"})


def test_evaluate_missing_column_returns_false():
    """Missing column behaves like None — comparisons against it
    evaluate to False rather than crashing the filter pipeline."""
    fn = compile_filter("nonexistent == 'x'")
    assert not fn({"label": "positive"})


def test_invalid_regex_raises_eval_error():
    fn = compile_filter("contains_pattern(text, '[unclosed')")
    with pytest.raises(FilterEvalError, match="invalid regex"):
        fn({"text": "anything"})


# ─── matches() convenience ─────────────────────────────────────────


def test_matches_passes_none_filter():
    assert matches(None, {"any": "row"})


def test_matches_swallows_runtime_errors():
    """matches() returns False on eval errors (a single bad row should
    not kill the curate pipeline)."""
    assert not matches("contains_pattern(text, '[unclosed')", {"text": "x"})
