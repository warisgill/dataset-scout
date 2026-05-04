"""A tiny, safe filter expression evaluator (M4b).

The strategy assessor produces filter strings like:

    label == "positive"
    len(text) > 50
    contains_pattern(prompt, "(?i)hidden")
    label != "junk" and len(text) > 30

We evaluate them per source row using Python's `ast` module rather
than a hand-rolled parser: parse to AST, walk an explicit allow-list
of node types, and refuse anything else. This is much safer than
`eval()` and much smaller than dragging in `simpleeval`.

Allowed shape:

- atoms: column-name identifiers, string/int/float/bool/None literals
- comparisons: == != > < >= <=
- boolean ops: and, or
- unary: not, -, +
- function calls (whitelisted): `len(<expr>)`, `contains_pattern(<expr>, <pattern>)`,
  `lower(<expr>)`, `startswith(<expr>, <prefix>)`, `endswith(<expr>, <suffix>)`,
  `int(<expr>)`, `str(<expr>)`

Everything else (attribute access, subscript, comprehensions, lambda,
imports, …) is rejected at compile time.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from typing import Any

from dataset_scout.errors import DatasetScoutError


class FilterCompileError(DatasetScoutError):
    """Raised when a filter expression contains disallowed syntax or names."""


class FilterEvalError(DatasetScoutError):
    """Raised when evaluation fails (rare; usually means the row shape
    doesn't match what the expression assumed)."""


# ─── whitelisted callables ───────────────────────────────────────────


def _fn_len(value: Any) -> int:
    if value is None:
        return 0
    return len(value) if hasattr(value, "__len__") else len(str(value))


def _fn_contains_pattern(value: Any, pattern: str) -> bool:
    if value is None:
        return False
    text = value if isinstance(value, str) else str(value)
    try:
        return re.search(pattern, text) is not None
    except re.error as exc:
        raise FilterEvalError(f"invalid regex {pattern!r}: {exc}") from exc


def _fn_lower(value: Any) -> str:
    return (value if isinstance(value, str) else str(value)).lower()


def _fn_startswith(value: Any, prefix: str) -> bool:
    text = value if isinstance(value, str) else str(value)
    return text.startswith(prefix)


def _fn_endswith(value: Any, suffix: str) -> bool:
    text = value if isinstance(value, str) else str(value)
    return text.endswith(suffix)


_ALLOWED_FUNCTIONS: dict[str, Callable[..., Any]] = {
    "len": _fn_len,
    "contains_pattern": _fn_contains_pattern,
    "lower": _fn_lower,
    "startswith": _fn_startswith,
    "endswith": _fn_endswith,
    "int": int,
    "str": str,
}


# ─── compiler ───────────────────────────────────────────────────────


_ALLOWED_NODES: tuple[type[ast.AST], ...] = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.In,
    ast.NotIn,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.Tuple,
    ast.List,
)


def _validate(node: ast.AST) -> None:
    """Walk the AST and reject anything outside the allow-list."""
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise FilterCompileError(
                f"filter expression contains disallowed syntax "
                f"({type(child).__name__!r}); allowed: comparisons, "
                "boolean operators, arithmetic, and the whitelisted "
                f"functions {sorted(_ALLOWED_FUNCTIONS)}"
            )
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                raise FilterCompileError(
                    f"filter calls must use plain function names (got {ast.dump(child.func)!r})"
                )
            if child.func.id not in _ALLOWED_FUNCTIONS:
                raise FilterCompileError(
                    f"filter function {child.func.id!r} is not allowed; "
                    f"choose from {sorted(_ALLOWED_FUNCTIONS)}"
                )


def compile_filter(expr: str) -> Callable[[dict[str, Any]], bool]:
    """Compile a filter expression into a callable that takes a row dict.

    The compiled callable returns truthy when the row passes the filter.
    Raises `FilterCompileError` at compile time for disallowed syntax;
    `FilterEvalError` at call time for runtime issues (rare).
    """
    if not expr or not expr.strip():
        raise FilterCompileError("empty filter expression")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise FilterCompileError(f"could not parse filter {expr!r}: {exc}") from exc
    _validate(tree)
    code = compile(tree, "<filter>", mode="eval")

    def _evaluate(row: dict[str, Any]) -> bool:
        # Build an exec environment with row columns + allowed functions.
        # Missing column references resolve to None — `column == "x"` is
        # then False rather than raising. Curate's per-row try/except
        # then drops the row cleanly. Other runtime errors (invalid
        # regex, type mismatch) still raise FilterEvalError.
        scope: dict[str, Any] = {**_ALLOWED_FUNCTIONS}
        scope.update(row)
        # Walk the AST looking for column-name references that aren't
        # in the row or in the function whitelist; default each to None.
        for child in ast.walk(tree):
            if isinstance(child, ast.Name) and child.id not in scope:
                scope[child.id] = None
        try:
            return bool(eval(code, {"__builtins__": {}}, scope))
        except FilterEvalError:
            raise
        except Exception as exc:
            raise FilterEvalError(f"filter evaluation failed: {exc}") from exc

    return _evaluate


def matches(expr: str | None, row: dict[str, Any]) -> bool:
    """Convenience: compile + evaluate. None expression always matches.

    Compile errors propagate; runtime errors return False so a single
    bad row doesn't kill the materialise. (Curate's caller logs the
    notice.)
    """
    if expr is None:
        return True
    fn = compile_filter(expr)
    try:
        return fn(row)
    except FilterEvalError:
        return False
