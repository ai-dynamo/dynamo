# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Assertion evaluation engine for testbed scenarios.

Supports the structured form (field / op / value / tolerance / ref) and the
expression fallback form (expr with restricted AST eval).

Structured predicates:
  ``at_tick: N``          — evaluated at exactly tick N
  ``always:``             — must hold at every tick
  ``eventually_by_tick: N`` — must hold at some tick ≤ N

Expression predicates use ``ast.literal_eval``-restricted eval with:
  ``history``  — list of TickSnapshot objects
  ``planner``  — PlannerSpec dict
  ``counters`` — dict of cumulative counter deltas

Assertion failure raises AssertionError with a descriptive message.
"""

from __future__ import annotations

import ast
import math
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from dynamo.planner.tests.testbed.recorder import TickHistory, TickSnapshot
    from dynamo.planner.tests.testbed.scenarios import (
        ExprAssertion,
        ScenarioSpec,
        StructuredAssertion,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ALLOWED_NAMES = frozenset(
    {"history", "planner", "counters", "abs", "min", "max", "len"}
)
_ALLOWED_NODE_TYPES = (
    ast.Expression,
    ast.BoolOp,
    ast.UnaryOp,
    ast.BinOp,
    ast.Compare,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.Index,
    ast.Name,
    ast.Constant,  # ast.Num/ast.Str deprecated in 3.8, removed in 3.14
    ast.And,
    ast.Or,
    ast.Not,
    ast.Gt,
    ast.GtE,
    ast.Lt,
    ast.LtE,
    ast.Eq,
    ast.NotEq,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.Mod,
    ast.USub,
    ast.UAdd,
    ast.Load,
)


def _safe_eval(expr: str, context: dict[str, Any]) -> Any:
    """Evaluate expression in restricted AST context."""
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise ValueError(
                f"Disallowed AST node type {type(node).__name__} in expr: {expr!r}"
            )
        if (
            isinstance(node, ast.Name)
            and node.id not in _ALLOWED_NAMES
            and node.id not in context
        ):
            raise ValueError(f"Unknown name {node.id!r} in expr: {expr!r}")
    return eval(compile(tree, "<assertion>", "eval"), {"__builtins__": {}}, context)


def _get_field(snap: "TickSnapshot", field: str) -> Any:
    return getattr(snap, field)


def _get_ref(ref: str, scenario: "ScenarioSpec") -> float:
    parts = ref.split(".")
    if parts[0] == "planner":
        return float(getattr(scenario.planner, parts[1]))
    raise ValueError(f"Unknown ref prefix: {ref!r}")


def _apply_op(actual: float, op: str, expected: float, tolerance: float = 0.0) -> bool:
    if op == "<":
        return actual < expected
    elif op == "<=":
        return actual <= expected
    elif op == "==":
        return math.isclose(actual, expected, rel_tol=1e-6)
    elif op == ">=":
        return actual >= expected
    elif op == ">":
        return actual > expected
    elif op == "!=":
        return not math.isclose(actual, expected, rel_tol=1e-6)
    elif op == "within":
        return abs(actual - expected) <= tolerance * max(1.0, abs(expected))
    return False


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------


def evaluate_all(
    history: "TickHistory",
    scenario: "ScenarioSpec",
    counters: Optional[dict[str, Any]] = None,
) -> list[str]:
    """Evaluate all scenario assertions against the recorded history.

    Returns a list of failure messages (empty list = all passed).

    ``counters`` is optional and exposed as ``counters`` inside ``expr:``
    assertions; α-runner currently doesn't track external counter deltas so
    the default empty dict is fine.
    """
    failures: list[str] = []
    parsed = scenario.parsed_assertions()
    counters = counters or {}

    for assertion in parsed:
        from dynamo.planner.tests.testbed.scenarios import (
            ExprAssertion,
            StructuredAssertion,
        )

        if isinstance(assertion, StructuredAssertion):
            _eval_structured(assertion, history, scenario, counters, failures)
        elif isinstance(assertion, ExprAssertion):
            _eval_expr(assertion, history, scenario, counters, failures)

    return failures


# Back-compat alias for older imports; remove once no callers remain.
evaluate_assertions = evaluate_all


def _eval_structured(
    a: "StructuredAssertion",
    history: "TickHistory",
    scenario: "ScenarioSpec",
    counters: dict[str, Any],
    failures: list[str],
) -> None:
    desc = a.description or f"field={a.field} op={a.op} value={a.value}"

    # Resolve expected value
    if a.value is not None:
        expected = a.value
    elif a.ref is not None:
        try:
            expected = _get_ref(a.ref, scenario)
        except Exception as e:
            failures.append(f"[{desc}] Could not resolve ref {a.ref!r}: {e}")
            return
    else:
        failures.append(f"[{desc}] Assertion has neither value nor ref")
        return

    tolerance = a.tolerance or 0.0

    # Select ticks to evaluate
    if a.at_tick is not None:
        tick_idx = a.at_tick
        if tick_idx >= len(history):
            failures.append(
                f"[{desc}] at_tick={tick_idx} but history only has {len(history)} ticks"
            )
            return
        snaps = [history[tick_idx]]
        _check_snaps(
            snaps, a.field, a.op, expected, tolerance, desc, failures, mode="at_tick"
        )

    elif a.always is True:
        snaps = list(history.snapshots)
        _check_snaps(
            snaps, a.field, a.op, expected, tolerance, desc, failures, mode="always"
        )

    elif a.eventually_by_tick is not None:
        limit = min(a.eventually_by_tick, len(history) - 1)
        snaps = history.snapshots[: limit + 1]
        _check_snaps_eventually(
            snaps, a.field, a.op, expected, tolerance, desc, failures
        )


def _check_snaps(
    snaps: list["TickSnapshot"],
    field: str,
    op: str,
    expected: float,
    tolerance: float,
    desc: str,
    failures: list[str],
    mode: str,
) -> None:
    for snap in snaps:
        actual = _get_field(snap, field)
        if not _apply_op(float(actual), op, expected, tolerance):
            failures.append(
                f"[{desc}] FAIL at tick {snap.tick}: {field}={actual!r} {op} {expected}"
                + (f" ±{tolerance * 100:.0f}%" if op == "within" else "")
                + f" ({mode})"
            )
            if mode == "at_tick":
                return


def _check_snaps_eventually(
    snaps: list["TickSnapshot"],
    field: str,
    op: str,
    expected: float,
    tolerance: float,
    desc: str,
    failures: list[str],
) -> None:
    for snap in snaps:
        actual = _get_field(snap, field)
        if _apply_op(float(actual), op, expected, tolerance):
            return
    if snaps:
        last = snaps[-1]
        failures.append(
            f"[{desc}] FAIL: {field} never satisfied {op} {expected} by tick {last.tick}"
        )
    else:
        failures.append(f"[{desc}] FAIL: no snapshots to check")


def _eval_expr(
    a: "ExprAssertion",
    history: "TickHistory",
    scenario: "ScenarioSpec",
    counters: dict[str, Any],
    failures: list[str],
) -> None:
    desc = a.description or f"expr={a.expr!r}"
    context: dict[str, Any] = {
        "history": history.snapshots,
        "planner": scenario.planner.model_dump(),
        "counters": counters,
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
    }

    if a.at_tick is not None:
        tick_idx = a.at_tick
        if tick_idx >= len(history):
            failures.append(
                f"[{desc}] at_tick={tick_idx} but history only has {len(history)} ticks"
            )
            return
        _eval_expr_at(a.expr, context, desc, failures, tick_idx)

    elif a.always is True:
        for snap in history.snapshots:
            context["history"] = history.snapshots
            if not _eval_expr_at(
                a.expr, context, desc, failures, snap.tick, silent=True
            ):
                failures.append(f"[{desc}] FAIL at tick {snap.tick}")

    elif a.eventually_by_tick is not None:
        for snap in history.snapshots[: a.eventually_by_tick + 1]:
            try:
                if _safe_eval(a.expr, context):
                    return
            except Exception:
                pass
        failures.append(
            f"[{desc}] FAIL: expression never true by tick {a.eventually_by_tick}"
        )


def _eval_expr_at(
    expr: str,
    context: dict[str, Any],
    desc: str,
    failures: list[str],
    tick: int,
    silent: bool = False,
) -> bool:
    try:
        result = _safe_eval(expr, context)
        if not result:
            if not silent:
                failures.append(
                    f"[{desc}] FAIL at tick {tick}: expr evaluated to {result!r}"
                )
            return False
        return True
    except Exception as e:
        failures.append(f"[{desc}] ERROR evaluating expr at tick {tick}: {e}")
        return False
