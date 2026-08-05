# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep the simulation readable by its own extractor.

The simulation can be profiled with nsys and the resulting sqlite fed back
through :mod:`egress_experiments.capture_params` -- the same extractor used on
the real 355778 capture. That round trip is what makes the reported numbers
checkable rather than self-asserted, and it depends entirely on the NVTX range
**names** matching. Rename one and the extractor silently reports zero for that
stage; nothing raises.

So this pins the contract statically: every name ``capture_params`` looks for
must be emitted somewhere. No nsys, no ``nvtx`` wheel, no profiling required --
it reads the sources.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from egress_experiments import capture_params
from egress_experiments.dynamo_sim import realcode

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.none,
]

_PKG = Path(capture_params.__file__).resolve().parent

#: Files the simulation emits ranges from.
_SIM_SOURCES = (
    _PKG / "fake_trtllm" / "result.py",
    _PKG / "fake_trtllm" / "llm.py",
    _PKG / "dynamo_sim" / "worker.py",
)


def _string_literals(path: Path) -> set[str]:
    """Every non-docstring string constant in a module.

    AST rather than a regex on ``range_("...")``: the four ingress stages are
    emitted from a loop over a tuple of ``(name, cost)`` pairs, so the names
    never appear adjacent to the call. Docstrings are excluded so a name that
    survives only in prose cannot mask a real miss.
    """
    tree = ast.parse(path.read_text())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            doc = ast.get_docstring(node, clean=False)
            if doc is not None:
                docstrings.add(doc)
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value not in docstrings
    }


def _emitted_names() -> set[str]:
    names: set[str] = set()
    for path in _SIM_SOURCES:
        names |= _string_literals(path)
    # trtllm:push_send is not ours -- it comes from the shipped push_egress.py,
    # which is exactly why the round trip is meaningful.
    module = realcode.load_push_egress()
    if module is not None and getattr(module, "__file__", None):
        names |= _string_literals(Path(module.__file__))
    return names


def test_every_range_capture_params_reads_is_actually_emitted():
    required = (
        set(capture_params.LOOP_STAGES)
        | set(capture_params.INGRESS_STAGES)
        | {capture_params.ITERATION_RANGE, capture_params.REQUEST_RANGE}
        | {"trtllm:engine_submit"}
    )
    missing = required - _emitted_names()
    assert not missing, (
        f"capture_params reads {sorted(missing)} but the simulation never emits "
        "them; a profiled run would report 0 for those stages instead of failing"
    )


def test_push_send_comes_from_the_shipped_driver_not_from_us():
    """If we ever emitted it ourselves the round trip would prove nothing."""
    ours: set[str] = set()
    for path in _SIM_SOURCES:
        ours |= _string_literals(path)
    assert "trtllm:push_send" not in ours

    module = realcode.load_push_egress()
    assert module is not None
    shipped = _string_literals(Path(module.__file__))
    assert "trtllm:push_send" in shipped


def test_nvtx_is_off_unless_asked_for(monkeypatch):
    """DYN_NVTX unset must leave the simulation exactly as it was."""
    from egress_experiments import nvtx_shim

    monkeypatch.delenv("DYN_NVTX", raising=False)
    assert nvtx_shim.enabled() is False

    # And the context manager stays a no-op that returns cleanly.
    with nvtx_shim.range_("probe:test", color="blue"):
        pass


def test_unclosed_ranges_are_dropped_from_the_mean():
    """A profiler stopped mid-range leaves an end timestamp far in the future.

    The real capture has this on ``trtllm:generate_locally`` and
    ``trtllm:push_egress`` (means of ~97,000 s); a windowed capture of the
    simulation gets it on whichever range was open when collection stopped.
    p50 shrugs it off, the mean does not.
    """
    assert capture_params._UNCLOSED_NS == 1_000_000_000
