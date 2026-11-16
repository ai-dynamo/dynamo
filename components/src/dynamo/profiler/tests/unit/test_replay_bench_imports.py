# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Keep replay benchmark imports usable without the optional AISimulate runtime."""

import builtins
import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
]


@pytest.fixture
def bench_without_aisimulate(monkeypatch):
    """Load the real benchmark with AISimulate blocked and replay siblings stubbed."""
    original_import = builtins.__import__
    attempts = []

    def without_aisimulate(name, *args, **kwargs):
        if name.split(".")[0] in {"aisimulate", "aisimulate_core"}:
            attempts.append(name)
            raise ModuleNotFoundError("AISimulate is unavailable", name=name)
        return original_import(name, *args, **kwargs)

    # Block imports even when another test has already loaded AISimulate.
    monkeypatch.setattr(builtins, "__import__", without_aisimulate)

    # Isolate bench.py from sibling modules that require Dynamo's native runtime.
    # Its source and AISimulate import remain real; no package initializer runs.
    package = "_replay_bench_import_test"
    dependencies = {
        "scoring": ("_pick_best_record",),
        "search": (
            "optimize_dense_agg_with_replay",
            "optimize_dense_disagg_with_replay",
        ),
        "specs": ("ReplayOptimizeSpec",),
    }
    for sibling, names in dependencies.items():
        module = ModuleType(f"{package}.{sibling}")
        for name in names:
            setattr(module, name, None)
        monkeypatch.setitem(sys.modules, module.__name__, module)

    path = Path(__file__).resolve().parents[2] / "utils/replay_optimize/bench.py"
    spec = importlib.util.spec_from_file_location(f"{package}.bench", path)
    assert spec is not None and spec.loader is not None
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench, attempts


def test_aic_comparison_requires_aisimulate_at_call_time(bench_without_aisimulate):
    bench, attempts = bench_without_aisimulate
    assert attempts == []
    spec = SimpleNamespace(
        workload=SimpleNamespace(isTraceBased=False, requestCount=1, concurrency=1)
    )

    with pytest.raises(ModuleNotFoundError, match="AISimulate is unavailable") as exc:
        bench.compare_aic_and_replay_disagg(spec)

    assert exc.value.name is not None
    assert exc.value.name.split(".")[0] == "aisimulate"
    assert attempts == [exc.value.name]
