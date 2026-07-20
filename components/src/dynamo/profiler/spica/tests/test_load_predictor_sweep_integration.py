# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402

"""Integration tests that exercise the real dynamo planner predictors and the
densify-fixed trace->window tool. Skipped unless the ``dynamo`` extra is
installed (Rust runtime + prophet/pmdarima/filterpy)."""

import json

import pytest

pytest.importorskip("dynamo.planner.core.load.predictors")
pytest.importorskip("dynamo.planner.offline.trace_data")

from dynamo.profiler.spica import SmartSearchConfig, sweep_load_predictor
from dynamo.profiler.spica.load_predictor_sweep import build_windows


def _trace(tmp_path, rows):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        "\n".join(
            json.dumps({"timestamp": ts, "input_length": i, "output_length": o})
            for ts, i, o in rows
        )
        + "\n"
    )
    return str(path)


def test_build_windows_densifies_middle_gaps(tmp_path):
    # interval 10s; activity at interval 0 and 30 -> empties at 10, 20.
    w = build_windows(_trace(tmp_path, [(0, 100, 10), (35_000, 500, 50)]), 10)
    assert [x.num_req for x in w] == [1, 0, 0, 1]
    assert w[0].isl == 100 and w[3].isl == 500
    assert w[1].isl == 0 and w[2].num_req == 0


def test_sweep_constant_traffic_constant_predictor_is_optimal(tmp_path):
    # 12 identical 180s windows (3 reqs each, isl=100, osl=10). The constant
    # predictor forecasts the (constant) next window exactly -> zero loss.
    rows = []
    for win in range(12):
        base = win * 180_000  # ms
        rows += [(base + k * 1000, 100, 10) for k in range(3)]

    cfg = SmartSearchConfig(
        search_space={
            "model_name": "m",
            "hardware_sku": "h200_sxm",
            "planner_scaling_policy": ["throughput_180_5"],
            # keep the grid fast (skip prophet/arima model fits) while still
            # exercising multi-preset selection per interval
            "load_predictor_candidates": ["constant_last", "kalman_default_raw"],
        },
        workload={"trace_path": _trace(tmp_path, rows), "trace_format": "mooncake"},
    )

    r = sweep_load_predictor(cfg)
    assert r.reason == "swept"
    assert set(r.best_by_interval) == {180}
    assert r.losses[180]["constant_last"] == pytest.approx(0.0, abs=1e-9)
    # the winner is at least as good as the exact constant predictor
    assert r.losses[180][r.best_by_interval[180]] == pytest.approx(0.0, abs=1e-6)
