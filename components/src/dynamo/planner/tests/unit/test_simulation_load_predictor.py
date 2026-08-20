# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Golden parity tests for the Planner adapter's load-predictor pre-sweep."""

import math

import pytest

pytest.importorskip(
    "aisimulate.sweeper",
    reason="AI Simulate is an optional Dynamo simulation dependency",
)

import dynamo.planner.simulation.load_predictor as load_predictor
from dynamo.planner.simulation.load_predictor import (
    Window,
    _entry_label,
    _internal_preset,
    complete_predictor_preset,
    evaluate_preset,
    predictor_fields,
    sweep_load_predictor,
    window_loss,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_window_loss_golden_values() -> None:
    assert window_loss(3, 100, 10, 3, 100, 10) == 0.0
    assert window_loss(0, 0, 0, 0, 0, 0) == 0.0
    assert window_loss(6, 100, 10, 3, 100, 10) > 0.0
    assert window_loss(5, 100, 10, 0, 0, 0) > 0.0

    loss = window_loss(1, 200, 10, 1, 100, 10)
    isl_error = abs(math.log1p(200) - math.log1p(100))
    assert loss == pytest.approx(0.5 * isl_error)


def test_static_and_disabled_trigger_paths() -> None:
    disabled = sweep_load_predictor(
        policies=["disabled", "load_180_5"],
        candidates=["constant_last"],
        trace_path=None,
        show_progress=False,
    )
    assert disabled.reason == "no_throughput_scaling_candidate"
    assert disabled.best_by_interval == {}

    static = sweep_load_predictor(
        policies=["throughput_180_5", "hybrid_600_5"],
        candidates=["constant_last"],
        trace_path=None,
        show_progress=False,
    )
    assert static.reason == "static_workload_constant"
    assert static.best_by_interval == {
        180: "constant_last",
        600: "constant_last",
    }


def test_predictor_search_values_decode_with_family_defaults() -> None:
    assert _internal_preset("prophet_w20_log1p") == {
        "family": "prophet",
        "log1p": True,
        "prophet_window_size": 20,
        "q_level": 1.0,
        "q_trend": 0.1,
        "r": 10.0,
        "min_points": 5,
    }
    internal = _internal_preset(
        {
            "load_predictor": "kalman",
            "load_predictor_log1p": True,
            "kalman_q_level": 3.0,
        }
    )
    assert internal["family"] == "kalman"
    assert internal["log1p"] is True
    assert internal["q_level"] == 3.0
    assert internal["min_points"] == 5

    with pytest.raises(ValueError, match="load_predictor must be one of"):
        _internal_preset({"load_predictor": "bogus"})


def test_every_named_predictor_preset_covers_every_knob() -> None:
    expected = {
        "load_predictor",
        "load_predictor_log1p",
        "prophet_window_size",
        "kalman_q_level",
        "kalman_q_trend",
        "kalman_r",
        "kalman_min_points",
    }

    for name in load_predictor.LOAD_PREDICTOR_PRESETS:
        assert set(complete_predictor_preset(name)) == expected


def test_predictor_fields_emit_only_selected_family() -> None:
    assert predictor_fields(
        {"load_predictor": "prophet", "prophet_window_size": 30}
    ) == {
        "load_predictor": "prophet",
        "load_predictor_log1p": False,
        "prophet_window_size": 30,
    }
    assert _entry_label("prophet_w20_raw", 0) == "prophet_w20_raw"
    assert _entry_label({"load_predictor": "kalman"}, 3) == "custom_3"


class _LastValuePredictor:
    minimum_data_points = 1

    def __init__(self) -> None:
        self._last = 0.0

    def add_data_point(self, value: float) -> None:
        self._last = value

    def get_last_value(self) -> float:
        return self._last

    def predict_next(self) -> float:
        return self._last


def test_evaluate_preset_preserves_one_step_ahead_cadence(monkeypatch) -> None:
    monkeypatch.setattr(
        load_predictor,
        "_new_predictors",
        lambda _preset, _interval: (
            _LastValuePredictor(),
            _LastValuePredictor(),
            _LastValuePredictor(),
        ),
    )
    windows = [Window(2, 100, 10), Window(2, 100, 10), Window(4, 200, 20)]
    loss = evaluate_preset(
        windows,
        {"family": "constant", "log1p": False},
        interval_s=180,
        warmup=1,
    )
    expected = (
        window_loss(2, 100, 10, 2, 100, 10) + window_loss(2, 100, 10, 4, 200, 20)
    ) / 2
    assert loss == pytest.approx(expected)
    assert (
        evaluate_preset(
            [],
            {"family": "constant", "log1p": False},
            interval_s=180,
            warmup=0,
        )
        == math.inf
    )


def test_short_trace_falls_back_to_constant_last(monkeypatch) -> None:
    monkeypatch.setattr(load_predictor, "build_windows", lambda _path, _iv: [])
    monkeypatch.setattr(load_predictor, "_common_warmup", lambda _values, _iv: 0)
    monkeypatch.setattr(
        load_predictor, "evaluate_preset", lambda *args, **kwargs: math.inf
    )

    result = sweep_load_predictor(
        policies=["throughput_180_5"],
        candidates=["constant_last", "kalman_default_raw"],
        trace_path="empty.jsonl",
        show_progress=False,
    )

    assert result.best_by_interval == {180: "constant_last"}
    assert "no_winner_fallback_constant_last" in result.reason
    assert all(math.isinf(value) for value in result.losses[180].values())
