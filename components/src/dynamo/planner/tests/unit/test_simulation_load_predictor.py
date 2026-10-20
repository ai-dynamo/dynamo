# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Optional-dependency preflight must run before the simulation imports.
# ruff: noqa: E402

"""Golden parity tests for the Planner adapter's load-predictor pre-sweep."""

import json
import logging
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
    build_windows_from_trace_paths,
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
    assert static.reason == "static_workload_configured_fallback"
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


class _RaisingPredictor(_LastValuePredictor):
    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self._exc = exc

    def predict_next(self) -> float:
        raise self._exc


def _raising_predictors(exc: Exception):
    return lambda _preset, _interval: (
        _RaisingPredictor(exc),
        _LastValuePredictor(),
        _LastValuePredictor(),
    )


def test_evaluate_preset_disqualifies_a_predictor_that_raises(
    monkeypatch, caplog
) -> None:
    monkeypatch.setattr(
        load_predictor, "_new_predictors", _raising_predictors(ValueError("unfit"))
    )
    windows = [Window(2, 100, 10), Window(2, 100, 10), Window(4, 200, 20)]
    with caplog.at_level(logging.WARNING, logger=load_predictor.__name__):
        loss = evaluate_preset(
            windows, {"family": "kalman", "log1p": False}, interval_s=180, warmup=1
        )
    assert loss == math.inf
    assert "scoring as inf" in caplog.text


def test_evaluate_preset_propagates_unexpected_errors(monkeypatch) -> None:
    monkeypatch.setattr(
        load_predictor, "_new_predictors", _raising_predictors(KeyError("bug"))
    )
    with pytest.raises(KeyError):
        evaluate_preset(
            [Window(2, 100, 10)],
            {"family": "kalman", "log1p": False},
            interval_s=180,
            warmup=0,
        )


def test_sweep_never_selects_a_preset_that_raises(monkeypatch) -> None:
    def new_predictors(preset, _interval):
        if preset["family"] == "kalman":
            return tuple(_RaisingPredictor(ValueError("unfit")) for _ in range(3))
        return tuple(_LastValuePredictor() for _ in range(3))

    monkeypatch.setattr(load_predictor, "_new_predictors", new_predictors)
    monkeypatch.setattr(
        load_predictor, "build_windows", lambda _path, _iv: [Window(2, 100, 10)] * 4
    )

    result = sweep_load_predictor(
        policies=["throughput_180_5"],
        candidates=["kalman_default_raw", "constant_last"],
        trace_path="trace.jsonl",
        show_progress=False,
    )

    assert result.best_by_interval == {180: "constant_last"}
    assert math.isinf(result.losses[180]["kalman_default_raw"])
    assert result.losses[180]["constant_last"] == 0.0
    assert result.reason == "swept"


def test_window_loss_clamps_negative_forecasts_before_products() -> None:
    # A negative forecast scores as a forecast of zero, and two negative
    # forecasts must not multiply into a plausible positive token product.
    assert window_loss(-5, -400, -20, 5, 400, 20) == window_loss(0, 0, 0, 5, 400, 20)
    assert window_loss(-5, -400, 20, 5, 400, 20) > window_loss(-5, 400, 20, 5, 400, 20)


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
    assert "no_winner_configured_fallback" in result.reason
    assert all(math.isinf(value) for value in result.losses[180].values())


def test_dynamo_multi_file_trace_builds_one_global_window_series(tmp_path) -> None:
    def request(request_id: str, timestamp_ms: int, isl: int, osl: int) -> str:
        return json.dumps(
            {
                "schema": "dynamo.request.trace.v1",
                "event_type": "request_end",
                "event_time_unix_ms": timestamp_ms + 10,
                "request": {
                    "request_id": request_id,
                    "request_received_ms": timestamp_ms,
                    "output_tokens": osl,
                    "replay": {
                        "trace_block_size": 64,
                        "input_length": isl,
                        "input_sequence_hashes": [1],
                    },
                },
            }
        )

    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first.write_text(
        json.dumps({"event_type": "request_payload", "request_id": "first"})
        + "\n"
        + request("first", 1_000_000, 100, 10)
        + "\n"
    )
    second.write_text(request("second", 1_180_000, 200, 20) + "\n")

    assert build_windows_from_trace_paths([str(first), str(second)], "dynamo", 180) == [
        Window(1.0, 100.0, 10.0),
        Window(1.0, 200.0, 20.0),
    ]
