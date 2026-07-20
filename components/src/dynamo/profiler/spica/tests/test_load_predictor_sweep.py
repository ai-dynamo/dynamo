# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure (no-dynamo) tests for the load-predictor sweep: loss math + the
trigger paths that return before touching the real predictors."""

import math

import pytest

import dynamo.profiler.spica.load_predictor_sweep as lps
from dynamo.profiler.spica import SmartSearchConfig, sweep_load_predictor, window_loss
from dynamo.profiler.spica.load_predictor_sweep import (
    Window,
    _entry_label,
    _internal_preset,
    evaluate_preset,
    predictor_fields,
)


def _cfg(**search_space):
    ss = {"model_name": "m", "hardware_sku": "h200_sxm", **search_space}
    return SmartSearchConfig(
        search_space=ss,
        workload={
            "isl": 4000,
            "osl": 1000,
            "request_rate": 25,
            "num_request_ratio": 40,
        },  # static
    )


def _trace_cfg(**search_space):
    ss = {"model_name": "m", "hardware_sku": "h200_sxm", **search_space}
    return SmartSearchConfig(search_space=ss, workload={"trace_path": "ignored.jsonl"})


def test_window_loss_zero_on_exact_match():
    assert window_loss(3, 100, 10, 3, 100, 10) == 0.0


def test_window_loss_zero_on_empty_window_predicted_zero():
    assert window_loss(0, 0, 0, 0, 0, 0) == 0.0


def test_window_loss_penalizes_overprediction():
    # over-predicting num_req inflates N*I and N*O
    assert window_loss(6, 100, 10, 3, 100, 10) > 0.0
    # predicting traffic during an actual lull is penalized
    assert window_loss(5, 100, 10, 0, 0, 0) > 0.0


def test_window_loss_weighting_isolates_isl_term():
    # With n_hat == n == 1 and o_hat == o, only the prefill product (N*I) and the
    # standalone isl term carry the i_hat != i error; the decode (N*O) and osl
    # terms vanish. The result is then (0.4 + 0.1) * |log1p(i_hat) - log1p(i)|.
    loss = window_loss(n_hat=1, i_hat=200, o_hat=10, n=1, i=100, o=10)
    isl_err = abs(math.log1p(200) - math.log1p(100))
    assert loss == pytest.approx((0.4 + 0.1) * isl_err)


def test_sweep_skips_when_no_throughput_policy():
    r = sweep_load_predictor(_cfg(planner_scaling_policy=["disabled", "load_180_5"]))
    assert r.reason == "no_throughput_scaling_candidate"
    assert r.best_by_interval == {}


def test_sweep_static_workload_uses_constant_per_interval():
    r = sweep_load_predictor(
        _cfg(planner_scaling_policy=["throughput_180_5", "hybrid_600_5"])
    )
    assert r.reason == "static_workload_constant"
    assert r.best_by_interval == {180: "constant_last", 600: "constant_last"}


# --- load_predictor_candidates as raw dicts (unrolled names) ---


def test_internal_preset_from_id_or_dict():
    # preset id -> the internal preset dict verbatim
    assert _internal_preset("prophet_w20_log1p") == {
        "family": "prophet",
        "log1p": True,
        "prophet_window_size": 20,
    }
    # custom dict (unrolled names) -> internal names + family defaults filled
    internal = _internal_preset(
        {
            "load_predictor": "kalman",
            "load_predictor_log1p": True,
            "kalman_q_level": 3.0,
        }
    )
    assert internal["family"] == "kalman" and internal["log1p"] is True
    assert internal["q_level"] == 3.0 and internal["min_points"] == 5  # default


def test_internal_preset_rejects_unknown_family():
    with pytest.raises(ValueError, match="load_predictor must be one of"):
        _internal_preset({"load_predictor": "bogus"})


def test_predictor_fields_dict_emits_family_knobs_only():
    fields = predictor_fields({"load_predictor": "prophet", "prophet_window_size": 30})
    assert fields == {
        "load_predictor": "prophet",
        "load_predictor_log1p": False,
        "prophet_window_size": 30,
    }
    assert "kalman_q_level" not in fields  # only the chosen family's knobs


def test_entry_label_ids_dicts_by_index():
    assert _entry_label("prophet_w20_raw", 0) == "prophet_w20_raw"
    assert _entry_label({"load_predictor": "kalman"}, 3) == "custom_3"


# --- evaluate_preset over a hand-built window list (fake predictors) ---


class _LastValuePredictor:
    """Deterministic constant-last stand-in for a dynamo predictor: predicts the
    most recent observed value (0.0 before any data). Lets us hand-verify
    ``evaluate_preset`` without prophet/pmdarima/filterpy."""

    def __init__(self, _config, minimum_data_points=1):
        self.minimum_data_points = minimum_data_points
        self._last = 0.0

    def add_data_point(self, value: float) -> None:
        self._last = value

    def get_last_value(self) -> float:
        return self._last

    def predict_next(self) -> float:
        return self._last


def test_evaluate_preset_constant_last_over_window_list(monkeypatch):
    # Inject a constant-last predictor so the forecast for window t is window t-1.
    monkeypatch.setattr(
        lps, "_load_predictors", lambda: {"constant": _LastValuePredictor}
    )
    windows = [Window(2, 100, 10), Window(2, 100, 10), Window(4, 200, 20)]
    preset = {"family": "constant", "log1p": False}
    # warmup=1: skip window 0 (predicted from empty history). Window 1 is predicted
    # from window 0 (exact match -> loss 0). Window 2 is predicted from window 1.
    loss = evaluate_preset(windows, preset, interval_s=180, warmup=1)
    w1 = window_loss(2, 100, 10, 2, 100, 10)  # 0.0
    w2 = window_loss(2, 100, 10, 4, 200, 20)
    assert loss == pytest.approx((w1 + w2) / 2)


def test_evaluate_preset_empty_windows_is_inf(monkeypatch):
    monkeypatch.setattr(
        lps, "_load_predictors", lambda: {"constant": _LastValuePredictor}
    )
    # No windows at/after warmup -> no scored windows -> inf (every preset ties).
    assert (
        evaluate_preset([], {"family": "constant", "log1p": False}, 180, warmup=0)
        == math.inf
    )


# --- short/empty-trace fallback: best_by_interval is never None ---


def test_sweep_falls_back_to_constant_last_when_no_winner(monkeypatch):
    # Short/empty trace: every preset ties at inf loss, so the per-interval best
    # stays None during scoring. The sweep must fall back to a defined default
    # instead of leaving best_by_interval[iv] == None (which would silently
    # inject no load predictor under throughput scaling).
    monkeypatch.setattr(
        lps, "build_windows", lambda _path, _iv: []
    )  # empty window list
    monkeypatch.setattr(lps, "_common_warmup", lambda _presets, _iv: 0)
    monkeypatch.setattr(
        lps, "evaluate_preset", lambda *a, **k: math.inf
    )  # all presets tie at inf

    r = sweep_load_predictor(
        _trace_cfg(planner_scaling_policy=["throughput_180_5"]), show_progress=False
    )

    assert r.best_by_interval == {180: "constant_last"}
    assert r.best_by_interval[180] is not None
    assert "no_winner_fallback_constant_last" in r.reason
    assert "180" in r.reason
    # losses are still recorded (all inf) for diagnostics
    assert all(v == math.inf for v in r.losses[180].values())
