# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner load-predictor pre-sweep used by the Sweeper sweep configuration provider."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import TYPE_CHECKING, Any, cast

from tqdm import tqdm  # type: ignore[import-untyped]

from .presets import throughput_intervals

if TYPE_CHECKING:
    from dynamo.planner.config.planner_config import PlannerConfig

LOAD_PREDICTOR_PRESETS: dict[str, dict[str, Any]] = {
    "constant_last": {"family": "constant", "log1p": False},
    "arima_raw": {"family": "arima", "log1p": False},
    "arima_log1p": {"family": "arima", "log1p": True},
    "prophet_w20_raw": {
        "family": "prophet",
        "log1p": False,
        "prophet_window_size": 20,
    },
    "prophet_w20_log1p": {
        "family": "prophet",
        "log1p": True,
        "prophet_window_size": 20,
    },
    "prophet_w50_raw": {
        "family": "prophet",
        "log1p": False,
        "prophet_window_size": 50,
    },
    "prophet_w50_log1p": {
        "family": "prophet",
        "log1p": True,
        "prophet_window_size": 50,
    },
    "kalman_default_raw": {
        "family": "kalman",
        "log1p": False,
        "q_level": 1.0,
        "q_trend": 0.1,
        "r": 10.0,
        "min_points": 5,
    },
    "kalman_default_log1p": {
        "family": "kalman",
        "log1p": True,
        "q_level": 1.0,
        "q_trend": 0.1,
        "r": 10.0,
        "min_points": 5,
    },
    "kalman_reactive_raw": {
        "family": "kalman",
        "log1p": False,
        "q_level": 10.0,
        "q_trend": 1.0,
        "r": 5.0,
        "min_points": 3,
    },
    "kalman_reactive_log1p": {
        "family": "kalman",
        "log1p": True,
        "q_level": 10.0,
        "q_trend": 1.0,
        "r": 5.0,
        "min_points": 3,
    },
}

_DEFAULTS = {
    "prophet_window_size": 50,
    "q_level": 1.0,
    "q_trend": 0.1,
    "r": 10.0,
    "min_points": 5,
}
_DEFAULT_PRESET = "constant_last"
_VALID_FAMILIES = frozenset(
    preset["family"] for preset in LOAD_PREDICTOR_PRESETS.values()
)


def complete_predictor_preset(entry: str | dict[str, Any]) -> dict[str, Any]:
    """Expand one predictor preset into every public predictor knob."""

    if isinstance(entry, str):
        preset = LOAD_PREDICTOR_PRESETS[entry]
        family = preset["family"]
        log1p = preset["log1p"]
        prophet_window_size = preset.get(
            "prophet_window_size", _DEFAULTS["prophet_window_size"]
        )
        kalman_q_level = preset.get("q_level", _DEFAULTS["q_level"])
        kalman_q_trend = preset.get("q_trend", _DEFAULTS["q_trend"])
        kalman_r = preset.get("r", _DEFAULTS["r"])
        kalman_min_points = preset.get("min_points", _DEFAULTS["min_points"])
    else:
        family = entry["load_predictor"]
        if family not in _VALID_FAMILIES:
            raise ValueError(
                f"load_predictor must be one of {sorted(_VALID_FAMILIES)}, got {family!r}"
            )
        log1p = bool(entry.get("load_predictor_log1p", False))
        prophet_window_size = entry.get(
            "prophet_window_size", _DEFAULTS["prophet_window_size"]
        )
        kalman_q_level = entry.get("kalman_q_level", _DEFAULTS["q_level"])
        kalman_q_trend = entry.get("kalman_q_trend", _DEFAULTS["q_trend"])
        kalman_r = entry.get("kalman_r", _DEFAULTS["r"])
        kalman_min_points = entry.get("kalman_min_points", _DEFAULTS["min_points"])

    return {
        "load_predictor": family,
        "load_predictor_log1p": log1p,
        "prophet_window_size": prophet_window_size,
        "kalman_q_level": kalman_q_level,
        "kalman_q_trend": kalman_q_trend,
        "kalman_r": kalman_r,
        "kalman_min_points": kalman_min_points,
    }


@dataclass(frozen=True)
class Window:
    """One aggregated traffic window."""

    num_req: float
    isl: float
    osl: float


@dataclass
class _PredictorConfig:
    load_predictor_log1p: bool
    prophet_window_size: int
    throughput_adjustment_interval_seconds: int
    kalman_q_level: float
    kalman_q_trend: float
    kalman_r: float
    kalman_min_points: int


@dataclass
class LoadPredictorResult:
    """Best predictor choice and diagnostics for every throughput interval."""

    best_by_interval: dict[int, str | dict[str, Any]] = field(default_factory=dict)
    losses: dict[int, dict[str, float]] = field(default_factory=dict)
    reason: str = ""

    def to_state(self) -> dict[str, Any]:
        """Return JSON-shaped adapter prepared state."""

        return {
            "best_by_interval": {
                str(interval): entry
                for interval, entry in self.best_by_interval.items()
            },
            "losses": {
                str(interval): {
                    label: loss if math.isfinite(loss) else None
                    for label, loss in values.items()
                }
                for interval, values in self.losses.items()
            },
            "reason": self.reason,
        }


def _internal_preset(entry: str | dict[str, Any]) -> dict[str, Any]:
    complete = complete_predictor_preset(entry)
    return {
        "family": complete["load_predictor"],
        "log1p": complete["load_predictor_log1p"],
        "prophet_window_size": complete["prophet_window_size"],
        "q_level": complete["kalman_q_level"],
        "q_trend": complete["kalman_q_trend"],
        "r": complete["kalman_r"],
        "min_points": complete["kalman_min_points"],
    }


def predictor_fields(entry: str | dict[str, Any]) -> dict[str, Any]:
    """Expand a winner into concrete PlannerConfig predictor fields."""

    preset = _internal_preset(entry)
    family = preset["family"]
    fields: dict[str, Any] = {
        "load_predictor": family,
        "load_predictor_log1p": preset["log1p"],
    }
    if family == "prophet":
        fields["prophet_window_size"] = preset["prophet_window_size"]
    elif family == "kalman":
        fields.update(
            kalman_q_level=preset["q_level"],
            kalman_q_trend=preset["q_trend"],
            kalman_r=preset["r"],
            kalman_min_points=preset["min_points"],
        )
    return fields


def build_windows(trace_path: str, interval_s: int) -> list[Window]:
    """Aggregate a Mooncake trace using the Planner's production trace utility."""

    from dynamo.planner.offline.trace_data import extract_metrics_from_mooncake

    return [
        Window(
            float(metrics["request_count"]),
            float(metrics["avg_isl"]),
            float(metrics["avg_osl"]),
        )
        for metrics in extract_metrics_from_mooncake(trace_path, interval_s)
    ]


def _error(predicted: float, actual: float) -> float:
    return abs(math.log1p(max(predicted, 0.0)) - math.log1p(max(actual, 0.0)))


def window_loss(
    n_hat: float,
    i_hat: float,
    o_hat: float,
    num_req: float,
    isl: float,
    osl: float,
) -> float:
    """Compute the existing weighted one-step-ahead forecast loss."""

    return (
        0.4 * _error(n_hat * i_hat, num_req * isl)
        + 0.4 * _error(n_hat * o_hat, num_req * osl)
        + 0.1 * _error(i_hat, isl)
        + 0.1 * _error(o_hat, osl)
    )


def _make_config(preset: dict[str, Any], interval_s: int) -> _PredictorConfig:
    return _PredictorConfig(
        load_predictor_log1p=preset["log1p"],
        prophet_window_size=preset.get(
            "prophet_window_size", _DEFAULTS["prophet_window_size"]
        ),
        throughput_adjustment_interval_seconds=interval_s,
        kalman_q_level=preset.get("q_level", _DEFAULTS["q_level"]),
        kalman_q_trend=preset.get("q_trend", _DEFAULTS["q_trend"]),
        kalman_r=preset.get("r", _DEFAULTS["r"]),
        kalman_min_points=preset.get("min_points", _DEFAULTS["min_points"]),
    )


def _new_predictors(preset: dict[str, Any], interval_s: int):
    # Forecasting packages are optional Dynamo simulation dependencies. Keep
    # them out of adapter discovery and disabled/load-only policy paths.
    from dynamo.planner.core.load.predictors import LOAD_PREDICTORS

    predictor_class = LOAD_PREDICTORS[preset["family"]]
    config = cast("PlannerConfig", _make_config(preset, interval_s))
    return predictor_class(config), predictor_class(config), predictor_class(config)


def evaluate_preset(
    windows: list[Window],
    preset: dict[str, Any],
    interval_s: int,
    warmup: int,
) -> float:
    """Return mean one-step-ahead loss using the production predictor cadence."""

    request_predictor, isl_predictor, osl_predictor = _new_predictors(
        preset, interval_s
    )
    losses: list[float] = []
    for index, window in enumerate(windows):
        try:
            n_hat, i_hat, o_hat = (
                request_predictor.predict_next(),
                isl_predictor.predict_next(),
                osl_predictor.predict_next(),
            )
        except Exception:
            n_hat, i_hat, o_hat = (
                request_predictor.get_last_value(),
                isl_predictor.get_last_value(),
                osl_predictor.get_last_value(),
            )
        if index >= warmup:
            losses.append(
                window_loss(
                    n_hat,
                    i_hat,
                    o_hat,
                    window.num_req,
                    window.isl,
                    window.osl,
                )
            )
        request_predictor.add_data_point(window.num_req)
        isl_predictor.add_data_point(window.isl)
        osl_predictor.add_data_point(window.osl)
    return mean(losses) if losses else math.inf


def _common_warmup(entries: list[str | dict[str, Any]], interval_s: int) -> int:
    minimum_points = [
        _new_predictors(_internal_preset(entry), interval_s)[0].minimum_data_points
        for entry in entries
    ]
    return max(minimum_points) if minimum_points else 0


def _entry_label(entry: str | dict[str, Any], index: int) -> str:
    return entry if isinstance(entry, str) else f"custom_{index}"


def sweep_load_predictor(
    *,
    policies: list[str | dict[str, Any]],
    candidates: list[str | dict[str, Any]],
    trace_path: str | None,
    show_progress: bool,
) -> LoadPredictorResult:
    """Choose the best candidate independently for each scaling interval."""

    intervals = throughput_intervals(policies)
    if not intervals:
        return LoadPredictorResult(reason="no_throughput_scaling_candidate")
    if trace_path is None:
        return LoadPredictorResult(
            best_by_interval=dict.fromkeys(intervals, _DEFAULT_PRESET),
            reason="static_workload_constant",
        )

    result = LoadPredictorResult(reason="swept")
    labels = [_entry_label(entry, index) for index, entry in enumerate(candidates)]
    fallback_intervals: list[int] = []
    for interval_s in intervals:
        windows = build_windows(trace_path, interval_s)
        warmup = _common_warmup(candidates, interval_s)
        losses: dict[str, float] = {}
        best_entry: str | dict[str, Any] | None = None
        best_loss = math.inf
        progress = tqdm(
            zip(labels, candidates, strict=True),
            total=len(candidates),
            desc=(
                f"load-predictor @ {interval_s}s "
                f"({len(windows)} windows, warmup {warmup})"
            ),
            unit="preset",
            disable=not show_progress,
        )
        for label, entry in progress:
            loss = evaluate_preset(windows, _internal_preset(entry), interval_s, warmup)
            losses[label] = loss
            progress.set_postfix_str(f"{label}={loss:.3f}")
            if loss < best_loss:
                best_loss = loss
                best_entry = entry
        result.losses[interval_s] = losses
        if best_entry is None:
            best_entry = _DEFAULT_PRESET
            fallback_intervals.append(interval_s)
        result.best_by_interval[interval_s] = best_entry
    if fallback_intervals:
        result.reason = (
            f"swept; no_winner_fallback_{_DEFAULT_PRESET}@{fallback_intervals}"
        )
    return result
