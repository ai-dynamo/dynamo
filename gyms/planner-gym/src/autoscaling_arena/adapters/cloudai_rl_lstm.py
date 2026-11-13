# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudAI RL agg LSTM policies (``v2_lstm_current`` / ``v3_lstm_current``).

Serves the recurrent agg checkpoints trained by ``cql_trainer``
(``trainer/agg/policy_raw_lstm.py``): current-interval observations from the
five-second replay telemetry, the last ten of them passed through a 32-unit
LSTM whose final hidden state is concatenated with the latest observation
and fed to the residual Q head (nine relative deltas, fleet 1-8, Q-only). The
leaderboard entry ``cloudai-rl`` (agg) is the ``v3_lstm_current`` export,
``checkpoints/cql_autoscaler_best_v3_lstm.pt``. The v3 observation encoders
(``V3StateEncoder`` with its smoothing switch, ``RawV3Encoder``) and the exact
bounded-target action helpers live here as well.

* ``v2_lstm_current`` (schema ``agg_v2_lstm_current_12``): the compact v2
  measurements recomputed from telemetry (current RPS / 200, current
  completed ISL / 4096 or zero, queue / 50, mean active KV, replicas /
  ``observation_max_replicas`` (32), RPS and ISL trends, a constant 0.5,
  shape validity, ISL-trend validity, interval / 5 s, observation validity).
  Actions keep the v2 interpretation: plain argmax over the nine deltas,
  clamped to the fleet bounds.
* ``v3_lstm_current`` (schema ``agg_v3_lstm_current_27``): the 24 v3 entries
  without smoothing plus the same three validity/interval fields; actions use
  the v3 exact bounded-target mask.

Histories are causal, bounded to the episode (left zero padding, zero hidden
state per window) and include the current observation. The checkpoint's
observation settings (SLO, replica bound, cold start, replica normalization
and per-worker KV capacity) are checked against the serving configuration.
Agg only.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from autoscaling_arena.adapters.cql_common import (
    INTERVAL_S,
    NUM_RELATIVE_ACTIONS,
    RELATIVE_ACTIONS,
    LSTMQNetwork,
    TelemetryPolicy,
    history_window,
    load_checkpoint,
    load_network,
    mismatched_settings,
)

from dynamo.planner.core.types import (
    PlannerEffects,
    ScalingDecision,
    ScheduledTick,
    TickInput,
    WorkerCapabilities,
)

# --- v3 observation contract (port of cql_trainer/trainer/agg/features_v3.py) ---
# The 24 smoothed v3 entries; the LSTM checkpoint observes their current-interval
# values (RawV3Encoder) plus three validity / interval fields.
V3_SMOOTHED_FEATURES = [
    "log_rps_ema",
    "log_completed_isl_ema",
    "log_completed_osl_ema",
    "log_waiting_requests",
    "log_running_requests",
    "mean_active_kv_usage",
    "active_replicas",
    "starting_replicas",
    "draining_replicas",
    "target_replicas",
    "next_startup_eta",
    "mean_startup_eta",
    "rps_trend",
    "isl_trend",
    "log_ttft_slo_ratio",
    "log_itl_slo_ratio",
    "ttft_valid",
    "itl_valid",
    "shape_valid",
    "scheduler_valid",
    "log_completion_rps",
    "log_ttft_count",
    "log_itl_count",
    "log_shape_age",
]


def action_targets(current: int, minimum: int = 1, maximum: int = 8) -> list[int]:
    """Absolute replica target of every relative action, clamped to the bounds."""
    return [max(minimum, min(maximum, current + delta)) for delta in RELATIVE_ACTIONS]


def action_mask(current: int, minimum: int = 1, maximum: int = 8) -> list[bool]:
    """One smallest-magnitude delta per distinct executable target."""
    targets = action_targets(current, minimum, maximum)
    chosen: dict[int, int] = {}
    for index in sorted(
        range(len(targets)), key=lambda i: (abs(RELATIVE_ACTIONS[i]), i)
    ):
        chosen.setdefault(targets[index], index)
    return [index in chosen.values() for index in range(len(targets))]


def _log(value: float, scale: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"Expected finite nonnegative measurement: {value}")
    return math.log1p(value) / math.log1p(scale)


class V3StateEncoder:
    """Causal 24-feature encoder over ``dynamo.replay.telemetry.v1`` samples.

    Port of ``cql_trainer/trainer/agg/features_v3.py::StateEncoder``. Every
    statistic describes past observations only; samples must advance in time.
    ``smooth=True`` (the v3 feedforward checkpoint) keeps EMAs of the arrival
    rate and completed ISL/OSL; ``smooth=False`` (the LSTM checkpoints) uses the
    current interval's values and zero shape when nothing completed.
    """

    def __init__(
        self,
        *,
        ttft_ms: float = 2000.0,
        itl_ms: float = 50.0,
        max_replicas: int = 8,
        cold_start_s: float = 30.0,
        smooth: bool = True,
    ) -> None:
        if not all(
            math.isfinite(value) and value > 0
            for value in (ttft_ms, itl_ms, max_replicas, cold_start_s)
        ):
            raise ValueError("Invalid observation settings")
        self.ttft_ms, self.itl_ms = ttft_ms, itl_ms
        self.max_replicas, self.cold_start_s = max_replicas, cold_start_s
        self.smooth = smooth
        self.last_s: Optional[float] = None
        self.rps = self.isl = self.osl = 0.0
        self.last_shape_s: Optional[float] = None
        self.starting_since: dict[Any, float] = {}

    def observe(self, sample: Mapping[str, Any]) -> list[float]:
        now = sample["sampled_at_ms"] / 1000.0
        if self.last_s is not None and now <= self.last_s:
            raise ValueError("Observations must advance in time")
        dt = now - self.last_s if self.last_s is not None else 0.0
        alpha = 1 - 0.6 ** (dt / INTERVAL_S)
        traffic = sample["traffic"]
        duration = traffic["duration_s"]
        if duration < 0 or not math.isclose(
            duration,
            (sample["sampled_at_ms"] - sample["interval_start_ms"]) / 1000,
            abs_tol=1e-6,
        ):
            raise ValueError("Inconsistent telemetry interval")
        active = set(sample["active_decode_ids"])
        starting = set(sample["starting_decode_ids"])
        draining = set(sample["draining_decode_ids"])
        if active & starting or active & draining or starting & draining:
            raise ValueError("Overlapping worker lifecycle states")
        # At a five-second control cadence, newly observed starting workers were
        # requested at the preceding sample. This is an observable ETA estimate,
        # not a future lifecycle lookup. Offline and live use identical logic.
        self.starting_since = {
            worker: self.starting_since.get(
                worker, self.last_s if self.last_s is not None else now
            )
            for worker in starting
        }
        etas = [
            max(0, self.cold_start_s - (now - birth)) / self.cold_start_s
            for birth in self.starting_since.values()
        ]
        rps = traffic["arriving_requests"] / duration if duration else 0.0
        completed = traffic["completed_requests"]
        old_rps, old_isl = self.rps, self.isl
        self.rps = self.rps + alpha * (rps - self.rps) if self.smooth else rps
        shape_valid = completed > 0
        if shape_valid:
            if self.last_shape_s is None or not self.smooth:
                self.isl, self.osl = traffic["avg_isl"], traffic["avg_osl"]
            else:
                shape_alpha = 1 - 0.6 ** ((now - self.last_shape_s) / INTERVAL_S)
                self.isl += shape_alpha * (traffic["avg_isl"] - self.isl)
                self.osl += shape_alpha * (traffic["avg_osl"] - self.osl)
            self.last_shape_s = now
        elif not self.smooth:
            self.isl = self.osl = 0.0
        age = now - self.last_shape_s if self.last_shape_s is not None else now
        workers = [
            m
            for m in sample["decode_scheduler_metrics"]
            if m["worker_id"] in active | draining
        ]
        queue = (
            sum(m["waiting_requests"] for m in workers)
            + sample["router_pending_decode_requests"]
        )
        running = sum(m["running_requests"] for m in workers)
        kv = (
            sum(m["active_cache_usage"] for m in workers) / len(workers)
            if workers
            else 0.0
        )
        nt, ni = traffic["ttft_count"], traffic["itl_count"]
        trend_scale = INTERVAL_S / dt if dt else 0.0

        def trend(a: float, b: float) -> float:
            return max(-2.0, min(2.0, (a - b) / max(1.0, b) * trend_scale))

        state = [
            _log(self.rps, 20),
            _log(self.isl, 4096),
            _log(self.osl, 1024),
            _log(queue, 50),
            _log(running, 50),
            kv,
            len(active) / self.max_replicas,
            len(starting) / self.max_replicas,
            len(draining) / self.max_replicas,
            max(1, len(active) + len(starting)) / self.max_replicas,
            min(etas) if etas else 0.0,
            sum(etas) / len(etas) if etas else 0.0,
            trend(self.rps, old_rps),
            trend(self.isl, old_isl) if shape_valid else 0.0,
            math.log1p(traffic["avg_ttft_ms"] / self.ttft_ms) if nt else 0.0,
            math.log1p(traffic["avg_itl_ms"] / self.itl_ms) if ni else 0.0,
            float(nt > 0),
            float(ni > 0),
            float(shape_valid),
            float(bool(workers)),
            _log(completed / duration if duration else 0, 20),
            _log(nt, 50),
            _log(ni, 50),
            _log(age, 60),
        ]
        if not all(math.isfinite(value) for value in state):
            raise ValueError("Nonfinite v3 observation")
        self.last_s = now
        return state


class RawV3Encoder(V3StateEncoder):
    """Current-window v3 values with explicit validity and no trends across gaps.

    Port of ``trainer/agg/features_raw_lstm.py::RawV3Encoder``: the 24 v3
    entries computed without smoothing, the ISL trend zeroed unless both this
    and the previous sample had completions, plus ``isl_trend_valid``,
    ``interval_over_5s`` and the ``observation_valid`` flag that marks real
    rows inside a left-padded LSTM window (27 entries).
    """

    def __init__(self, **settings: Any) -> None:
        super().__init__(**settings, smooth=False)
        self.previous_shape_valid = False

    def observe(self, sample: Mapping[str, Any]) -> list[float]:
        prior_valid = self.previous_shape_valid
        state = super().observe(sample)
        valid = bool(state[18])
        trend_valid = valid and prior_valid
        if not trend_valid:
            state[13] = 0.0
        self.previous_shape_valid = valid
        return state + [float(trend_valid), sample["traffic"]["duration_s"] / 5, 1.0]


V2_FEATURES = [
    "rps",
    "isl",
    "queue_depth",
    "kv_utilization",
    "target_replicas",
    "rps_trend",
    "isl_trend",
    "constant",
    "shape_valid",
    "isl_trend_valid",
    "interval_over_5s",
    "observation_valid",
]
V3_FEATURES = [name.replace("_ema", "") for name in V3_SMOOTHED_FEATURES] + [
    "isl_trend_valid",
    "interval_over_5s",
    "observation_valid",
]
VERSIONS: dict[str, tuple[str, list[str]]] = {
    "v2_lstm_current": ("agg_v2_lstm_current_12", V2_FEATURES),
    "v3_lstm_current": ("agg_v3_lstm_current_27", V3_FEATURES),
}
DEFAULT_HISTORY_LENGTH = 10


class RawV2Encoder:
    """Compact v2 measurements from the five-second telemetry stream.

    Port of ``trainer/agg/features_raw_lstm.py::RawV2Encoder``: the rich raw
    v3 encoder (default SLO / cold start, replica bound 8) supplies the queue,
    KV, trends and validity entries; RPS and ISL are the current interval's
    values; the replica count is normalized by ``observation_max_replicas``.
    States are rounded to float32 exactly as the training views were.
    """

    def __init__(
        self, *, max_kv_tokens: float, observation_max_replicas: int = 32
    ) -> None:
        if max_kv_tokens <= 0 or observation_max_replicas <= 0:
            raise ValueError("Invalid normalization")
        self.max_kv_tokens = max_kv_tokens
        self.maximum = observation_max_replicas
        self.encoder = RawV3Encoder(max_replicas=8)

    def observe(self, sample: Mapping[str, Any]) -> list[float]:
        rich = self.encoder.observe(sample)
        traffic = sample["traffic"]
        duration = traffic["duration_s"]
        rps = traffic["arriving_requests"] / duration if duration else 0.0
        isl = traffic["avg_isl"] if rich[18] else 0.0
        queue = math.expm1(rich[3] * math.log1p(50))
        current = max(
            1, len(sample["active_decode_ids"]) + len(sample["starting_decode_ids"])
        )
        state = [
            rps / 200,
            isl / 4096,
            min(queue, 50) / 50,
            rich[5],
            current / self.maximum,
            rich[12],
            rich[13],
            0.5,
            rich[18],
            *rich[24:],
        ]
        if not all(math.isfinite(x) for x in state):
            raise ValueError("Nonfinite raw v2 state")
        return np.asarray(state, dtype=np.float32).tolist()


class CloudAIRLAutoscaleLSTM(TelemetryPolicy):
    """Bounded-history Q-only inference for the agg LSTM checkpoints.

    Args:
        checkpoint_path: ``cql_trainer`` agg LSTM export (``metadata.version``
            ``v2_lstm_current`` or ``v3_lstm_current``).
        slo_ttft_ms / slo_itl_ms / cold_start_s / max_replicas: must equal the
            checkpoint's training settings.
        capabilities: when given, the decode engine's ``max_kv_tokens`` must
            equal the checkpoint's ``settings.max_kv_tokens``.
        decision_log: optional JSONL file with one record per decision
            (encoded state, Q values, history length, current and target).
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        mode: str = "agg",
        capabilities: Optional[WorkerCapabilities] = None,
        slo_ttft_ms: float = 2000.0,
        slo_itl_ms: float = 50.0,
        min_replicas: int = 1,
        max_replicas: int = 8,
        cold_start_s: float = 30.0,
        poll_interval_s: float = 5.0,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        decision_log: Path | None = None,
    ) -> None:
        if mode != "agg":
            raise ValueError("CloudAI RL LSTM supports the agg topology only")
        super().__init__(poll_interval_s=poll_interval_s, decision_log=decision_log)
        if not 1 <= min_replicas <= max_replicas:
            raise ValueError(
                "replica bounds must satisfy 1 <= min_replicas <= max_replicas"
            )
        self._capabilities = capabilities
        self._min_replicas = int(min_replicas)
        self._max_replicas = int(max_replicas)

        checkpoint, metadata = load_checkpoint(checkpoint_path)
        version = metadata.get("version")
        if version not in VERSIONS:
            raise ValueError(
                f"Expected an agg LSTM checkpoint ({sorted(VERSIONS)}); got "
                f"version={version!r}"
            )
        schema, features = VERSIONS[version]
        if metadata.get("feature_schema") != schema:
            raise ValueError(
                f"Checkpoint feature schema {metadata.get('feature_schema')!r} differs "
                f"from the serving schema {schema!r}"
            )
        if list(metadata.get("features", features)) != features:
            raise ValueError(
                "Checkpoint feature order differs from the serving encoder"
            )
        if metadata.get("inference_mode", "q_only") != "q_only":
            raise ValueError("Agg LSTM checkpoints are served Q-only")
        if (
            metadata.get("network_type", "lstm") != "lstm"
            or metadata.get("input_mode", "current") != "current"
        ):
            raise ValueError("Expected an LSTM checkpoint over current inputs")
        history_length = int(metadata.get("history_length", DEFAULT_HISTORY_LENGTH))
        if history_length < 1:
            raise ValueError("Invalid history length")
        architecture = dict(metadata.get("architecture") or {})
        if (
            architecture.get("state_dim") != len(features)
            or architecture.get("num_actions", NUM_RELATIVE_ACTIONS)
            != NUM_RELATIVE_ACTIONS
            or architecture.get("hidden_dim", hidden_dim) != hidden_dim
            or architecture.get("num_blocks", num_blocks) != num_blocks
            or "lstm_hidden_dim" not in architecture
        ):
            raise ValueError(
                f"Unsupported checkpoint architecture {architecture}; serving uses "
                f"state_dim={len(features)}, num_actions={NUM_RELATIVE_ACTIONS}, "
                f"hidden_dim={hidden_dim}, num_blocks={num_blocks}"
            )
        settings = metadata.get("settings")
        if not isinstance(settings, Mapping):
            raise ValueError("Agg LSTM checkpoint metadata lacks observation settings")
        serving = {
            "max_replicas": self._max_replicas,
            "slo_ttft_ms": float(slo_ttft_ms),
            "slo_itl_ms": float(slo_itl_ms),
            "cold_start_delay_s": float(cold_start_s),
        }
        decode_caps = (
            getattr(capabilities, "decode", None) if capabilities is not None else None
        )
        capacity = getattr(decode_caps, "max_kv_tokens", None)
        if capacity is not None and settings.get("max_kv_tokens") is not None:
            serving["max_kv_tokens"] = float(capacity)
        mismatched = mismatched_settings(settings, serving)
        if mismatched:
            raise ValueError(
                "Checkpoint/serving feature configuration mismatch "
                "(checkpoint, serving): "
                + ", ".join(f"{key}={pair}" for key, pair in sorted(mismatched.items()))
            )
        if settings.get("topology", "agg") != "agg":
            raise ValueError("Agg LSTM checkpoint was not trained for the agg topology")
        observation_interval = settings.get(
            "observation_interval_s", self._poll_interval_s
        )
        if not math.isclose(float(observation_interval), self._poll_interval_s):
            raise ValueError(
                f"Checkpoint observation interval {observation_interval!r} s differs "
                f"from the serving cadence {self._poll_interval_s:g} s"
            )
        self._checkpoint_metadata = dict(metadata)
        self._version = version
        self._exact_actions = version.startswith("v3")
        self._history_length = history_length
        self._q_net = load_network(LSTMQNetwork(**architecture), checkpoint)
        if version.startswith("v2"):
            max_kv_tokens = settings.get("max_kv_tokens", capacity)
            if max_kv_tokens is None:
                raise ValueError("v2 LSTM checkpoint settings lack max_kv_tokens")
            self._encoder: Any = RawV2Encoder(
                max_kv_tokens=float(max_kv_tokens),
                observation_max_replicas=int(
                    settings.get("observation_max_replicas", 32)
                ),
            )
        else:
            self._encoder = RawV3Encoder(
                ttft_ms=serving["slo_ttft_ms"],
                itl_ms=serving["slo_itl_ms"],
                max_replicas=self._max_replicas,
                cold_start_s=serving["cold_start_delay_s"],
            )
        self._history: deque[list[float]] = deque(maxlen=history_length)
        self._state: Optional[list[float]] = None

    @property
    def version(self) -> str:
        return self._version

    @property
    def history_length(self) -> int:
        return self._history_length

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return dict(self._checkpoint_metadata)

    def _observe(self, sample: Mapping[str, Any]) -> None:
        self._state = self._encoder.observe(sample)
        self._history.append(self._state)

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        now = tick_input.now_s
        sample = self._current_sample(tick_input, ("decode",))
        assert self._state is not None
        current = max(
            self._min_replicas,
            len(sample["active_decode_ids"]) + len(sample["starting_decode_ids"]),
        )
        if not self._min_replicas <= current <= self._max_replicas:
            raise ValueError(
                f"Runtime fleet of {current} replicas is outside the checkpoint "
                f"action bounds [{self._min_replicas}, {self._max_replicas}]"
            )
        mask = (
            action_mask(current, self._min_replicas, self._max_replicas)
            if self._exact_actions
            else [True] * NUM_RELATIVE_ACTIONS
        )
        with torch.no_grad():
            window = history_window(self._history, self._history_length)
            q = self._q_net(torch.from_numpy(window[None]))[0]
            if not torch.isfinite(q).all():
                raise ValueError("Nonfinite recurrent policy Q values")
            index = int(q.masked_fill(~torch.tensor(mask), float("-inf")).argmax())
        target = action_targets(current, self._min_replicas, self._max_replicas)[index]
        self._record(
            {
                "timestamp_s": now,
                "current_replicas": current,
                "history_observations": len(self._history),
                "action_index": index,
                "action_delta": RELATIVE_ACTIONS[index],
                "action": target,
                "q_values": q.tolist(),
                "state": list(self._state),
            }
        )
        return PlannerEffects(
            scale_to=ScalingDecision(num_prefill=None, num_decode=target),
            next_tick=self._schedule(now + self._poll_interval_s),
        )


__all__ = [
    "DEFAULT_HISTORY_LENGTH",
    "V3_SMOOTHED_FEATURES",
    "RawV3Encoder",
    "V3StateEncoder",
    "action_mask",
    "action_targets",
    "V2_FEATURES",
    "V3_FEATURES",
    "VERSIONS",
    "CloudAIRLAutoscaleLSTM",
    "RawV2Encoder",
]
