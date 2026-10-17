# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CloudAI RL disagg policies: ``disagg_v2`` / ``disagg_v3`` and their LSTMs.

Serves the four disaggregated CQL checkpoints trained by ``cql_trainer``
(``trainer/disagg/features.py`` encoders and the ``trainer/disagg/replay.py``
``Controller``). Every five seconds the policy observes the pre-decision
``dynamo.replay.telemetry.v1`` sample together with the targets it last
requested, and emits independent prefill and decode targets:

* observation: arrival RPS and completed ISL/OSL (EMA with alpha 0.4 per 5 s
  for the feedforward models, current-interval values for the LSTMs), per-pool
  waiting requests (ready and draining workers plus router pending), mean
  ready-decode active KV, per-pool requested / ready / starting / draining
  counts over the pool maximum, validity flags; v3 adds per-pool running
  requests and startup ETAs, trends, SLO-relative latency, counts and shape
  age (18 / 35 entries; LSTM variants append interval / 5 s and an
  observation-validity flag: 20 / 37);
* actions: one delta in ``[-4, -2, -1, 0, +1, +2, +4]`` per pool relative to
  the last *requested* target (prefill 1-16, decode 1-8), infeasible deltas
  masked rather than clamped, chosen per head from the additive joint critic
  ``Q(s, aP, aD) = qP(s, aP) + qD(s, aD)``.

The checkpoint's settings (bounds, SLO, cold start, smoothing, GPUs per
worker, action deltas) must equal the serving settings; the runtime must
honour every requested target by the next tick (verified against the
telemetry fleet, as the trainer's controller does). Disagg only.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
from autoscaling_arena.adapters.cql_common import (
    BranchingLSTMQNetwork,
    BranchingQNetwork,
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

INTERVAL_S = 5.0
DELTAS = [-4, -2, -1, 0, 1, 2, 4]
POOLS = ("prefill", "decode")
VERSIONS = ("v2", "v3")
HISTORY_LENGTH = 10
V2_FEATURES = [
    "log_arrival_rps_ema",
    "log_completed_isl_ema",
    "log_completed_osl_ema",
    "log_prefill_waiting",
    "log_decode_waiting",
    "mean_active_decode_kv",
    "prefill_requested",
    "prefill_ready",
    "prefill_starting",
    "prefill_draining",
    "decode_requested",
    "decode_ready",
    "decode_starting",
    "decode_draining",
    "arrival_valid",
    "completed_shape_valid",
    "prefill_scheduler_valid",
    "decode_scheduler_valid",
]
# v3 adds running requests, startup estimates, trends, SLO-relative latency,
# measurement counts and shape age, mirroring the agg v3 contract per pool.
V3_EXTRA = [
    "log_prefill_running",
    "log_decode_running",
    "prefill_next_startup_eta",
    "prefill_mean_startup_eta",
    "decode_next_startup_eta",
    "decode_mean_startup_eta",
    "rps_trend",
    "isl_trend",
    "isl_trend_valid",
    "log_ttft_slo_ratio",
    "log_itl_slo_ratio",
    "ttft_valid",
    "itl_valid",
    "log_completion_rps",
    "log_ttft_count",
    "log_itl_count",
    "log_shape_age",
]
# LSTM models use current-interval traffic values instead of EMAs and append
# the interval duration plus an observation-validity flag for packed windows.
LSTM_EXTRA = ["interval_over_5s", "observation_valid"]
SETTINGS_SCHEMA = "disagg_v2_smoothed_18"
DEFAULT_SETTINGS: dict[str, Any] = dict(
    interval_s=5.0,
    min_prefill=1,
    max_prefill=16,
    min_decode=1,
    max_decode=8,
    slo_ttft_ms=2000.0,
    slo_itl_ms=50.0,
    cold_start_s=30.0,
    smoothing_alpha=0.4,
    prefill_gpus_per_worker=1,
    decode_gpus_per_worker=1,
    action_deltas=DELTAS,
    feature_schema=SETTINGS_SCHEMA,
)


def model_name(version: str, lstm: bool = False) -> str:
    if version not in VERSIONS:
        raise ValueError(f"Unsupported disagg version: {version}")
    return f"disagg_{version}" + ("_lstm" if lstm else "")


def feature_names(version: str, lstm: bool = False) -> list[str]:
    model_name(version, lstm)
    names = list(V2_FEATURES) + (list(V3_EXTRA) if version == "v3" else [])
    if lstm:
        names = [n.replace("_ema", "") for n in names] + LSTM_EXTRA
    return names


def schema_name(version: str, lstm: bool = False) -> str:
    return f'disagg_{version}_{"current" if lstm else "smoothed"}_{len(feature_names(version, lstm))}'


def history_length(lstm: bool = False) -> int:
    return HISTORY_LENGTH if lstm else 1


def masks(requested: Any, settings: Mapping[str, Any]) -> list[list[bool]]:
    result = []
    for pool, count in zip(POOLS, requested, strict=True):
        lo, hi = settings["min_" + pool], settings["max_" + pool]
        if isinstance(count, bool) or int(count) != count or not lo <= count <= hi:
            raise ValueError("Requested count outside pool bounds")
        result.append([lo <= count + d <= hi for d in DELTAS])
    return result


def targets(requested: Any, actions: Any, settings: Mapping[str, Any]) -> list[int]:
    valid = masks(requested, settings)
    if len(actions) != 2 or any(
        not 0 <= a < len(DELTAS) or not valid[i][a] for i, a in enumerate(actions)
    ):
        raise ValueError("Infeasible action; clamping is forbidden")
    return [count + DELTAS[a] for count, a in zip(requested, actions, strict=True)]


def _log(value: Any, scale: float) -> float:
    if value is None or not math.isfinite(value) or value < 0:
        raise ValueError("Measurement must be finite and nonnegative")
    return math.log1p(value) / math.log1p(scale)


def _pool_rows(sample: Mapping[str, Any], pool: str):
    active = set(sample["active_" + pool + "_ids"])
    starting = set(sample["starting_" + pool + "_ids"])
    draining = set(sample["draining_" + pool + "_ids"])
    if active & starting or active & draining or starting & draining:
        raise ValueError("Overlapping lifecycle states")
    # Ignore starting-worker scheduler rows; include draining queues, but only
    # ready workers contribute to the decode capacity-pressure average.
    rows = [
        r
        for r in sample[pool + "_scheduler_metrics"]
        if r["worker_id"] in active | draining
    ]
    known = {r["worker_id"] for r in rows}
    pending = sample.get("router_pending_" + pool + "_requests")
    valid = known == active | draining and pending is not None
    if pool == "decode":
        valid = valid and bool(active)
    return rows, active, pending, valid


def pool_signals(sample: Mapping[str, Any], pool: str) -> tuple[float, float, bool]:
    rows, active, pending, valid = _pool_rows(sample, pool)
    if not valid:
        return 0.0, 0.0, False
    queue = sum(r["waiting_requests"] for r in rows) + pending
    ready = [r for r in rows if r["worker_id"] in active]
    kv = sum(r["active_cache_usage"] for r in ready) / len(ready) if ready else 0.0
    _log(queue, 50)
    _log(kv, 1)
    return queue, kv, True


def pool_running(sample: Mapping[str, Any], pool: str) -> int:
    rows, _, _, valid = _pool_rows(sample, pool)
    return sum(r["running_requests"] for r in rows) if valid else 0


class DisaggStateEncoder:
    """One encoder per model contract (port of ``trainer/disagg/features.py``)."""

    def __init__(
        self,
        settings: Optional[Mapping[str, Any]] = None,
        version: str = "v2",
        lstm: bool = False,
    ) -> None:
        self.settings = dict(DEFAULT_SETTINGS if settings is None else settings)
        self.version, self.lstm = version, lstm
        self.names = feature_names(version, lstm)
        self.smooth = not lstm
        self.last_s: Optional[float] = None
        self.last_shape_s: Optional[float] = None
        self.rps = self.isl = self.osl = 0.0
        self.previous_shape_valid = False
        self.starting_since: dict[str, dict[Any, float]] = {pool: {} for pool in POOLS}

    def observe(self, sample: Mapping[str, Any], requested: Any) -> list[float]:
        masks(requested, self.settings)
        now = sample["sampled_at_ms"] / 1000
        if self.last_s is not None and now <= self.last_s:
            raise ValueError("Observations must advance in time")
        traffic = sample["traffic"]
        duration = traffic["duration_s"]
        if duration < 0 or not math.isclose(
            duration,
            (sample["sampled_at_ms"] - sample["interval_start_ms"]) / 1000,
            abs_tol=1e-6,
        ):
            raise ValueError("Inconsistent telemetry interval")
        dt = now - self.last_s if self.last_s is not None else 0.0
        alpha = 1 - (1 - self.settings["smoothing_alpha"]) ** (duration / INTERVAL_S)
        arrival_valid = duration > 0
        old_rps, old_isl = self.rps, self.isl
        if arrival_valid:
            rate = traffic["arriving_requests"] / duration
            _log(rate, 20)
            self.rps = self.rps + alpha * (rate - self.rps) if self.smooth else rate
        elif not self.smooth:
            self.rps = 0.0
        shape_valid = traffic["completed_requests"] > 0
        if shape_valid:
            isl, osl = traffic["avg_isl"], traffic["avg_osl"]
            _log(isl, 4096)
            _log(osl, 1024)
            if self.smooth and self.last_shape_s is not None:
                a = 1 - (1 - self.settings["smoothing_alpha"]) ** (
                    (now - self.last_shape_s) / INTERVAL_S
                )
                self.isl += a * (isl - self.isl)
                self.osl += a * (osl - self.osl)
            else:
                self.isl, self.osl = isl, osl
            self.last_shape_s = now
        elif not self.smooth:
            self.isl = self.osl = 0.0
        pq, _, pv = pool_signals(sample, "prefill")
        dq, kv, dv = pool_signals(sample, "decode")
        state = [
            _log(self.rps, 20),
            _log(self.isl, 4096),
            _log(self.osl, 1024),
            _log(pq, 50),
            _log(dq, 50),
            kv,
        ]
        for i, pool in enumerate(POOLS):
            maximum = self.settings["max_" + pool]
            state.extend(
                [requested[i] / maximum]
                + [
                    len(sample[k + "_" + pool + "_ids"]) / maximum
                    for k in ("active", "starting", "draining")
                ]
            )
        state.extend(map(float, (arrival_valid, shape_valid, pv, dv)))
        if self.version == "v3":
            state.extend(
                self._v3_extra(
                    sample, now, dt, duration, traffic, old_rps, old_isl, shape_valid
                )
            )
        if self.lstm:
            state.extend([duration / INTERVAL_S, 1.0])
        self.previous_shape_valid = shape_valid
        self.last_s = now
        if len(state) != len(self.names) or not all(math.isfinite(v) for v in state):
            raise ValueError("Invalid state vector")
        return state

    def _v3_extra(
        self, sample, now, dt, duration, traffic, old_rps, old_isl, shape_valid
    ) -> list[float]:
        cold = self.settings["cold_start_s"]
        extra = [_log(pool_running(sample, pool), 50) for pool in POOLS]
        for pool in POOLS:
            # At a five-second cadence, a newly observed starting worker was
            # requested at the preceding sample: an observable ETA estimate.
            starting = set(sample["starting_" + pool + "_ids"])
            birth = self.last_s if self.last_s is not None else now
            self.starting_since[pool] = {
                w: self.starting_since[pool].get(w, birth) for w in starting
            }
            etas = [
                max(0.0, cold - (now - b)) / cold
                for b in self.starting_since[pool].values()
            ]
            extra += [
                min(etas) if etas else 0.0,
                sum(etas) / len(etas) if etas else 0.0,
            ]
        trend_scale = INTERVAL_S / dt if dt else 0.0

        def trend(a: float, b: float) -> float:
            return max(-2.0, min(2.0, (a - b) / max(1.0, b) * trend_scale))

        isl_trend_valid = shape_valid and self.previous_shape_valid
        extra += [
            trend(self.rps, old_rps),
            trend(self.isl, old_isl) if isl_trend_valid else 0.0,
            float(isl_trend_valid),
        ]
        nt, ni = traffic["ttft_count"], traffic["itl_count"]
        age = now - self.last_shape_s if self.last_shape_s is not None else now
        extra += [
            math.log1p(traffic["avg_ttft_ms"] / self.settings["slo_ttft_ms"])
            if nt
            else 0.0,
            math.log1p(traffic["avg_itl_ms"] / self.settings["slo_itl_ms"])
            if ni
            else 0.0,
            float(nt > 0),
            float(ni > 0),
            _log(traffic["completed_requests"] / duration if duration else 0.0, 20),
            _log(nt, 50),
            _log(ni, 50),
            _log(age, 60),
        ]
        return extra


def checkpoint_contract(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Model name, version and history for a saved disagg checkpoint."""
    version = str(metadata.get("version", ""))
    if not version.startswith("disagg_") or version[7:] not in VERSIONS:
        raise ValueError(
            f"Expected a disagg checkpoint (version disagg_v2 / disagg_v3); got {version!r}"
        )
    short, lstm = version[7:], bool(metadata.get("lstm", False))
    return dict(
        model=model_name(short, lstm),
        version=short,
        lstm=lstm,
        history_length=history_length(lstm),
    )


class CloudAIRLDisaggAutoscaler(TelemetryPolicy):
    """Two-pool Q-only CQL controller (port of the trainer's ``Controller``).

    Args:
        checkpoint_path: ``cql_trainer`` disagg export; version and LSTM-ness
            are read from its metadata.
        min_prefill / max_prefill / min_decode / max_decode / slo_* /
        cold_start_s / smoothing_alpha / prefill_gpus_per_worker /
        decode_gpus_per_worker: the serving settings, which must equal the
            checkpoint's ``metadata.settings``.
        decision_log: optional JSONL file with one record per decision.
    """

    def __init__(
        self,
        checkpoint_path: str,
        *,
        mode: str = "disagg",
        capabilities: Optional[WorkerCapabilities] = None,
        slo_ttft_ms: float = 2000.0,
        slo_itl_ms: float = 50.0,
        min_prefill: int = 1,
        max_prefill: int = 16,
        min_decode: int = 1,
        max_decode: int = 8,
        cold_start_s: float = 30.0,
        smoothing_alpha: float = 0.4,
        prefill_gpus_per_worker: int = 1,
        decode_gpus_per_worker: int = 1,
        poll_interval_s: float = INTERVAL_S,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        decision_log: Path | None = None,
    ) -> None:
        if mode != "disagg":
            raise ValueError("CloudAI RL disagg supports the disagg topology only")
        super().__init__(poll_interval_s=poll_interval_s, decision_log=decision_log)
        self._capabilities = capabilities
        settings: dict[str, Any] = dict(
            interval_s=INTERVAL_S,
            min_prefill=int(min_prefill),
            max_prefill=int(max_prefill),
            min_decode=int(min_decode),
            max_decode=int(max_decode),
            slo_ttft_ms=float(slo_ttft_ms),
            slo_itl_ms=float(slo_itl_ms),
            cold_start_s=float(cold_start_s),
            smoothing_alpha=float(smoothing_alpha),
            prefill_gpus_per_worker=int(prefill_gpus_per_worker),
            decode_gpus_per_worker=int(decode_gpus_per_worker),
            action_deltas=list(DELTAS),
            feature_schema=SETTINGS_SCHEMA,
        )
        for pool in POOLS:
            if not 1 <= settings["min_" + pool] <= settings["max_" + pool]:
                raise ValueError(f"{pool} bounds must satisfy 1 <= min <= max")

        checkpoint, metadata = load_checkpoint(checkpoint_path)
        contract = checkpoint_contract(metadata)
        names = feature_names(contract["version"], contract["lstm"])
        if metadata.get("feature_schema") != schema_name(
            contract["version"], contract["lstm"]
        ):
            raise ValueError(
                f"Checkpoint feature schema {metadata.get('feature_schema')!r} differs "
                f"from the serving schema {schema_name(contract['version'], contract['lstm'])!r}"
            )
        if list(metadata.get("features", names)) != names:
            raise ValueError(
                "Checkpoint feature order differs from the serving encoder"
            )
        if list(metadata.get("relative_actions", DELTAS)) != DELTAS:
            raise ValueError("Checkpoint action deltas differ from the serving deltas")
        if (
            int(metadata.get("history_length", contract["history_length"]))
            != contract["history_length"]
        ):
            raise ValueError("Checkpoint history length differs from its contract")
        if metadata.get("inference_mode", "q_only") != "q_only":
            raise ValueError("Disagg checkpoints are served Q-only")
        trained = metadata.get("settings")
        if not isinstance(trained, Mapping):
            raise ValueError("Disagg checkpoint metadata lacks settings")
        mismatched = mismatched_settings(trained, settings)
        if mismatched or set(trained) != set(settings):
            extra = sorted(set(trained) ^ set(settings))
            raise ValueError(
                "Checkpoint/serving settings mismatch (checkpoint, serving): "
                + ", ".join(f"{key}={pair}" for key, pair in sorted(mismatched.items()))
                + (f"; keys only on one side: {extra}" if extra else "")
            )
        architecture = dict(metadata.get("architecture") or {})
        if (
            architecture.get("state_dim") != len(names)
            or architecture.get("num_actions") != len(DELTAS)
            or architecture.get("hidden_dim", hidden_dim) != hidden_dim
            or architecture.get("num_blocks", num_blocks) != num_blocks
            or (contract["lstm"] and "lstm_hidden_dim" not in architecture)
        ):
            raise ValueError(
                f"Invalid checkpoint architecture {architecture} for {contract['model']}"
            )
        network = (BranchingLSTMQNetwork if contract["lstm"] else BranchingQNetwork)(
            **architecture
        )
        self._q_net = load_network(network, checkpoint)
        self._settings = settings
        self._contract = contract
        self._checkpoint_metadata = dict(metadata)
        self._encoder = DisaggStateEncoder(
            settings, contract["version"], contract["lstm"]
        )
        self._history: deque[list[float]] = deque(maxlen=contract["history_length"])
        self._requested: Optional[list[int]] = None
        self._state: Optional[list[float]] = None

    @property
    def model(self) -> str:
        return self._contract["model"]

    @property
    def contract(self) -> dict[str, Any]:
        return dict(self._contract)

    @property
    def requested(self) -> Optional[list[int]]:
        """Targets requested at the last decision (the action baseline)."""
        return list(self._requested) if self._requested is not None else None

    @property
    def checkpoint_metadata(self) -> dict[str, Any]:
        return dict(self._checkpoint_metadata)

    def _observe(self, sample: Mapping[str, Any]) -> None:
        if self._requested is None:
            # The first sample describes the initial fleet, which is also the
            # baseline the trainer's controller started from.
            self._requested = [
                len(sample["active_" + pool + "_ids"])
                + len(sample["starting_" + pool + "_ids"])
                for pool in POOLS
            ]
        self._state = self._encoder.observe(sample, self._requested)
        self._history.append(self._state)

    async def tick(
        self,
        scheduled_tick: ScheduledTick,
        tick_input: TickInput,
    ) -> PlannerEffects:
        now = tick_input.now_s
        sample = self._current_sample(tick_input, POOLS)
        assert self._state is not None and self._requested is not None
        accepted = [
            len(sample["active_" + pool + "_ids"])
            + len(sample["starting_" + pool + "_ids"])
            for pool in POOLS
        ]
        if accepted != self._requested:
            raise ValueError(
                f"Runtime changed requested targets: sent={self._requested}, accepted={accepted}"
            )
        current = list(self._requested)
        valid = masks(current, self._settings)
        with torch.no_grad():
            if self._contract["lstm"]:
                window = history_window(self._history, self._contract["history_length"])
                q = self._q_net(torch.from_numpy(window[None]))[0]
            else:
                q = self._q_net(torch.tensor([self._state], dtype=torch.float32))[0]
            if not torch.isfinite(q).all():
                raise ValueError("Nonfinite serving Q values")
            actions = q.masked_fill(~torch.tensor(valid), -torch.inf).argmax(1).tolist()
        chosen = targets(current, actions, self._settings)
        self._record(
            {
                "timestamp_s": now,
                "requested_before": current,
                "action_index": actions,
                "action_deltas": [DELTAS[a] for a in actions],
                "policy_target": list(chosen),
                "history_observations": len(self._history),
                "q_values": q.tolist(),
                "state": list(self._state),
            }
        )
        self._requested = list(chosen)
        return PlannerEffects(
            scale_to=ScalingDecision(num_prefill=chosen[0], num_decode=chosen[1]),
            next_tick=self._schedule(now + self._poll_interval_s),
        )


__all__ = [
    "DEFAULT_SETTINGS",
    "DELTAS",
    "HISTORY_LENGTH",
    "POOLS",
    "VERSIONS",
    "CloudAIRLDisaggAutoscaler",
    "DisaggStateEncoder",
    "checkpoint_contract",
    "feature_names",
    "masks",
    "model_name",
    "pool_running",
    "pool_signals",
    "schema_name",
    "targets",
]
