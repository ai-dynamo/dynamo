# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serving pieces shared by the telemetry-driven CloudAI CQL adapters.

Ported from ``cql_trainer`` (``trainer/core/cql.py``, ``trainer/core/cql_lstm.py``,
``trainer/core/cql_branching.py`` and the ``history_window`` helpers): only
what inference needs, no trainers or datasets.

* :class:`DiscreteQNetwork` — the residual MLP Q head (LayerNorm/GELU, four
  blocks) every checkpoint family ends in.

* :class:`LSTMQNetwork` — one unidirectional 32-unit LSTM over the last ten
  five-second observations; its final hidden state is concatenated with the
  latest observation and fed to the shared residual Q head. Left padding is
  skipped via packing and the hidden state is zero for every window.
* :class:`BranchingQNetwork` / :class:`BranchingLSTMQNetwork` — the disagg
  additive two-head critic: ``Q(s, aP, aD) = qP(s, aP) + qD(s, aD)`` with one
  seven-action output per pool.
* :func:`history_window` — left-zero-padded bounded history.
* :class:`TelemetryPolicy` — the replay-telemetry adapter contract
  (``consumes_replay_telemetry``, ``on_telemetry``) with the pre-decision
  sample check, the fleet/ordering cross-check and the optional decision log
  that every CQL adapter shares.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch
from autoscaling_arena.adapters._regression_bootstrap import _NoopRegressionBootstrap
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from dynamo.planner.core.types import ScheduledTick, TickInput

INTERVAL_S = 5.0
# Nine relative fleet deltas shared by every agg checkpoint.
RELATIVE_ACTIONS = [-8, -4, -2, -1, 0, +1, +2, +4, +8]
NUM_RELATIVE_ACTIONS = len(RELATIVE_ACTIONS)


class DiscreteQNetwork(nn.Module):
    """Residual Q-network: state -> Q-values for each discrete action.

    Must match the architecture used during training.
    """

    def __init__(
        self,
        state_dim: int = 8,
        num_actions: int = NUM_RELATIVE_ACTIONS,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(
                nn.Sequential(
                    nn.LayerNorm(hidden_dim),
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(0.05),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
            )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(state)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class LSTMQNetwork(nn.Module):
    """Small temporal encoder feeding the residual CQL Q head."""

    def __init__(
        self,
        state_dim: int = 8,
        num_actions: int = 9,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        lstm_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.lstm = nn.LSTM(state_dim, lstm_hidden_dim, num_layers=1, batch_first=True)
        self.q_head = DiscreteQNetwork(
            state_dim + lstm_hidden_dim, num_actions, hidden_dim, num_blocks
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if (
            states.ndim != 3
            or states.shape[-1] != self.state_dim
            or not states.shape[1]
        ):
            raise ValueError("LSTM expects (batch, history, state_dim)")
        # A zero hidden/cell state for every window matches bounded live history.
        valid = states[:, :, -1] > 0
        lengths = valid.sum(1)
        if not torch.all(lengths > 0) or not torch.all(valid[:, -1]):
            raise ValueError("Every history must end with a valid current observation")
        expected = torch.arange(states.shape[1], device=states.device)[None] >= (
            states.shape[1] - lengths[:, None]
        )
        if not torch.equal(valid, expected):
            raise ValueError("Only contiguous left padding is supported")
        if bool(valid.all()):
            _, (hidden, _) = self.lstm(states)
        else:
            # Move actual observations to the left, then omit padding entirely.
            index = (
                torch.arange(states.shape[1], device=states.device)[None]
                + states.shape[1]
                - lengths[:, None]
            ) % states.shape[1]
            ordered = states.gather(1, index[:, :, None].expand_as(states))
            packed = pack_padded_sequence(
                ordered, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, _) = self.lstm(packed)
        return self.q_head(torch.cat((states[:, -1], hidden[-1]), dim=-1))


class BranchingQNetwork(DiscreteQNetwork):
    """Residual head with two ``num_actions``-wide outputs (prefill, decode)."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 4,
    ) -> None:
        super().__init__(state_dim, 2 * num_actions, hidden_dim, num_blocks)
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return super().forward(state).reshape(-1, 2, self.num_actions)


class BranchingLSTMQNetwork(nn.Module):
    """Ten-step LSTM encoder feeding the shared head with two seven-action outputs."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int = 7,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        lstm_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.encoder = LSTMQNetwork(
            state_dim, 2 * num_actions, hidden_dim, num_blocks, lstm_hidden_dim
        )
        # Alias kept so checkpoints saved by the trainer (which registers the
        # LSTM twice) load with strict=True.
        self.lstm = self.encoder.lstm

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.encoder(states).reshape(-1, 2, self.num_actions)


def history_window(states: Any, length: int) -> np.ndarray:
    """Left-zero-pad short histories; hidden state never carries across windows."""
    values = np.asarray(list(states), dtype=np.float32)
    if length <= 0 or values.ndim != 2 or not len(values):
        raise ValueError("Expected a nonempty history and positive window length")
    result = np.zeros((length, values.shape[1]), dtype=np.float32)
    recent = values[-length:]
    result[-len(recent) :] = recent
    return result


def load_checkpoint(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(checkpoint, metadata)``; CQL adapters need the training metadata."""
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    metadata = checkpoint.get("metadata") if isinstance(checkpoint, dict) else None
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{path}: CQL checkpoint has no training metadata")
    return checkpoint, dict(metadata)


def load_network(network: nn.Module, checkpoint: Mapping[str, Any]) -> nn.Module:
    network.load_state_dict(checkpoint["q_net_state_dict"], strict=True)
    if not all(torch.isfinite(t).all() for t in network.state_dict().values()):
        raise ValueError("Nonfinite checkpoint parameters")
    return network.eval()


def mismatched_settings(
    trained: Mapping[str, Any], serving: Mapping[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """``{key: (checkpoint, serving)}`` for every serving key that differs."""
    out: dict[str, tuple[Any, Any]] = {}
    for key, value in serving.items():
        other = trained.get(key, None)
        if (
            isinstance(value, (int, float))
            and isinstance(other, (int, float))
            and not isinstance(value, bool)
        ):
            same = math.isclose(float(other), float(value))
        else:
            same = other == value
        if not same:
            out[key] = (other, value)
    return out


class TelemetryPolicy(_NoopRegressionBootstrap):
    """Base for adapters that observe ``dynamo.replay.telemetry.v1`` samples.

    ``runners.sims.run_arena_replay`` wires ``on_telemetry`` to the native
    ``telemetry_callback`` when ``consumes_replay_telemetry`` is set; the
    replay emits each sample *before* the scaling callback at the same
    instant. Subclasses implement :meth:`_observe` (encode one sample) and
    ``tick``; :meth:`_current_sample` enforces that a decision only happens on
    a sample taken at the decision time with a fleet that matches the
    scaling snapshot.
    """

    consumes_replay_telemetry = True
    telemetry_interval_s = INTERVAL_S

    def __init__(
        self, *, poll_interval_s: float = INTERVAL_S, decision_log: Path | None = None
    ) -> None:
        if not math.isclose(float(poll_interval_s), INTERVAL_S):
            raise ValueError(
                f"{type(self).__name__} decides at its {INTERVAL_S:g} s telemetry "
                f"cadence; got poll_interval_s={poll_interval_s!r}"
            )
        self._poll_interval_s = INTERVAL_S
        self._latest: Optional[dict[str, Any]] = None
        self._decisions = 0
        self._log = decision_log.open("x", encoding="utf-8") if decision_log else None

    # --- subclass hooks -----------------------------------------------------

    def _observe(
        self, sample: Mapping[str, Any]
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError

    # --- replay telemetry ---------------------------------------------------

    def on_telemetry(self, sample: Any) -> None:
        """Consume one pre-decision telemetry sample."""
        if isinstance(sample, (str, bytes, bytearray)):
            sample = json.loads(sample)
        if not isinstance(sample, Mapping):
            raise TypeError("telemetry sample must be a mapping")
        at = sample["sampled_at_ms"] / 1000.0
        if self._latest is not None and at == self._latest["sampled_at_ms"] / 1000.0:
            # A native final sample may duplicate a periodic sample. No further
            # decision is taken after final telemetry.
            if sample.get("kind") != "final":
                raise ValueError("Duplicate nonterminal telemetry")
            return
        self._observe(sample)
        self._latest = dict(sample)

    @property
    def decisions(self) -> int:
        return self._decisions

    @property
    def inference_mode(self) -> str:
        return "q_only"

    # --- EngineProtocol -----------------------------------------------------

    def initial_tick(self, start_s: float) -> ScheduledTick:
        return self._schedule(start_s)

    def _schedule(self, at_s: float) -> ScheduledTick:
        # Worker states are requested only to cross-check that the telemetry
        # sample describes the same fleet the scaling callback sees.
        return ScheduledTick(at_s=at_s, need_worker_states=True)

    def _current_sample(
        self, tick_input: TickInput, pools: tuple[str, ...]
    ) -> dict[str, Any]:
        now = tick_input.now_s
        if self._latest is None or not math.isclose(
            self._latest["sampled_at_ms"] / 1000.0, now, abs_tol=1e-6
        ):
            raise ValueError(
                f"{type(self).__name__} requires a replay telemetry sample at every "
                f"decision; none for t={now:.3f} s (replay telemetry must be "
                f"enabled at a {INTERVAL_S:g} s cadence)"
            )
        counts = tick_input.worker_counts
        if counts is not None:
            for pool in pools:
                ready = getattr(counts, f"ready_num_{pool}", None)
                active = len(self._latest[f"active_{pool}_ids"])
                if ready is not None and ready != active:
                    raise ValueError(
                        "Telemetry/scaling callback ordering differs: telemetry "
                        f"reports {active} active {pool} workers, the scaling "
                        f"snapshot {ready}"
                    )
        return self._latest

    def _record(self, record: Mapping[str, Any]) -> None:
        self._decisions += 1
        if self._log:
            self._log.write(json.dumps(record, allow_nan=False) + "\n")
            self._log.flush()

    async def shutdown(self) -> None:
        if self._log:
            self._log.close()
            self._log = None


__all__ = [
    "INTERVAL_S",
    "NUM_RELATIVE_ACTIONS",
    "RELATIVE_ACTIONS",
    "DiscreteQNetwork",
    "BranchingLSTMQNetwork",
    "BranchingQNetwork",
    "LSTMQNetwork",
    "TelemetryPolicy",
    "history_window",
    "load_checkpoint",
    "load_network",
    "mismatched_settings",
]
