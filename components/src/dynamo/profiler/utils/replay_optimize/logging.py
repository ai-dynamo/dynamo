# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from dynamo.runtime.logging import configure_dynamo_logging

from .models import DenseAggReplayState, DenseReplayState

logger = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False


def ensure_dynamo_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    configure_dynamo_logging()
    _LOGGING_CONFIGURED = True


def format_dense_state(state: DenseReplayState) -> str:
    return (
        "prefill_tp=%s decode_tp=%s prefill_workers=%s decode_workers=%s "
        "router_mode=%s overlap_score_weight=%s total_gpus=%s"
    ) % (
        state.prefill_tp,
        state.decode_tp,
        state.prefill_workers,
        state.decode_workers,
        state.router_mode,
        state.overlap_score_weight,
        state.total_gpus_used,
    )


def format_agg_state(state: DenseAggReplayState) -> str:
    return ("tp=%s workers=%s router_mode=%s overlap_score_weight=%s total_gpus=%s") % (
        state.tp,
        state.workers,
        state.router_mode,
        state.overlap_score_weight,
        state.total_gpus_used,
    )


def summarize_constraints(
    report: Mapping[str, Any],
    constraints: Mapping[str, float],
    total_gpus_used: int,
) -> str:
    if not constraints:
        return "constraints=none"

    statuses: list[str] = []
    for key, bound in constraints.items():
        if bound <= 0:
            continue
        value = total_gpus_used if key == "max_total_gpus" else report.get(key)
        if value is None:
            statuses.append(f"{key}=missing<={bound:g} unsatisfied")
            continue
        metric = float(value)
        state = "satisfied" if metric <= bound else "unsatisfied"
        statuses.append(f"{key}={metric:.3f}<={bound:g} {state}")

    return "constraints=" + ", ".join(statuses) if statuses else "constraints=none"


def log_dense_state_start(state: DenseReplayState) -> None:
    logger.info("Replay optimize evaluating %s", format_dense_state(state))


def log_dense_state_finish(
    *,
    state: DenseReplayState,
    report: Mapping[str, Any],
    constraints: Mapping[str, float],
    score: float,
    feasible: bool,
    violation_penalty: float,
) -> None:
    logger.info(
        "Replay optimize finished %s score=%.3f feasible=%s violation_penalty=%.6f %s",
        format_dense_state(state),
        score,
        feasible,
        violation_penalty,
        summarize_constraints(report, constraints, state.total_gpus_used),
    )


def log_agg_state_start(state: DenseAggReplayState) -> None:
    logger.info("Replay optimize evaluating %s", format_agg_state(state))


def log_agg_state_finish(
    *,
    state: DenseAggReplayState,
    report: Mapping[str, Any],
    constraints: Mapping[str, float],
    score: float,
    feasible: bool,
    violation_penalty: float,
) -> None:
    logger.info(
        "Replay optimize finished %s score=%.3f feasible=%s violation_penalty=%.6f %s",
        format_agg_state(state),
        score,
        feasible,
        violation_penalty,
        summarize_constraints(report, constraints, state.total_gpus_used),
    )
