# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from dynamo.runtime.logging import configure_dynamo_logging

from .models import DenseAggReplayState, DenseReplayState, ReplayConstraints

logger = logging.getLogger(__name__)
_LOGGING_CONFIGURED = False


def ensure_dynamo_logging() -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED:
        return
    configure_dynamo_logging()
    _LOGGING_CONFIGURED = True


def log_state_start(state: DenseReplayState | DenseAggReplayState) -> None:
    logger.info("Replay optimize evaluating %s", state.format_summary())


def log_state_finish(
    *,
    state: DenseReplayState | DenseAggReplayState,
    report: Mapping[str, Any],
    constraints: ReplayConstraints,
    score: float,
    feasible: bool,
    violation_penalty: float,
) -> None:
    logger.info(
        "Replay optimize finished %s score=%.3f feasible=%s violation_penalty=%.6f %s",
        state.format_summary(),
        score,
        feasible,
        violation_penalty,
        constraints.summarize(report, state.total_gpus_used),
    )
