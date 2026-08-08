# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo replay API and adapter surface.

The shared offline implementation is owned by AISimulate. Dynamo retains these
entry points for single-run replay and for Router, Planner, and online adapters.
"""

from dynamo.replay.api import run_synthetic_trace_replay, run_trace_replay
from dynamo.replay.report import PlannerReplayDetails, ReplayReport

__all__ = [
    "PlannerReplayDetails",
    "ReplayReport",
    "run_synthetic_trace_replay",
    "run_trace_replay",
]
