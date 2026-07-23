# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.replay.api import run_synthetic_trace_replay, run_trace_replay

__all__ = [
    "ReplayDeploymentConfig",
    "ReplayGlobalPlannerConfig",
    "ReplaySyntheticWorkload",
    "ReplayTraceWorkload",
    "ReplayWorldReport",
    "run_replay_world",
    "run_synthetic_trace_replay",
    "run_trace_replay",
]

_WORLD_EXPORTS = frozenset(__all__[:-2])


def __getattr__(name: str):
    # Preserve the existing cheap `dynamo.replay` import path. The world API
    # pulls in the planner and Global Planner stacks only when requested.
    if name in _WORLD_EXPORTS:
        from dynamo.replay import world

        return getattr(world, name)
    raise AttributeError(name)
