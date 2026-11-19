# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backends that drive the bench suite.

- ``sims``  — DynoSim (offline, deterministic) backend; drives autoscaler
  adapters against the mocker. Needs the Dynamo runtime.
- ``real``  — online backend; shells out to the AIPerf CLI against live
  endpoints. Needs the ``aiperf`` CLI, NOT the Dynamo runtime.

Imports are lazy so the two backends don't force each other's (very different)
dependencies: ``import autoscaling_arena.runners.real`` works without a Dynamo
build, and vice versa.
"""

__all__ = ["run_arena_replay", "run_endpoint_leaderboard"]


def __getattr__(name):  # PEP 562 lazy attribute import
    if name == "run_arena_replay":
        from autoscaling_arena.runners.sims import run_arena_replay

        return run_arena_replay
    if name == "run_endpoint_leaderboard":
        from autoscaling_arena.runners.real import run_endpoint_leaderboard

        return run_endpoint_leaderboard
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
