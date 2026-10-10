# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Autoscaling Arena — a fair leaderboard for LLM-inference autoscalers.

Ranks the Dynamo Planner against rival autoscalers (KEDA, reactive, static, and
later Ray Serve / llm-d / an offline oracle) on a common DynoSim substrate,
common workloads, and common metrics. Every autoscaler enters by implementing
one ``EngineProtocol`` adapter; the harness drives it in a closed loop on the
simulated fleet.

Submodules are imported explicitly so each stays independently usable:

* ``autoscaling_arena.workloads`` — pure-Python trace generation (no Dynamo build needed).
* ``autoscaling_arena.adapters``  — autoscaler adapters (import Dynamo planner types).
* ``autoscaling_arena.runners``   — backends that drive a match (need the Dynamo runtime).
"""

__version__ = "0.1.0"
