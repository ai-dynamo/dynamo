# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workload definitions and Mooncake-JSONL trace generation.

A workload is composed over three orthogonal axes, each mapping to one field of
the Mooncake-JSONL trace schema the DynoSim substrate consumes:

* arrival process  -> ``timestamp``            (flat / staircase / square-wave / flash-crowd / diurnal)
* request shape    -> ``input_length`` / ``output_length``  (prefill-heavy / decode-heavy / balanced)
* prefix structure -> ``hash_ids``             (none / shared-prefix-RAG / real)

Generators serialize to Mooncake JSONL so workloads flow through the existing
``create_disagg`` path unchanged. The v1 registry ships 8 workloads spanning all
three axes; see :data:`WORKLOADS`.
"""

from autoscaling_arena.workloads import axes
from autoscaling_arena.workloads.generator import Workload, validate_mooncake_trace
from autoscaling_arena.workloads.registry import WORKLOADS, get_workload, list_workloads

__all__ = [
    "axes",
    "Workload",
    "validate_mooncake_trace",
    "WORKLOADS",
    "get_workload",
    "list_workloads",
]
