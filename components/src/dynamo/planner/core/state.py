# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field

from dynamo.planner.monitoring.traffic_metrics import Metrics


@dataclass
class PlannerSharedState:
    last_metrics: Metrics = field(default_factory=Metrics)
    num_p_workers: int = 0
    num_d_workers: int = 0
    cumulative_gpu_hours: float = 0.0
    last_adjustment_time: float = 0.0
    # Lower bounds from throughput-based scaling (used when both modes enabled)
    throughput_lower_bound_p: int = 1
    throughput_lower_bound_d: int = 1
    # Separate timestamp for load-based adjustment loop
    last_load_adjustment_time: float = 0.0
