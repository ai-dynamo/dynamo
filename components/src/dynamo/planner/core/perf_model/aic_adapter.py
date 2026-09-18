# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility imports for the renamed AISimulate Planner adapter."""

from dynamo.planner.core.perf_model.ais_adapter import (
    PlannerEngineCapacity,
    PlannerEnginePerfModel,
)

__all__ = ["PlannerEngineCapacity", "PlannerEnginePerfModel"]
