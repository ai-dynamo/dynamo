# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GlobalPlanner - Centralized scaling execution service.

The GlobalPlanner is a standalone component that receives scale requests from
Planners and executes them via the Kubernetes API. It provides centralized
scaling management across multiple deployments and namespaces.

Architecture:
- Planners make scaling decisions (observe, predict, decide)
- Planners in delegating mode send requests to GlobalPlanner
- GlobalPlanner executes scaling via Kubernetes API
- GlobalPlanner is stateless and can scale horizontally

Usage:
    DYN_NAMESPACE=global-infra python -m dynamo.global_planner \
        --managed-namespaces app-ns-1 app-ns-2
"""

__all__ = [
    "ScaleRequestHandler",
]

from dynamo.global_planner.scale_handler import ScaleRequestHandler
