# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "PlannerConnector",
    "KubernetesConnector",
    "VirtualConnector",
    "GlobalPlannerConnector",
    "SLAPlannerDefaults",
    "TargetReplica",
    "SubComponentType",
    "WorkerInfo",
]

from dynamo.planner.config.defaults import (
    SLAPlannerDefaults,
    SubComponentType,
    TargetReplica,
)
from dynamo.planner.connectors.base import PlannerConnector
from dynamo.planner.connectors.global_planner import GlobalPlannerConnector
from dynamo.planner.connectors.kubernetes import KubernetesConnector
from dynamo.planner.connectors.virtual import VirtualConnector
from dynamo.planner.monitoring.worker_info import WorkerInfo

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"
