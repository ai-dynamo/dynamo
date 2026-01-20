# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

__all__ = [
    "PlannerConnector",
    "KubernetesConnector",
    "VirtualConnector",
    "LoadPlannerDefaults",
    "SLAPlannerDefaults",
    "TargetReplica",
    "SubComponentType",
]
# Import the classes
from dynamo.planner.defaults import (
    LoadPlannerDefaults,
    SLAPlannerDefaults,
    SubComponentType,
)
from dynamo.planner.kubernetes_connector import KubernetesConnector, TargetReplica
from dynamo.planner.planner_connector import PlannerConnector
from dynamo.planner.virtual_connector import VirtualConnector

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"
