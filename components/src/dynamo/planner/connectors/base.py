# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner connector interface.

A connector is the deployment-control peripheral consumed by
``PlannerEnvironment``.  It owns scaling, deployment validation, worker
capability discovery, and replica-state introspection for one deployment mode.
"""

from __future__ import annotations

from typing import Optional, Protocol

from dynamo.planner.config.defaults import SubComponentType, TargetReplica
from dynamo.planner.monitoring.worker_info import WorkerInfo


class WorkerInfoProvider(Protocol):
    def get_worker_info(
        self,
        sub_component_type: SubComponentType,
        backend: str = "vllm",
    ) -> WorkerInfo:
        pass


class PlannerConnector(WorkerInfoProvider, Protocol):
    async def async_init(self) -> None:
        pass

    async def validate_deployment(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> None:
        pass

    async def wait_for_deployment_ready(self, include_planner: bool = True) -> None:
        pass

    def get_model_name(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> str:
        pass

    def get_gpu_counts(
        self,
        require_prefill: bool = True,
        require_decode: bool = True,
    ) -> tuple[Optional[int], Optional[int]]:
        pass

    def get_actual_worker_counts(
        self,
        prefill_component_name: Optional[str] = None,
        decode_component_name: Optional[str] = None,
    ) -> tuple[int, int, bool]:
        pass

    async def set_component_replicas(
        self, target_replicas: list[TargetReplica], blocking: bool = True
    ) -> None:
        pass


__all__ = ["PlannerConnector", "WorkerInfoProvider"]
