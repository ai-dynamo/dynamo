# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Protocol

from kubernetes.client import ApiException
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.errors import PlannerError
from dynamo.runtime import DistributedRuntime

logger = logging.getLogger(__name__)


class RuntimeNamespaceResolver(Protocol):
    def get_worker_runtime_namespace(self, base_dynamo_namespace: str) -> str:
        pass


class RuntimeNamespaceBinding:
    """Optional runtime namespace binding for runtime-backed providers."""

    def __init__(
        self,
        *,
        namespace: str,
        runtime: DistributedRuntime,
        resolver: RuntimeNamespaceResolver,
    ) -> None:
        self.namespace = namespace
        self.runtime_namespace_value = namespace
        self.runtime = runtime
        self.resolver = resolver

    def runtime_namespace(self) -> str:
        return self.runtime_namespace_value

    async def refresh_runtime_namespace(self) -> bool:
        try:
            runtime_namespace = self.resolver.get_worker_runtime_namespace(
                self.namespace
            )
        except (ApiException, ConfigException, PlannerError) as exc:
            logger.warning(
                "Failed to resolve worker runtime namespace: %s; keeping %s",
                exc,
                self.runtime_namespace_value,
            )
            return False
        if runtime_namespace == self.runtime_namespace_value:
            return False
        self.runtime_namespace_value = runtime_namespace
        return True

    async def get_or_create_client(self, component_name: str, endpoint_name: str):
        return await self.runtime.endpoint(
            f"{self.runtime_namespace_value}.{component_name}.{endpoint_name}"
        ).client()
