# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import Protocol

from kubernetes.client import ApiException
from kubernetes.config.config_exception import ConfigException

from dynamo.planner.errors import PlannerError

logger = logging.getLogger(__name__)


class RuntimeNamespaceResolver(Protocol):
    def get_worker_runtime_namespace(self, base_dynamo_namespace: str) -> str:
        pass


class RuntimeNamespaceBinding:
    """Mutable binding for the worker generation's Dynamo namespace."""

    def __init__(
        self,
        *,
        namespace: str,
        resolver: RuntimeNamespaceResolver,
    ) -> None:
        self.namespace = namespace
        self.runtime_namespace_value = namespace
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
