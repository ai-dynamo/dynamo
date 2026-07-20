# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional, Protocol

from dynamo.planner.core.types import FpmObservations, TrafficObservation
from dynamo.planner.environment.state import DeploymentState


class TrafficMetricsProvider(Protocol):
    async def collect_traffic(self) -> Optional[TrafficObservation]:
        pass

    def collect_accept_length(self, interval_str: str) -> Optional[float]:
        pass

    async def collect_kv_hit_rate_observation(
        self, duration_s: float
    ) -> Optional[TrafficObservation]:
        pass


class FpmMetricsProvider(Protocol):
    async def async_init(self, namespace: Optional[str] = None) -> None:
        pass

    async def refresh(self, state: DeploymentState) -> None:
        pass

    def collect_fpm(self) -> FpmObservations:
        pass

    async def shutdown(self) -> None:
        pass


__all__ = ["FpmMetricsProvider", "TrafficMetricsProvider"]
