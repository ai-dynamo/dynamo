# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
from itertools import product
from typing import NamedTuple


FRONTEND_COUNTS = (1, 2, 4)
CONCURRENCIES = (4096, 8192, 16384, 32768)
MOCKER_COUNTS = (10, 40, 80)
MODEL = "Qwen/Qwen3-0.6B"
VALKEY_GIT_REVISION = "5b690cefd6cad707a748879c2bab6b72e18efcb7"
ISL = 1024
OSL = 1024
INDEX_SCOPE = "k8s-sweep"
NAMESPACE = "bis-rl-3"
PART_OF = "valkey-router-sweep"
CLIENT_DEPLOYMENT = "valkey-sweep-client"
FRONTEND_DEPLOYMENT = "valkey-sweep-frontend"
MOCKER_DEPLOYMENT = "valkey-sweep-mocker"


def router_write_durability() -> dict[str, bool | int | str]:
    """Describe the explicit router write tradeoff exercised by this sweep."""
    return {
        "required_replica_acks": 1,
        "allow_degraded_writes": True,
        "mode": "availability-first-degraded-durability",
    }


class MatrixPoint(NamedTuple):
    frontends: int
    concurrency: int
    mockers: int

    @property
    def slug(self) -> str:
        return f"m{self.frontends}-c{self.concurrency}-n{self.mockers}"


def matrix_points() -> list[MatrixPoint]:
    return [
        MatrixPoint(frontends, concurrency, mockers)
        for frontends, concurrency, mockers in product(
            FRONTEND_COUNTS, CONCURRENCIES, MOCKER_COUNTS
        )
    ]


def request_count(concurrency: int) -> int:
    """Measure at least four full closed-loop waves after warmup."""
    return max(16384, 4 * concurrency)


def warmup_count(concurrency: int) -> int:
    """Warm one complete closed-loop wave before collecting measurements."""
    return concurrency


def runtime_namespace(campaign: str, point: MatrixPoint, generation: str) -> str:
    campaign_hash = hashlib.sha256(campaign.encode()).hexdigest()[:6]
    value = f"vks-{campaign[:12]}-{campaign_hash}-{point.slug}-a{generation[:8]}"
    if len(value) > 63:
        raise ValueError(f"generated Dynamo namespace is too long: {value}")
    return value
