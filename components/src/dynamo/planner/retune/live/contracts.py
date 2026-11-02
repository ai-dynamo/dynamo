# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Types shared by the live retune modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

# ------------------------------------------------------- frontend config


@dataclass(frozen=True)
class FrontendStartup:
    """The frontend router configuration parsed from the DGD spec.

    ``router_config`` is the startup ``KvRouterConfig`` JSON (only keys the
    frontend sets); ``startup_maps`` is ``policy_parameters(wt)`` for each
    worker type of the planner mode. ``fingerprint_source`` is a canonical
    JSON string of everything parsed that the state fingerprint covers.
    """

    router_mode: str  # "kv_router" or "round_robin"
    router_config: dict[str, Any]
    startup_maps: dict[str, dict[str, float]]
    frontend_component: str  # DGD component name of the frontend
    replicas: int
    replica_sync: bool
    fingerprint_source: str


@dataclass(frozen=True)
class PremiseReport:
    """Spec-class premise evaluation (P4, P7-P10; P14 is the actuator's).

    ``p4_ok`` alone decides whether a controller can be built; ``spec_ok``
    is the conjunction of every spec premise. ``reasons`` explain each
    failed premise as ``"P<n>: ..."``; ``warnings`` carry warn-only ones.
    """

    p4_ok: bool
    spec_ok: bool
    reasons: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    startup: Optional[FrontendStartup] = None
