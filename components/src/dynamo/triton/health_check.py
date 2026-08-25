# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triton-specific health check payload for tensor-inference workers."""

from __future__ import annotations

from typing import Any

from dynamo.health_check import HEALTH_CHECK_KEY, HealthCheckPayload


class TritonHealthCheckPayload(HealthCheckPayload):
    """Readiness-only payload for Triton tensor workers.

    Carries no request body; the handler short-circuits health probes and
    answers with Server.ready() / Model.ready() rather than invoking Triton.
    """

    def __init__(self, model_name: str) -> None:
        self.default_payload: dict[str, Any] = {"model": model_name}
        super().__init__()

    def to_dict(self) -> dict[str, Any]:
        # Stamp the canary marker last so an operator override via
        # DYN_HEALTH_CHECK_PAYLOAD can't strip it.
        payload = dict(super().to_dict())
        payload[HEALTH_CHECK_KEY] = True
        return payload
