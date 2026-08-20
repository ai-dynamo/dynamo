# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Worker-local HTTP route for updating model routing taints."""

from __future__ import annotations

from typing import Any

from dynamo.llm import update_model_taints
from dynamo.runtime import DistributedRuntime, Endpoint

MODEL_TAINT_ROUTE = "update/model_taints"
TOPOLOGY_TAINT_PREFIX = "dynamo.topology/"


def register_model_taint_route(runtime: DistributedRuntime, endpoint: Endpoint) -> None:
    """Register POST /engine/update/model_taints on the system status server."""

    async def _update_model_taints(body: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(body, dict):
            raise ValueError("request body must be a JSON object")

        taints = body.get("taints")
        if not isinstance(taints, list) or not all(
            isinstance(taint, str) for taint in taints
        ):
            raise ValueError("'taints' must be a JSON array of strings")
        if reserved := next(
            (taint for taint in taints if taint.startswith(TOPOLOGY_TAINT_PREFIX)),
            None,
        ):
            raise ValueError(
                f"taint '{reserved}' uses reserved prefix '{TOPOLOGY_TAINT_PREFIX}'"
            )

        unique_taints = set(taints)
        await update_model_taints(endpoint, unique_taints)
        return {
            "status": "ok",
            "taints": sorted(unique_taints),
        }

    runtime.register_engine_route(MODEL_TAINT_ROUTE, _update_model_taints)
