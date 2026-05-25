# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Entry point for the cost-eval decision service.

Run via:

    python -m dynamo.cost_eval --config /path/to/cost_eval.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import AsyncIterator

from dynamo.cost_eval.config import CostEvalConfig
from dynamo.cost_eval.service import CostEvalService
from dynamo.cost_eval.wire import COMPONENT_NAME, ENDPOINT_NAME, CostEvalRequest
from dynamo.runtime import DistributedRuntime, dynamo_worker

logger = logging.getLogger(__name__)


async def _start(runtime: DistributedRuntime, config: CostEvalConfig) -> None:
    service = CostEvalService(runtime, config)
    await service.start()

    # Endpoint that the router's slow-path RPC targets — mirror the planner's
    # serve_endpoint shape. The dynamo runtime marshals the typed Pydantic
    # request/response between the Rust caller and this handler.
    endpoint = runtime.endpoint(f"{config.namespace}.{COMPONENT_NAME}.{ENDPOINT_NAME}")

    async def evaluate(request: dict) -> AsyncIterator[dict]:
        # dynamo runtime delivers the payload as a dict and expects a dict
        # back; we marshal the Pydantic models at the boundary explicitly
        # (Planner does the same — see
        # `components/src/dynamo/planner/__main__.py` precedent).
        parsed = CostEvalRequest.model_validate(request)
        response = service.evaluate_safe(parsed)
        yield response.model_dump()

    # Health-check probe uses an arbitrary request; the response shape is
    # what matters, not the numbers. Set a low-overlap request that the
    # service can answer even before any FPM observations have arrived
    # (the regression returns Some(...)/None per its readiness; either is
    # a valid health-check response).
    health_check_payload = CostEvalRequest(
        request_id="__health__",
        prompt_tokens=1,
        agg_kv_hit_rate=0.0,
        disagg_kv_hit_rate=0.0,
    ).model_dump()

    # serve_endpoint registers the health check and flips HealthStatus::Ready
    # once the handler is registered. Run it concurrently with the FPM drain
    # loop so the system server reports the service ready only after start()
    # finished.
    await asyncio.gather(
        endpoint.serve_endpoint(
            evaluate,  # type: ignore[arg-type]
            health_check_payload=health_check_payload,
        ),
        service.run(),
    )


def _parse_config() -> CostEvalConfig:
    parser = argparse.ArgumentParser(description="Dynamo Cost-Eval Decision Service")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string or path to a JSON/YAML config file",
    )
    args = parser.parse_args()
    return CostEvalConfig.from_config_arg(args.config)


@dynamo_worker()
async def worker(runtime: DistributedRuntime, config: CostEvalConfig):
    await _start(runtime, config)


def main():
    config = _parse_config()
    asyncio.run(worker(config))  # type: ignore[call-arg]


if __name__ == "__main__":
    main()
