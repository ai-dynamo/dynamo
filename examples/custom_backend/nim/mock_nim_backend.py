#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Mock NIM Backend Server for Metrics Testing

This server mocks a NIM (NVIDIA Inference Microservices) backend that exposes
runtime statistics via the runtime_stats endpoint.

NOTE: This is temporary code for testing purposes only. Once NIM starts using
Dynamo backend components natively, this mock server and the associated NIM
metrics polling code in the frontend will be removed. The NIM-specific metrics
collection exists only as a bridge until NIM adopts the Dynamo runtime.

The server demonstrates:
- Dynamic metric generation (gauges and counters)
- Proper async generator pattern for Dynamo endpoints
- JSON-encoded metric responses compatible with the frontend metrics collector
"""
import asyncio
import json
import time
from typing import Any, AsyncGenerator

import uvloop

from dynamo.runtime import DistributedRuntime

# Global counter for incrementing metrics
request_count = 0


async def handle_stats_request(request: Any) -> AsyncGenerator[str, None]:
    """Mock stats handler - returns incrementing metrics for testing

    Args:
        request: JsonLike input from the client (can be dict, list, str, int, float, bool, or None)

    Yields:
        str: JSON string of stats dict conforming to the runtime_stats schema
    """
    global request_count
    request_count += 1

    print(f"Received stats request #{request_count}: {request!r}")

    # Simulate changing metrics
    kv_cache_usage = 0.3 + (request_count % 10) * 0.07  # Cycles between 0.3 and 0.93
    gpu_utilization = 50 + (request_count % 20) * 2.5  # Cycles between 50 and 97.5
    active_requests = request_count % 15  # Cycles 0-14

    stats = {
        "schema_version": 1,
        "worker_id": "mock-worker-1",
        "backend": "vllm",
        "ts": int(time.time()),
        "metrics": {
            "gauges": {
                "kv_cache_usage_perc": round(kv_cache_usage, 2),
                "gpu_utilization_perc": round(gpu_utilization, 2),
                "active_requests": active_requests,
            },
        },
    }
    # Yield as JSON string for Rust Annotated<String> compatibility
    yield json.dumps(stats)


async def worker(runtime: DistributedRuntime):
    import argparse

    parser = argparse.ArgumentParser(description="Mock NIM Backend Server")
    parser.add_argument(
        "--custom-backend-metrics-endpoint",
        type=str,
        default="nim.backend.runtime_stats",
        help="Custom backend metrics endpoint in format 'namespace.component.endpoint' (default: 'nim.backend.runtime_stats')",
    )
    parser.add_argument(
        "--use-etcd",
        action="store_true",
        help="Use etcd for service discovery (dynamic mode). Default is static mode (no etcd).",
    )
    args = parser.parse_args()

    # Parse endpoint (namespace.component.endpoint)
    parts = args.custom_backend_metrics_endpoint.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"Invalid endpoint format. Expected 'namespace.component.endpoint', got: {args.custom_backend_metrics_endpoint}"
        )

    namespace, comp_name, endpoint_name = parts

    component = runtime.namespace(namespace).component(comp_name)

    stats_endpoint = component.endpoint(endpoint_name)
    print(
        f"Mock NIM stats server started on {namespace}/{comp_name}/{endpoint_name} endpoint"
    )
    print(
        "Exposing incrementing metrics: kv_cache_usage_perc, gpu_utilization_perc, active_requests, memory_used_gb, counters"
    )

    await stats_endpoint.serve_endpoint(handle_stats_request)  # type: ignore[arg-type]


async def main():
    import argparse

    # Parse args before calling dynamo_worker to determine static mode
    parser = argparse.ArgumentParser(
        description="Mock NIM Backend Server", add_help=False
    )
    parser.add_argument("--use-etcd", action="store_true")
    args, _ = parser.parse_known_args()

    # Set static mode based on --use-etcd flag (default is static/no etcd)
    is_static = not args.use_etcd

    loop = asyncio.get_running_loop()
    if is_static:
        runtime = DistributedRuntime(loop, "file", "nats")
    else:
        runtime = DistributedRuntime(loop, "etcd", "nats")

    try:
        await worker(runtime)  # type: ignore[arg-type]
    finally:
        runtime.shutdown()


if __name__ == "__main__":
    uvloop.run(main())
