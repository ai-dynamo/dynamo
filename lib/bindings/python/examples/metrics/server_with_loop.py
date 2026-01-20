#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating metrics updates via background loop instead of callback.

This shows an alternative approach where:
1. Metrics are created and registered with an endpoint
2. A background thread continuously updates metrics in a loop
3. No callback is used - metrics are updated directly by the thread
4. The metrics are automatically served via the /metrics endpoint

Usage:
    DYN_SYSTEM_PORT=8081 ./server_with_loop.py

    # In another terminal, query the metrics:
    curl http://localhost:8081/metrics
"""

import asyncio
import threading
import time

import uvloop

from dynamo.prometheus_metrics import Gauge, IntCounter, IntGauge, IntGaugeVec
from dynamo.runtime import Component, DistributedRuntime, Endpoint, dynamo_worker


def metrics_updater_thread(
    request_total_slots: IntGauge,
    gpu_cache_usage_perc: Gauge,
    worker_active_requests: IntGaugeVec,
    update_count: IntCounter,
):
    """Background thread that continuously updates metrics."""
    print("[python] Metrics updater thread started")

    while True:
        update_count.inc()
        count = update_count.get()

        # Update simple metrics
        request_total_slots.set(1024 + count)
        gpu_cache_usage_perc.set(0.01 + (count * 0.01))

        # Update vector metrics with varying values
        worker_active_requests.set(
            5 + (count % 10), {"worker_id": "worker_1", "model": "llama-3"}
        )
        worker_active_requests.set(
            3 + (count % 5), {"worker_id": "worker_2", "model": "llama-3"}
        )

        print(f"[python] Updated metrics in loop (iteration #{count})")

        # Update every 2 seconds
        time.sleep(2)


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    await init(runtime)


async def init(runtime: DistributedRuntime):
    # Create component and endpoint
    component: Component = runtime.namespace("ns557").component("cp557")

    endpoint: Endpoint = component.endpoint("ep557")

    # Create metrics using the endpoint's metrics property
    print("[python] Creating metrics...")

    request_total_slots: IntGauge = endpoint.metrics.create_intgauge(
        "request_total_slots", "Total request slots available"
    )
    gpu_cache_usage_perc: Gauge = endpoint.metrics.create_gauge(
        "gpu_cache_usage_percent", "GPU cache usage percentage"
    )

    worker_active_requests: IntGaugeVec = endpoint.metrics.create_intgaugevec(
        "worker_active_requests",
        "Active requests per worker",
        ["worker_id", "model"],
    )

    update_count: IntCounter = endpoint.metrics.create_intcounter(
        "update_count",
        "Number of times metrics were updated",
        [("update_method", "background_thread")],
    )

    print(f"[python] Created IntGauge: {request_total_slots.name()}")
    print(f"[python] Created Gauge: {gpu_cache_usage_perc.name()}")
    print(f"[python] Created IntGaugeVec: {worker_active_requests.name()}")
    print(f"[python] Created IntCounter: {update_count.name()}")
    print("[python] Metrics automatically registered with endpoint!")

    # Set initial values
    print("[python] Setting initial metric values...")
    request_total_slots.set(1024)
    gpu_cache_usage_perc.set(0.00)
    worker_active_requests.set(5, {"worker_id": "worker_1", "model": "llama-3"})
    worker_active_requests.set(3, {"worker_id": "worker_2", "model": "llama-3"})

    # Start background thread to update metrics
    print("[python] Starting background thread to update metrics...")
    updater = threading.Thread(
        target=metrics_updater_thread,
        args=(
            request_total_slots,
            gpu_cache_usage_perc,
            worker_active_requests,
            update_count,
        ),
        daemon=True,
    )
    updater.start()

    print("[python] ✅ Metrics are now registered and served via /metrics endpoint")
    print("[python]    Metrics are being updated every 2 seconds by background thread")
    print(
        "[python]    Check the system status server port to see them in Prometheus format"
    )

    # Note: This example does not call serve_endpoint() to keep it simple.
    # In a real service, you would call: await endpoint.serve_endpoint(handler, ...)
    # Keep running so metrics endpoint stays up
    _ = await asyncio.Event().wait()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
