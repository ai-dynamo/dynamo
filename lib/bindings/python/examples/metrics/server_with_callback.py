#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Example demonstrating the new typed Prometheus metrics API for declarative metrics registration.

This shows how Python code can:
1. Create typed metric objects directly (Gauge, IntGauge, GaugeVec, IntGaugeVec, etc.)
2. Register them with an endpoint
3. Update their values using type-safe methods (set for gauges, inc for counters)
4. The metrics are automatically served via the /metrics endpoint

Usage:
    DYN_SYSTEM_PORT=8081 ./server_with_callback.py

    # In another terminal, query the metrics:
    curl http://localhost:8081/metrics
"""

import asyncio

import uvloop

# Note that these imports are for type hints only. They cannot be instantiated directly.
# You can instantiate them using the endpoint.metrics.create_*() methods.
from dynamo.prometheus_metrics import Gauge, IntCounter, IntGauge, IntGaugeVec
from dynamo.runtime import Component, DistributedRuntime, Endpoint, dynamo_worker


@dynamo_worker()
async def worker(runtime: DistributedRuntime) -> None:
    await init(runtime)


async def init(runtime: DistributedRuntime):
    # Create component and endpoint
    component: Component = runtime.namespace("ns556").component("cp556")
    await component.create_service()

    endpoint: Endpoint = component.endpoint("ep556")

    # Step 1: Create metrics using the endpoint's metrics property
    print("[python] Creating metrics...")

    # Simple metrics (Gauge and IntGauge) - automatically registered
    request_total_slots: IntGauge = endpoint.metrics.create_intgauge(
        "request_total_slots", "Total request slots available"
    )
    gpu_cache_usage_perc: Gauge = endpoint.metrics.create_gauge(
        "gpu_cache_usage_percent", "GPU cache usage percentage"
    )

    # Vector metrics (IntGaugeVec with labels)
    worker_active_requests: IntGaugeVec = endpoint.metrics.create_intgaugevec(
        "worker_active_requests",
        "Active requests per worker",
        ["worker_id", "model"],
    )

    # Counter metric to track updates (with constant label values)
    update_count: IntCounter = endpoint.metrics.create_intcounter(
        "update_count",
        "Number of times metrics were updated",
        [("update_method", "callback")],
    )

    print(f"[python] Created IntGauge: {request_total_slots.name()}")
    print(f"[python] Created Gauge: {gpu_cache_usage_perc.name()}")
    print(f"[python] Created IntGaugeVec: {worker_active_requests.name()}")
    print(f"[python] Created IntCounter: {update_count.name()}")
    print("[python] Metrics automatically registered with endpoint!")

    # Step 2: Register a callback to update metrics on-demand
    print("[python] Registering metrics callback...")

    def update_metrics():
        """Called automatically before /metrics endpoint is scraped"""
        update_count.inc()
        # Update metrics with fresh values
        count = update_count.get()
        request_total_slots.set(1024 + count)
        gpu_cache_usage_perc.set(0.01 + (count * 0.01))
        print(f"[python] Updated metrics (call #{count})")

    endpoint.metrics.register_callback(update_metrics)
    print("[python] update (metrics) callback registered!")

    # Step 3: Set initial values and test vector metrics
    print("[python] Setting initial metric values...")
    request_total_slots.set(1024)
    gpu_cache_usage_perc.set(0.00)
    print(f"[python] request_total_slots = {request_total_slots.get()}")
    print(f"[python] gpu_cache_usage_perc = {gpu_cache_usage_perc.get()}")

    print("[python] Updating vector metric with labels...")
    worker_active_requests.set(5, {"worker_id": "worker_1", "model": "llama-3"})
    worker_active_requests.set(3, {"worker_id": "worker_2", "model": "llama-3"})
    print("[python] worker_active_requests set for worker_1 and worker_2")

    # The metrics are now available at:
    # http://localhost:<system_status_port>/metrics
    print("[python] ✅ Metrics are now registered and served via /metrics endpoint")
    print(
        "[python]    Check the system status server port to see them in Prometheus format"
    )
    print(
        "[python]    Supported types: Counter, IntCounter, Gauge, IntGauge, Histogram, and their Vec variants"
    )

    # Note: This example does not call serve_endpoint() to keep it simple.
    # In a real service, you would call: await endpoint.serve_endpoint(handler, ...)
    # Keep running so metrics endpoint stays up
    _ = await asyncio.Event().wait()


if __name__ == "__main__":
    uvloop.install()
    asyncio.run(worker())
