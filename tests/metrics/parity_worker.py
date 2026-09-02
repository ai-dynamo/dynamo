# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal worker for the OTLP/Prometheus parity test.

Deliberately dependency-free beyond the runtime itself: the test is about the
metrics wiring, so anything an engine would drag in is noise that can break the
test for unrelated reasons.

Registers an engine-style registry through the same helper the real backends
use. That helper is what registers *both* callbacks -- exposition text for
``/metrics``, typed families for OTLP -- so this worker exercises the path
where the two surfaces can actually drift apart. A worker with only the
runtime's built-in metrics would pass even if engine metrics reached one
surface and not the other.
"""

import uvloop
from prometheus_client import CollectorRegistry, Counter, Histogram

from dynamo.common.utils.prometheus import register_engine_metrics_callback
from dynamo.runtime import DistributedRuntime, dynamo_worker

ENGINE_PREFIX = "parity:"


@dynamo_worker()
async def parity_worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("parity.echo.generate")

    registry = CollectorRegistry()
    Counter(
        f"{ENGINE_PREFIX}requests",
        "Requests seen by the parity worker",
        registry=registry,
    ).inc(3)
    Histogram(
        f"{ENGINE_PREFIX}latency_seconds",
        "Latency observed by the parity worker",
        buckets=[0.1, 1.0],
        registry=registry,
    ).observe(0.5)

    register_engine_metrics_callback(
        endpoint,
        registry,
        metric_prefix_filters=[ENGINE_PREFIX],
        namespace_name="parity",
        component_name="echo",
        endpoint_name="generate",
    )

    await endpoint.serve_endpoint(generate)


async def generate(request):
    yield {"echo": request}


if __name__ == "__main__":
    uvloop.run(parity_worker())
