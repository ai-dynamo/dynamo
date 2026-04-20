# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Subprocess helper for test_router_per_worker_config.py.

Usage:
    python _counter_worker.py <count_file> <device_type> <endpoint_path>

    count_file:    path to file where the request count is written after each request
    device_type:   "cpu" sets CUDA_VISIBLE_DEVICES=""; "gpu" sets CUDA_VISIBLE_DEVICES="0"
    endpoint_path: dotted endpoint path, e.g. "test.counter.generate"
"""

import asyncio
import os
import sys

request_count = 0

# Register needs a model path, so we use a HF model name here.
HF_MODEL_NAME = "Qwen/Qwen3-0.6B"


async def generate(request, context):
    global request_count
    request_count += 1
    with open(count_file, "w") as f:
        f.write(str(request_count))
    yield {"ok": True}


async def main():
    global count_file

    count_file = sys.argv[1]
    device_type = sys.argv[2]
    endpoint_path = sys.argv[3]

    # Set device type BEFORE importing dynamo so the Rust side sees the correct env var
    # when the endpoint instance registers itself with the discovery system.
    if device_type == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    elif device_type == "gpu":
        # Any non-empty, non-"-1", non-"none" value → Cuda in endpoint_device_type().
        # "0" works even without a physical GPU since detection is purely env-var based.
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    from dynamo.llm import (
        ModelInput,
        ModelType,
        RouterConfig,
        RouterMode,
        register_model,
    )
    from dynamo.runtime import DistributedRuntime

    loop = asyncio.get_event_loop()
    runtime = DistributedRuntime(loop, "file", "nats")
    endpoint = runtime.endpoint(endpoint_path)

    await register_model(
        ModelInput.Text,
        ModelType.Chat,
        endpoint,
        HF_MODEL_NAME,
        "counter",
        router_config=RouterConfig(RouterMode.DeviceAwareWeighted),
    )

    await endpoint.serve_endpoint(generate)


asyncio.run(main())
