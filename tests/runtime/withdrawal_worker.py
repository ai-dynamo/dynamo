# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU engine for the worker-withdrawal transport integration test."""

import asyncio
import sys
from types import SimpleNamespace

from dynamo._core import WorkerDraining
from dynamo.health_check import HEALTH_CHECK_KEY
from dynamo.llm import (
    ModelInput,
    ModelType,
    RouterConfig,
    RouterMode,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime
from dynamo.sglang.drain_controls import configure_worker_drain


async def main():
    namespace, name, model_path = sys.argv[1:]
    runtime = DistributedRuntime(
        asyncio.get_running_loop(),
        "file",
        "tcp",
        event_plane="zmq",
        response_plane="tcp",
    )
    endpoint = runtime.endpoint(f"{namespace}.backend.generate")
    ready, shutdown, terminal = asyncio.Event(), asyncio.Event(), asyncio.Event()
    release = asyncio.Event()
    counts = {"business": 0, "probes": 0, "rejected": 0, "held": False}
    handler = SimpleNamespace(
        generate_endpoint=endpoint,
        _engine_route_lock=asyncio.Lock(),
        engine=SimpleNamespace(tokenizer_manager=SimpleNamespace(is_pause=False)),
        config=SimpleNamespace(
            dynamo_args=SimpleNamespace(response_plane="tcp", engine_routes=[])
        ),
    )
    controller = configure_worker_drain(handler, runtime, ready, shutdown, terminal)

    async def generate(request, context):
        if request.get(HEALTH_CHECK_KEY):
            counts["probes"] += 1
            yield {"token_ids": [32]}
            return
        counts["business"] += 1
        yield {"token_ids": [32 if name == "a" else 33]}
        if name == "a" and request["stop_conditions"].get("max_tokens") == 64:
            counts["held"] = True
            await release.wait()
        yield {"token_ids": [32], "finish_reason": "stop"}

    admitted = controller.wrap(generate)

    async def observed(request, context):
        try:
            async for item in admitted(request, context):
                yield item
        except WorkerDraining:
            counts["rejected"] += 1
            raise

    client = await endpoint.client()

    async def state(body):
        return {
            **counts,
            "worker_id": endpoint.connection_id(),
            "instances": client.instance_ids(),
            **await controller.status({}),
        }

    async def unblock(body):
        release.set()
        return {"released": True}

    runtime.register_engine_route("test/state", state)
    runtime.register_engine_route("test/release", unblock)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        model_path,
        model_name="withdrawal-test",
        router_config=RouterConfig(RouterMode.RoundRobin),
        worker_type=WorkerType.Aggregated,
        ignore_weights=True,
    )
    ready.set()
    await endpoint.serve_endpoint(
        observed, health_check_payload={HEALTH_CHECK_KEY: True}
    )


if __name__ == "__main__":
    asyncio.run(main())
