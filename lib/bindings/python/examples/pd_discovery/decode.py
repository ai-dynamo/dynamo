# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock decode worker that publishes its side-channel address in discovery.

It starts a TCP server and puts the server's address in the runtime config of
its model card. Prefill workers read the address from discovery, so they never
have to ask this worker for it.
"""

import asyncio
import json
import socket
import uuid

import uvloop

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime, dynamo_worker

NAMESPACE = "pd-discovery"
SIDE_CHANNEL_KEY = "side_channel"

uvloop.install()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    # The pod IP on Kubernetes. Across machines without Kubernetes, use an
    # address that the prefill workers can reach.
    host = socket.gethostbyname(socket.gethostname())
    endpoint = runtime.endpoint(f"{NAMESPACE}.decode.generate")
    my_id = endpoint.connection_id()

    # 1. Start the server that prefill workers connect to. It stands in for
    #    whatever the prefill side needs to reach, such as a KV receiver.
    async def on_connect(reader, writer):
        line = (await reader.readline()).decode().strip()
        print(f"decode {my_id:x}: received {line!r}")
        writer.write(f"ack from decode {my_id:x}\n".encode())
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(on_connect, "0.0.0.0", 0)  # any free port
    port = server.sockets[0].getsockname()[1]
    print(f"decode {my_id:x}: side-channel server on {host}:{port}")

    # 2. Publish the address in the runtime config of this worker's model card.
    #    A real decode worker already registers its model; only the
    #    set_engine_specific call is new. The incarnation changes on every
    #    start, so prefill workers can tell a restarted worker from the old
    #    one: on Kubernetes, a restarted container keeps its instance ID.
    runtime_config = ModelRuntimeConfig()
    side_channel = {"host": host, "port": port, "incarnation": uuid.uuid4().hex}
    runtime_config.set_engine_specific(SIDE_CHANNEL_KEY, json.dumps(side_channel))
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        "pd-discovery-mock",
        worker_type=WorkerType.Decode,
        needs=[[WorkerType.Prefill]],
        runtime_config=runtime_config,
        skip_model_assets=True,
    )

    async def generate_handler(request):
        yield f"decoded by {my_id:x}: {request}"

    await endpoint.serve_endpoint(generate_handler)


asyncio.run(worker())
