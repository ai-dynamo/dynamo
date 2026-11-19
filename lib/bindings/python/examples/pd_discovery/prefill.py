# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock prefill worker that finds decode workers in discovery and connects.

While it serves `prefill.generate`, it reads the runtime configs that decode
workers registered, every two seconds. Each decode worker publishes its
side-channel address in its runtime config, so this worker reads it from
discovery instead of asking the decode worker. It connects to each new decode
worker and drops the ones that leave.
"""

import asyncio
import json

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker

NAMESPACE = "pd-discovery"
SIDE_CHANNEL_KEY = "side_channel"
INTERVAL = 2.0  # seconds between checks
# Seconds for one connection, so a stuck decode worker can't block the watch.
CONNECT_TIMEOUT = 5.0

uvloop.install()


async def connect(decode_id: int, address: dict, my_id: int):
    """The operation to run against a new decode worker: connect to its server."""
    host, port = address["host"], address["port"]
    async with asyncio.timeout(CONNECT_TIMEOUT):
        reader, writer = await asyncio.open_connection(host, port)
        try:
            writer.write(f"hello from prefill {my_id:x}\n".encode())
            await writer.drain()
            reply = (await reader.readline()).decode().strip()
        finally:
            writer.close()
    print(f"prefill {my_id:x}: decode {decode_id:x} at {host}:{port} -> {reply!r}")


async def watch_decode_workers(runtime: DistributedRuntime, my_id: int):
    decode = runtime.endpoint(f"{NAMESPACE}.decode.generate")
    known = {}  # (decode instance ID, incarnation) -> address
    while True:
        # Read the address each decode worker published in its runtime config.
        current = {}
        for decode_id, config in (await decode.runtime_configs()).items():
            if published := config.runtime_data.get(SIDE_CHANNEL_KEY):
                address = json.loads(published)
                current[decode_id, address["incarnation"]] = address

        for key in known.keys() - current.keys():
            address = known.pop(key)
            print(
                f"prefill {my_id:x}: decode {key[0]:x} at "
                f"{address['host']}:{address['port']} left"
            )

        for key in current.keys() - known.keys():
            # A worker can stop or stall before this connection, so log the
            # failure (TimeoutError is an OSError) and retry on the next check
            # instead of stopping the watch.
            try:
                await connect(key[0], current[key], my_id)
            except OSError as e:
                print(f"prefill {my_id:x}: decode {key[0]:x} failed: {e!r}")
                continue
            known[key] = current[key]

        await asyncio.sleep(INTERVAL)


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint(f"{NAMESPACE}.prefill.generate")
    my_id = endpoint.connection_id()

    async def generate_handler(request):
        yield f"prefilled by {my_id:x}: {request}"

    # Watch decode workers while serving prefill requests.
    await asyncio.gather(
        watch_decode_workers(runtime, my_id),
        endpoint.serve_endpoint(generate_handler),
    )


asyncio.run(worker())
