# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock prefill worker that finds decode workers in discovery and connects.

While it serves `prefill.generate`, it watches the runtime configs that decode
workers publish in their model cards. Each decode worker publishes its
side-channel address there, so this worker reads it from discovery instead of
asking the decode worker. It connects to each new decode worker and drops the
ones that leave.
"""

import argparse
import asyncio
import json

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker

NAMESPACE = "pd-discovery"
SIDE_CHANNEL_KEY = "side_channel"

uvloop.install()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Most seconds between checks; a change in discovery wakes it sooner",
    )
    return parser.parse_args()


async def connect(decode_id: int, address: dict, my_id: int):
    """The operation to run against a new decode worker: connect to its server."""
    host, port = address["host"], address["port"]
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(f"hello from prefill {my_id:x}\n".encode())
    await writer.drain()
    reply = (await reader.readline()).decode().strip()
    writer.close()
    await writer.wait_closed()
    print(f"prefill {my_id:x}: decode {decode_id:x} at {host}:{port} -> {reply!r}")


async def watch_decode_workers(
    runtime: DistributedRuntime, my_id: int, interval: float
):
    # Discovery keeps this view of every ready decode worker's runtime config
    # up to date. Reading it makes no network calls.
    decode = runtime.endpoint(f"{NAMESPACE}.decode.generate")
    configs = await decode.watch_runtime_configs()
    known = {}  # (decode instance ID, published side channel) -> address
    while True:
        current = {
            (decode_id, side_channel)
            for decode_id, config in configs.snapshot().items()
            if (side_channel := config.runtime_data.get(SIDE_CHANNEL_KEY))
        }

        for key in known.keys() - current:
            address = known.pop(key)
            print(
                f"prefill {my_id:x}: decode {key[0]:x} at "
                f"{address['host']}:{address['port']} left"
            )

        for key in current - known.keys():
            decode_id, side_channel = key
            address = json.loads(side_channel)
            # A worker can stop before this connection, so log the failure and
            # retry on a later check instead of stopping the watch.
            try:
                await connect(decode_id, address, my_id)
            except OSError as e:
                print(f"prefill {my_id:x}: decode {decode_id:x} failed: {e}")
                continue
            known[key] = address

        # Wake when a decode worker joins, leaves, or republishes, or after
        # `interval` seconds to retry failed connections.
        try:
            await asyncio.wait_for(configs.changed(), timeout=interval)
        except TimeoutError:
            pass


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    args = parse_args()
    endpoint = runtime.endpoint(f"{NAMESPACE}.prefill.generate")
    my_id = endpoint.connection_id()

    async def generate_handler(request):
        yield f"prefilled by {my_id:x}: {request}"

    # Watch decode workers while serving prefill requests.
    await asyncio.gather(
        watch_decode_workers(runtime, my_id, args.interval),
        endpoint.serve_endpoint(generate_handler),
    )


asyncio.run(worker())
