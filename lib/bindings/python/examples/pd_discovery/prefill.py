# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock prefill worker that watches decode workers and connects to each new one.

While it serves `prefill.generate`, a background task polls the decode
workers. For each new decode worker it asks for the worker's side-channel
address and connects to it; when a decode worker disappears it forgets it.
"""

import argparse
import asyncio

import uvloop

from dynamo.runtime import Client, DistributedRuntime, dynamo_worker

NAMESPACE = "pd-discovery"

uvloop.install()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--interval", type=float, default=2.0, help="Seconds between polls"
    )
    return parser.parse_args()


async def on_new_decode(client: Client, decode_id: int, my_id: int):
    """Look up a decode worker's address, then run the operation against it."""
    # Ask that decode worker, by instance id, for its side-channel address.
    async for address in await client.direct({}, decode_id, annotated=False):
        host, port = address["host"], address["port"]

    # The operation: connect to the server on the decode side.
    reader, writer = await asyncio.open_connection(host, port)
    writer.write(f"hello from prefill {my_id:x}\n".encode())
    await writer.drain()
    reply = (await reader.readline()).decode().strip()
    writer.close()
    await writer.wait_closed()
    print(f"prefill {my_id:x}: decode {decode_id:x} at {host}:{port} -> {reply!r}")
    return host, port


async def watch_decode_workers(
    runtime: DistributedRuntime, my_id: int, interval: float
):
    client = await runtime.endpoint(f"{NAMESPACE}.decode.side_channel").client()
    known = {}  # (decode instance id, transport address) -> (host, port)
    while True:
        # A local snapshot that Dynamo keeps current from discovery. The
        # transport address tells a restarted worker apart: on Kubernetes, a
        # restarted container keeps its pod's instance id but gets a new port.
        current = {(i.instance_id, i.transport.address) for i in client.instances()}

        for key in known.keys() - current:
            host, port = known.pop(key)
            print(f"prefill {my_id:x}: decode {key[0]:x} at {host}:{port} left")

        for key in current - known.keys():
            decode_id = key[0]
            # A worker can stop between the snapshot and this call, so log the
            # failure and retry on the next poll instead of stopping the watch.
            try:
                known[key] = await on_new_decode(client, decode_id, my_id)
            except Exception as e:
                print(f"prefill {my_id:x}: decode {decode_id:x} failed: {e}")

        await asyncio.sleep(interval)


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
