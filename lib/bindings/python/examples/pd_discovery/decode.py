# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Mock decode worker that advertises a side-channel server to prefill workers.

It serves two endpoints from one process, so both share one instance id:
- `decode.generate`: the decode endpoint (mock output).
- `decode.side_channel`: returns the host and port of this worker's TCP server.
"""

import argparse
import asyncio
import socket

import uvloop

from dynamo.runtime import DistributedRuntime, dynamo_worker

NAMESPACE = "pd-discovery"

uvloop.install()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        help="Address that prefill workers use to reach this worker "
        "(default: this machine's hostname resolved to an IP)",
    )
    parser.add_argument(
        "--port", type=int, default=0, help="Server port; 0 picks a free port"
    )
    return parser.parse_args()


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    args = parse_args()
    host = args.host or socket.gethostbyname(socket.gethostname())
    generate = runtime.endpoint(f"{NAMESPACE}.decode.generate")
    side_channel = runtime.endpoint(f"{NAMESPACE}.decode.side_channel")
    my_id = generate.connection_id()

    # 1. Start the server that prefill workers connect to. It stands in for
    #    whatever the prefill side needs to reach, such as a KV receiver.
    async def on_connect(reader, writer):
        line = (await reader.readline()).decode().strip()
        print(f"decode {my_id:x}: received {line!r}")
        writer.write(f"ack from decode {my_id:x}\n".encode())
        await writer.drain()
        writer.close()

    server = await asyncio.start_server(on_connect, "0.0.0.0", args.port)
    port = server.sockets[0].getsockname()[1]
    print(f"decode {my_id:x}: side-channel server on {host}:{port}")

    # 2. Advertise the server: the side_channel endpoint returns its address.
    async def side_channel_handler(request):
        yield {"host": host, "port": port}

    async def generate_handler(request):
        yield f"decoded by {my_id:x}: {request}"

    await asyncio.gather(
        generate.serve_endpoint(generate_handler),
        side_channel.serve_endpoint(side_channel_handler),
    )


asyncio.run(worker())
