# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dynamo discovers the realtime worker; one WebSocket stays on that worker."""

import argparse
import asyncio
import logging
import os
import signal

from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web

from dynamo.runtime import DistributedRuntime

LOG = logging.getLogger("voicechat")
MODEL = "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B"
MAX_MESSAGE_BYTES = 1024 * 1024
CONNECTIONS = web.AppKey("connections", set)


async def describe(client):
    async with asyncio.timeout(5):
        stream = await client.round_robin({"op": "describe"}, annotated=False)
        replies = [reply async for reply in stream]
    if len(replies) != 1 or not replies[0].get("ready"):
        raise web.HTTPServiceUnavailable(text="VoiceChat backend is starting")
    return replies[0]


async def relay(source, target):
    async for message in source:
        if message.type == WSMsgType.TEXT:
            await target.send_str(message.data)
        elif message.type == WSMsgType.BINARY:
            await target.send_bytes(message.data)
        elif message.type == WSMsgType.ERROR:
            raise source.exception()


async def gateway(client, session):
    app = web.Application(client_max_size=MAX_MESSAGE_BYTES)
    connections = set()
    app[CONNECTIONS] = connections

    async def backend():
        try:
            return await describe(client)
        except web.HTTPException:
            raise
        except Exception as error:
            # The runtime exposes discovery/RPC failures as plain Exception.
            LOG.info("backend_unavailable: %s", error)
            raise web.HTTPServiceUnavailable(
                text="No ready Dynamo realtime worker"
            ) from error

    async def status(request):
        selected = await backend()
        return web.json_response({**selected, "active_connections": len(connections)})

    async def realtime(request):
        # shortcut: this demo admits one connection; use backend-wide reservations
        # before running multiple gateway replicas against a single-session worker.
        if connections:
            raise web.HTTPServiceUnavailable(
                text="VoiceChat is in use; end the other conversation first"
            )
        token = object()
        connections.add(token)
        downstream = None
        try:
            selected = await backend()
            url = selected["url"] + "/v1/realtime"
            async with session.ws_connect(
                url, params=request.query, max_msg_size=MAX_MESSAGE_BYTES
            ) as upstream:
                downstream = web.WebSocketResponse(
                    max_msg_size=MAX_MESSAGE_BYTES, heartbeat=20
                )
                downstream.headers["X-Dynamo-Backend"] = str(selected["instance_id"])
                await downstream.prepare(request)
                connections.remove(token)
                connections.add(downstream)
                LOG.info("session_open instance=%s", selected["instance_id"])
                tasks = [
                    asyncio.create_task(relay(downstream, upstream)),
                    asyncio.create_task(relay(upstream, downstream)),
                ]
                try:
                    done, _ = await asyncio.wait(
                        tasks, return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                finally:
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    code = upstream.close_code or 1000
                    await downstream.close(
                        code=1011 if code in (1005, 1006, 1015) else code
                    )
                    LOG.info("session_close instance=%s", selected["instance_id"])
                return downstream
        except ClientError as error:
            if downstream is None:
                raise web.HTTPServiceUnavailable(
                    text="VoiceChat backend refused the connection"
                ) from error
            raise
        finally:
            connections.discard(token)
            connections.discard(downstream)

    async def asset(request):
        selected = await backend()
        async with session.get(
            selected["url"] + request.path, params=request.query
        ) as response:
            body = await response.read()
            if request.path == "/" and response.status == 200:
                body = body.replace(
                    b"<title>Duplex Live</title>",
                    b"<title>Dynamo + Nemotron VoiceChat</title>",
                )
                body = body.replace(
                    b"full duplex \xc2\xb7 sglang-omni", b"Dynamo + SGLang-Omni"
                )
            return web.Response(
                body=body,
                status=response.status,
                headers={
                    "Content-Type": response.headers.get(
                        "Content-Type", "application/octet-stream"
                    ),
                    "Cache-Control": "no-store",
                },
            )

    async def shutdown(application):
        await asyncio.gather(
            *(
                ws.close(code=1001, message=b"Gateway shutting down")
                for ws in tuple(connections)
                if isinstance(ws, web.WebSocketResponse)
            )
        )

    app.on_shutdown.append(shutdown)
    app.router.add_get("/health", status)
    app.router.add_get("/api/status", status)
    app.router.add_get("/v1/realtime", realtime)
    for path in ("/", "/api/presets", "/v1/realtime/capabilities", "/static/{path:.*}"):
        app.router.add_get(path, asset)
    return app


async def run(args):
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "etcd", "tcp", event_plane="zmq"
    )
    endpoint = runtime.endpoint(f"{args.namespace}.voicechat.realtime")
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)
    async with ClientSession(
        timeout=ClientTimeout(total=30, sock_connect=5)
    ) as session:
        try:
            if args.mode == "worker":

                async def handler(request):
                    try:
                        async with session.get(
                            args.backend_url + "/health"
                        ) as response:
                            ready = response.status == 200
                    except (ClientError, TimeoutError):
                        ready = False
                    yield {
                        "ready": ready,
                        "url": args.backend_url,
                        "instance_id": endpoint.connection_id(),
                        "model": MODEL,
                        "max_sessions": 1,
                    }

                serving = asyncio.ensure_future(endpoint.serve_endpoint(handler))
                stopping = asyncio.create_task(stop.wait())
                try:
                    done, _ = await asyncio.wait(
                        (serving, stopping), return_when=asyncio.FIRST_COMPLETED
                    )
                    for task in done:
                        task.result()
                finally:
                    runtime.shutdown()
                    serving.cancel()
                    stopping.cancel()
                    await asyncio.gather(serving, stopping, return_exceptions=True)
            else:
                client = await endpoint.client()
                app = await gateway(client, session)
                runner = web.AppRunner(app)
                await runner.setup()
                await web.TCPSite(runner, "0.0.0.0", args.port).start()
                LOG.info("Dynamo realtime gateway listening on %s", args.port)
                try:
                    await stop.wait()
                finally:
                    await runner.cleanup()
        finally:
            runtime.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("gateway", "worker"))
    parser.add_argument("--namespace", default="voicechat")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--backend-url", default=os.environ.get("OMNI_URL", "http://127.0.0.1:8097")
    )
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(parser.parse_args()))
