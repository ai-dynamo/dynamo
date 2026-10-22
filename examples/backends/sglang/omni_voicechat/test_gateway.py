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

"""CPU tests for the gateway's own duplex and connection-lifecycle behavior."""

import asyncio
import unittest

from aiohttp import ClientSession, WSServerHandshakeError, web
from aiohttp.test_utils import TestClient, TestServer
from gateway import CONNECTIONS, gateway


class Discovery:
    def __init__(self, url):
        self.url = url
        self.calls = 0

    async def round_robin(self, request, annotated):
        self.calls += 1

        async def stream():
            yield {"ready": True, "url": self.url, "instance_id": 42}

        return stream()


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_worker_is_unavailable(self):
        class Unavailable:
            async def round_robin(self, request, annotated):
                # Match the exception type emitted by the real runtime binding.
                raise Exception("Unavailable: No workers available")  # noqa: TRY002

        async with (
            ClientSession() as session,
            TestClient(TestServer(await gateway(Unavailable(), session))) as browser,
        ):
            response = await browser.get("/health")
            self.assertEqual(response.status, 503)

    async def test_independent_duplex_admission_and_disconnect_cleanup(self):
        opened, closed = asyncio.Queue(), asyncio.Queue()
        native = web.Application()

        async def realtime(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await opened.put(dict(request.query))
            await ws.send_str("server-first")
            try:
                async for message in ws:
                    await ws.send_bytes(message.data)
            finally:
                await closed.put(True)
            return ws

        native.router.add_get("/v1/realtime", realtime)
        async with TestServer(native) as upstream, ClientSession() as session:
            discovery = Discovery(str(upstream.make_url("")).rstrip("/"))
            app = await gateway(discovery, session)
            async with TestClient(TestServer(app)) as browser:
                async with browser.ws_connect("/v1/realtime?intent=conversation") as ws:
                    self.assertEqual(ws._response.headers["X-Dynamo-Backend"], "42")
                    self.assertEqual((await opened.get())["intent"], "conversation")
                    self.assertEqual(await ws.receive_str(), "server-first")
                    await ws.send_bytes(b"caller-audio")
                    self.assertEqual(await ws.receive_bytes(), b"caller-audio")
                    with self.assertRaises(WSServerHandshakeError) as error:
                        await browser.ws_connect("/v1/realtime")
                    self.assertEqual(error.exception.status, 503)
                    self.assertEqual(discovery.calls, 1)
                await asyncio.wait_for(closed.get(), 2)
                for _ in range(100):
                    if not app[CONNECTIONS]:
                        break
                    await asyncio.sleep(0.01)
                self.assertFalse(app[CONNECTIONS])
                async with browser.ws_connect("/v1/realtime") as ws:
                    self.assertEqual(await ws.receive_str(), "server-first")

    async def test_shutdown_closes_active_upstream(self):
        closed = asyncio.Event()
        native = web.Application()

        async def realtime(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_str("ready")
            async for _ in ws:
                pass
            closed.set()
            return ws

        native.router.add_get("/v1/realtime", realtime)
        async with TestServer(native) as upstream, ClientSession() as session:
            app = await gateway(
                Discovery(str(upstream.make_url("")).rstrip("/")), session
            )
            browser = TestClient(TestServer(app))
            await browser.start_server()
            ws = await browser.ws_connect("/v1/realtime")
            self.assertEqual(await ws.receive_str(), "ready")
            await browser.close()
            await asyncio.wait_for(closed.wait(), 2)


if __name__ == "__main__":
    unittest.main()
