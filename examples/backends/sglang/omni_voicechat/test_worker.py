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

"""Exercise the adapter's independent streams, worker admission and cleanup."""

import asyncio
import unittest

from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer
from worker import MODEL, VoiceChatWorker


class Context:
    stopped = False

    def id(self):
        return "test-session"

    def is_stopped(self):
        return self.stopped


async def stream(queue):
    while (event := await queue.get()) is not None:
        yield event


def update():
    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "model": MODEL,
            "output_modalities": ["audio"],
        },
    }


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def test_duplex_worker_reservation_and_idle_disconnect(self):
        closed = asyncio.Event()
        received = []
        app = web.Application()

        async def native(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            await ws.send_json(
                {"type": "session.created", "session": {"id": "native-session"}}
            )
            try:
                async for message in ws:
                    event = message.json()
                    received.append(event)
                    if event["type"] == "session.update":
                        await ws.send_json(
                            {
                                "type": "session.updated",
                                "event_id": "updated",
                                "session": {
                                    "sglang": {
                                        "granted": {
                                            "input_audio_format": {"rate": 16000},
                                            "output_audio_format": {"rate": 22050},
                                        }
                                    }
                                },
                            }
                        )
                    elif event["type"] == "input_audio_buffer.append":
                        await ws.send_json({"type": "sglang.input_audio.accepted"})
                        await ws.send_json(
                            {
                                "type": "response.output_audio.delta",
                                "delta": event["audio"],
                                "sglang": {"unit_id": "unit0"},
                            }
                        )
                    elif event["type"] == "sglang.input_audio.end":
                        await ws.send_json(
                            {
                                "type": "response.done",
                                "response": {
                                    "id": "response",
                                    "status": "completed",
                                    "status_details": {"reason": "stop"},
                                },
                            }
                        )
                        await ws.send_json(
                            {
                                "type": "sglang.input_audio.drained",
                                "event_id": "drained",
                            }
                        )
            finally:
                closed.set()
            return ws

        app.router.add_get("/v1/realtime", native)
        async with TestServer(app) as server, ClientSession() as session:
            worker = VoiceChatWorker(session, str(server.make_url("")).rstrip("/"), 42)
            context, queue = Context(), asyncio.Queue()
            await queue.put(update())
            events = worker.generate(stream(queue), context)
            configured = await asyncio.wait_for(anext(events), 2)
            self.assertEqual(configured["session"]["id"], "native-session")
            self.assertIsInstance(received[0]["event_id"], str)
            busy = worker.generate(stream(asyncio.Queue()), Context())
            self.assertEqual((await anext(busy))["error"]["code"], "server_busy")
            await busy.aclose()
            await queue.put(
                {
                    "type": "input_audio_buffer.append",
                    "audio": "AQACAA==",
                    "event_id": "frame",
                }
            )
            # Output must arrive while the input stream remains open and idle.
            audio = await asyncio.wait_for(anext(events), 2)
            self.assertEqual(audio["delta"], "AQACAA==")
            self.assertNotIn("sglang", audio)
            self.assertEqual(received[-1]["sglang"], {"seq": 0, "t_start_ms": 0})
            self.assertEqual(received[-1]["event_id"], "frame")
            await queue.put({"type": "input_audio_buffer.commit"})
            done = await asyncio.wait_for(anext(events), 2)
            self.assertEqual(done["response"]["metadata"]["input_acks"], "1")
            self.assertIsNone(done["response"]["status_details"])
            self.assertEqual(
                done["response"]["metadata"]["native_finish_reason"], "stop"
            )
            self.assertEqual(done["response"]["metadata"]["dynamo_worker_id"], "42")
            committed = await asyncio.wait_for(anext(events), 2)
            self.assertEqual(committed["type"], "input_audio_buffer.committed")
            await queue.put(None)
            waiting = asyncio.create_task(anext(events, None))
            await asyncio.sleep(0.1)
            self.assertFalse(
                waiting.done(), "input EOF must retain pending model output"
            )
            context.stopped = True
            self.assertIsNone(await asyncio.wait_for(waiting, 2))
            await asyncio.wait_for(closed.wait(), 2)
            self.assertFalse(worker.active)

    async def test_failed_native_connection_releases_worker(self):
        app = web.Application()
        async with TestServer(app) as server, ClientSession() as session:
            worker = VoiceChatWorker(session, str(server.make_url("")).rstrip("/"), 42)
            events = worker.generate(stream(asyncio.Queue()), Context())
            with self.assertLogs("voicechat.worker", level="ERROR"):
                event = await anext(events)
            self.assertEqual(event["error"]["code"], "backend_error")
            await events.aclose()
            self.assertFalse(worker.active)


if __name__ == "__main__":
    unittest.main()
