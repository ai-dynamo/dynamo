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

"""CPU tests run inside the pinned SGLang-Omni image, without loading weights."""

import asyncio
import importlib.util
import unittest
from unittest.mock import AsyncMock

if importlib.util.find_spec("sglang_omni") is None:
    raise unittest.SkipTest("Requires the SGLang-Omni image")

from demo_tools import generate_random_number, registry
from native_tools import ToolOutput
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    NemotronVoiceChatToolPipelineConfig,
)
from sglang_omni.pipeline.session_graph import SessionGraphRunner
from sglang_omni.pipeline.sessions import SessionGraphRuntime
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import OutputChunk, SessionIdentity, SessionLimits
from sglang_omni.serve.realtime.adapters import CoordinatorAdapter
from sglang_omni.serve.realtime.output import ToolCallEvent, ToolResultEvent
from sglang_omni.serve.realtime.output_buffer import OutputBuffer
from sglang_omni.serve.realtime.projection import project_output
from sglang_omni.serve.realtime.types import RuntimeLimits, Unit


class NativeToolsTests(unittest.IsolatedAsyncioTestCase):
    async def test_late_tool_result_survives_unit_change_and_idle_gap(self):
        queue, emitted = asyncio.Queue(), []
        arrived = asyncio.Event()
        identity = SessionIdentity("native")

        class Client:
            async def session_outputs(self, session_id):
                while (chunk := await queue.get()) is not None:
                    yield chunk

        async def sink(event, unit=None):
            emitted.append(event)
            arrived.set()

        adapter = CoordinatorAdapter(
            Client(),
            request_builder=lambda config: OmniRequest(inputs=None),
            output_converter=ToolOutput(),
            atomic_consumption=True,
        )
        adapter.session_identity, adapter.output_sink = identity, sink
        adapter.active_unit = Unit(3, 3840, b"", 1280)
        adapter.unit_completion = asyncio.get_running_loop().create_future()

        def chunk(seq, fmt=None, payload=None, kind="data"):
            return OutputChunk(
                identity,
                0,
                seq,
                "audio" if fmt == "voicechat" else "tool",
                0,
                0,
                payload,
                format=fmt,
                kind=kind,
            )

        task = asyncio.create_task(adapter.read_session_outputs())
        try:
            await queue.put(
                chunk(3, "voicechat", {"pcm": b"\0\0", "text": "", "eos": False})
            )
            await queue.put(
                chunk(
                    3,
                    "tool_call",
                    {
                        "calls": [
                            {
                                "id": "call1",
                                "name": "generate_random_number",
                                "arguments": {"min": 1, "max": 50},
                            }
                        ]
                    },
                )
            )
            await queue.put(chunk(3, kind="input_done"))
            await asyncio.wait_for(adapter.unit_completion, 2)
            self.assertEqual(type(emitted[0]).__name__, "ResponseStarted")
            self.assertIsInstance(emitted[-1], ToolCallEvent)
            # The tool retains seq=3 while audio has advanced to seq=4.
            adapter.active_unit = Unit(4, 5120, b"", 1280)
            adapter.unit_completion = asyncio.get_running_loop().create_future()
            await queue.put(
                chunk(
                    3,
                    "tool_response",
                    {"responses": [{"id": "call1", "response": {"result": 37}}]},
                )
            )
            await queue.put(chunk(4, kind="input_done"))
            await asyncio.wait_for(adapter.unit_completion, 2)
            self.assertIsInstance(emitted[-1], ToolResultEvent)
            self.assertEqual(emitted[-1].call_id, "native-call1")
            self.assertEqual(project_output(emitted[-1])["output"], '{"result": 37}')
            # A result between input units is also published without waiting for audio.
            adapter.active_unit = None
            arrived.clear()
            await queue.put(
                chunk(
                    3,
                    "tool_response",
                    {"responses": [{"id": "call2", "response": {"error": "failed"}}]},
                )
            )
            await asyncio.wait_for(arrived.wait(), 2)
            self.assertEqual(emitted[-1].call_id, "native-call2")
        finally:
            adapter.is_closing = True
            await queue.put(None)
            await asyncio.wait_for(task, 2)

    async def test_native_registry_and_real_function_validation(self):
        config = NemotronVoiceChatToolPipelineConfig(
            model_path="unused", tools=registry()
        )
        self.assertEqual(config.session_graph().order(), ("voicechat", "tool"))
        self.assertIn(
            "generate_random_number", config.stage_factory_kwargs("tool")["tools"]
        )
        self.assertEqual(generate_random_number({"min": 37, "max": 37}), {"result": 37})
        for args in (
            {"min": True, "max": 4},
            {"min": 5, "max": 1},
            {"min": 0, "max": 1000001},
        ):
            with self.assertRaises(ValueError):
                generate_random_number(args)

    async def test_published_tool_result_also_reaches_model_control_edge(self):
        config = NemotronVoiceChatToolPipelineConfig(
            model_path="unused",
            tools=registry(),
            session_publish={"tool": ["tool_response"]},
        )
        native = SessionIdentity("native:tool")
        output = OutputChunk(
            native,
            0,
            0,
            "tool",
            0,
            0,
            {"responses": [{"id": "call1", "response": {"result": 37}}]},
            format="tool_response",
        )
        graph = SessionGraphRuntime(
            session_identity=SessionIdentity("native"),
            graph=config.session_graph(),
            nodes={"tool": native},
            limits=SessionLimits(),
            inboxes={},
            outputs=asyncio.Queue(),
        )
        runner = SessionGraphRunner()

        async def outputs(identity):
            yield output

        runner.session_outputs = outputs
        runner.control_edge = AsyncMock()
        await runner.route_graph_outputs(graph, "tool")
        runner.control_edge.assert_awaited_once()
        self.assertEqual(runner.control_edge.call_args.args[1], "voicechat")
        self.assertFalse(
            graph.outputs.empty(),
            "Published tool result was suppressed by its control edge",
        )
        published, _ = graph.outputs.get_nowait()
        self.assertEqual(published.payload, output.payload)
        self.assertEqual(published.session_identity, graph.session_identity)

    async def test_tool_payload_bytes_count_toward_output_limit(self):
        output = OutputBuffer(RuntimeLimits(max_output_bytes=512))
        with self.assertRaisesRegex(RuntimeError, "outbound event budget"):
            output.emit(ToolResultEvent("call", "x" * 513), None, ("audio",))


if __name__ == "__main__":
    unittest.main()
