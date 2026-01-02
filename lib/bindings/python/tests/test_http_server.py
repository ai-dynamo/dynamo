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

# This test verifies that the HTTP server can be started and responds correctly to requests.

import asyncio
import json
import time
from typing import AsyncGenerator, Dict

import aiohttp
import pytest

from dynamo.llm import HttpAsyncEngine, HttpError, HttpService
from dynamo.runtime import DistributedRuntime

MSG_CONTAINS_ERROR = "This message contains an 400error."
MSG_CONTAINS_INTERNAL_ERROR = "This message contains an internal server error."

pytestmark = pytest.mark.pre_merge


class MockHttpEngine:
    """A mock engine that returns a completion or raises an error."""

    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name

    async def generate(self, request: Dict, context) -> AsyncGenerator[Dict, None]:
        """
        Raises HttpError if message contains 'error', otherwise streams a mock response.
        """
        user_message = ""
        for message in request.get("messages", []):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        # verifies that cancellation is propagated
        if context.is_stopped():
            print(f"Request {context.id()} was cancelled before starting.")
            return

        if MSG_CONTAINS_ERROR.lower() in user_message.lower():
            raise HttpError(code=400, message=MSG_CONTAINS_ERROR)
        elif MSG_CONTAINS_INTERNAL_ERROR.lower() in user_message.lower():
            raise ValueError("Simulated internal error")

        # Stream a mock response
        created = int(time.time())
        response_text = "This is a mock response."
        for i, char in enumerate(response_text):
            finish_reason = "stop" if i == len(response_text) - 1 else None
            yield {
                "id": f"chatcmpl-{context.id()}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": self.model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": char},
                        "finish_reason": finish_reason,
                    }
                ],
            }
            await asyncio.sleep(0.01)


@pytest.fixture(scope="function", autouse=False)
async def http_server(runtime: DistributedRuntime):
    """Fixture to start a mock HTTP server using HttpService, contributed by Baseten."""
    port = 8008
    model_name = "test_model"
    start_done = asyncio.Event()
    child_token = runtime.child_token()
    checksum = "abc123"  # Checksum of ModelDeplomentCard for that model

    async def worker():
        """The server worker task."""
        try:
            loop = asyncio.get_running_loop()
            python_engine = MockHttpEngine(model_name)
            engine = HttpAsyncEngine(python_engine.generate, loop)

            service = HttpService(port=port)
            service.add_chat_completions_model(model_name, checksum, engine)
            service.enable_endpoint("chat", True)

            shutdown_signal = service.run(child_token)
            print("Starting service on port", port)
            start_done.set()
            await shutdown_signal
        except Exception as e:
            print("Server encountered an error:", e)
            start_done.set()
            raise ValueError(f"Server failed to start: {e}")
        finally:
            child_token.cancel()

    server_task = asyncio.create_task(worker())
    await asyncio.wait_for(start_done.wait(), timeout=30.0)
    if server_task.done() and server_task.exception():
        raise ValueError(f"Server task failed to start {server_task.exception()}")
    yield f"http://localhost:{port}", model_name

    # Teardown: Cancel the server task if it's still running
    child_token.cancel()
    await asyncio.sleep(0.1)  # Give some time for graceful shutdown
    if not server_task.done():
        server_task.cancel()
        try:
            # Await cancellation to ensure proper cleanup for up to 10s
            await asyncio.wait_for(server_task, timeout=10.0)
        except asyncio.CancelledError:
            print("Server task cancelled during teardown.")
            pass


@pytest.mark.asyncio
@pytest.mark.forked
async def test_chat_completion_success(http_server):
    """Tests a successful chat completion request."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": "Hello, this is a test."}],
        "stream": True,
    }
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
        async with session.post(url, json=data) as response:
            response.raise_for_status()

            content = ""
            async for line in response.content:
                if line.startswith(b"data: "):
                    chunk_data = line[len(b"data: ") :]
                    if chunk_data.strip() == b"[DONE]":
                        break
                    chunk = json.loads(chunk_data)
                    if (
                        chunk["choices"]
                        and chunk["choices"][0]["delta"]
                        and chunk["choices"][0]["delta"].get("content")
                    ):
                        content += chunk["choices"][0]["delta"]["content"]

            assert content == "This is a mock response."


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "msg_to_code",
    [
        (MSG_CONTAINS_ERROR, 500),  # # TODO: should be 400, but currently 500
        (
            MSG_CONTAINS_INTERNAL_ERROR,
            500,
        ),  # Placeholder for future internal error test
    ],
)
@pytest.mark.forked
async def test_chat_completion_http_error(http_server, msg_to_code: tuple[str, int]):
    """Tests that an HttpError is raised when the message contains 'error'."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": msg_to_code[0]}],
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == msg_to_code[1]
            error_json = await response.json()
            if msg_to_code[0] == MSG_CONTAINS_ERROR:
                assert MSG_CONTAINS_ERROR in str(error_json)
            elif msg_to_code[0] == MSG_CONTAINS_INTERNAL_ERROR:
                assert "a python exception was caught" in str(error_json).lower()
