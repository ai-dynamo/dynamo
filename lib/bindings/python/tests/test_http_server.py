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
import contextlib
import json
import time
from typing import AsyncGenerator, Dict

import aiohttp
import pytest

from dynamo.llm import HttpAsyncEngine, HttpError, HttpService
from dynamo.runtime import DistributedRuntime

MSG_CONTAINS_ERROR = "This message contains an 400error."
MSG_CONTAINS_STATUS_ERROR = "This message contains a 415 status error."
MSG_CONTAINS_INVALID_ARGUMENT = "This message contains an invalid argument."
MSG_CONTAINS_INTERNAL_ERROR = "This message contains an internal server error."


class _StatusLikeError(Exception):
    """Mimics dynamo.common.http.HttpStatusError's .status + .message shape."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.integration,
]


class MockHttpEngine:
    """A mock engine that returns a completion or raises an error."""

    def __init__(self, model_name: str = "test_model"):
        self.model_name = model_name

    async def generate(self, request: Dict, context) -> AsyncGenerator[Dict, None]:
        """
        Raises the requested exception, otherwise streams a mock response.
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
        elif MSG_CONTAINS_STATUS_ERROR.lower() in user_message.lower():
            raise _StatusLikeError(status=415, message=MSG_CONTAINS_STATUS_ERROR)
        elif MSG_CONTAINS_INVALID_ARGUMENT.lower() in user_message.lower():
            raise ValueError(MSG_CONTAINS_INVALID_ARGUMENT)
        elif MSG_CONTAINS_INTERNAL_ERROR.lower() in user_message.lower():
            raise RuntimeError("Simulated internal error")

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
    checksum = "abc123"  # Checksum of ModelDeplomentCard for that model
    service = HttpService(port=port)  # Create service outside worker so we can shutdown

    async def worker():
        """The server worker task."""
        try:
            loop = asyncio.get_running_loop()
            python_engine = MockHttpEngine(model_name)
            engine = HttpAsyncEngine(python_engine.generate, loop)

            service.add_chat_completions_model(model_name, checksum, engine)
            service.enable_endpoint("chat", True)

            shutdown_signal = service.run(runtime)
            print("Starting service on port", port)
            start_done.set()
            await shutdown_signal
        except Exception as e:
            print("Server encountered an error:", e)
            start_done.set()
            raise ValueError(f"Server failed to start: {e}")

    server_task = asyncio.create_task(worker())

    def raise_if_server_exited():
        # A finished startup task means the server never came up.
        if server_task.done():
            raise ValueError(
                "HTTP server exited during startup"
            ) from server_task.exception()

    async def wait_until_accepting():
        # start_done fires when service.run() returns, but the socket is bound
        # later on the runtime's background threads; wait until it accepts.
        while True:
            try:
                _, writer = await asyncio.open_connection("localhost", port)
                writer.close()
                return
            except ConnectionError:
                raise_if_server_exited()
                await asyncio.sleep(0.1)

    async def stop_server():
        service.shutdown()
        server_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await asyncio.wait_for(server_task, timeout=10.0)

    try:
        await asyncio.wait_for(start_done.wait(), timeout=30.0)
        raise_if_server_exited()
        await asyncio.wait_for(wait_until_accepting(), timeout=10.0)
    except BaseException:
        await stop_server()  # teardown past the yield won't run on setup failure
        raise

    yield f"http://localhost:{port}", model_name

    await stop_server()


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
    ("trigger", "status", "expected_message", "expected_type"),
    [
        (MSG_CONTAINS_ERROR, 400, MSG_CONTAINS_ERROR, "Bad Request"),
        (
            MSG_CONTAINS_STATUS_ERROR,
            415,
            MSG_CONTAINS_STATUS_ERROR,
            "Unsupported Media Type",
        ),
        (
            MSG_CONTAINS_INVALID_ARGUMENT,
            400,
            f"ValueError: {MSG_CONTAINS_INVALID_ARGUMENT}",
            "Bad Request",
        ),
        (
            MSG_CONTAINS_INTERNAL_ERROR,
            500,
            "Internal server error",
            "Internal Server Error",
        ),
    ],
)
@pytest.mark.forked
async def test_chat_completion_http_error(
    http_server,
    trigger: str,
    status: int,
    expected_message: str,
    expected_type: str,
):
    """Tests that backend exceptions map to the expected HTTP responses."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": trigger}],
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == status
            error_json = await response.json()
            assert error_json == {
                "message": expected_message,
                "type": expected_type,
                "code": status,
            }
