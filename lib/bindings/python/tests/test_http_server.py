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
import base64
import contextlib
import json
import re
import time
from pathlib import Path
from typing import AsyncGenerator, Dict

import aiohttp
import numpy as np
import pytest

from dynamo.common.multimodal import video_loader as video_loader_module
from dynamo.common.multimodal.media_source import decode_data_uri
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.frontend.utils import PromptLogprobsAccumulator
from dynamo.llm import HttpAsyncEngine, HttpError, HttpService
from dynamo.runtime import DistributedRuntime

# Real VP9 clip. The probe names vp9, so this is a video, not a 415.
_VALID_CLIP = Path(__file__).resolve().parents[4] / "lib/llm/tests/data/media/2p_10.mp4"

MSG_CONTAINS_ERROR = "This message contains an 400error."
MSG_CONTAINS_STATUS_ERROR = "This message contains a 415 status error."
MSG_CONTAINS_INVALID_ARGUMENT = "This message contains an invalid argument."
MSG_CONTAINS_INTERNAL_ERROR = "This message contains an internal server error."


class _OpenIfCodec:
    """Software-decode stand-in: a probed codec plays, anything else fails open."""

    def load_bytes(self, content: bytes):
        if video_loader_module.probe_video_codec(content) is None:
            raise ValueError("Could not open video stream")
        frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        return frames, {"fps": 1.0, "frames_indices": [0]}


async def _decode_request_videos(content: list) -> None:
    """Run each video part through the real loader before the mock completion."""
    loader = VideoLoader()
    for part in content:
        if not isinstance(part, dict) or part.get("type") != "video_url":
            continue
        url = part["video_url"]["url"]
        await loader._decode_video_bytes(decode_data_uri(url), _OpenIfCodec(), url)


def _valid_clip_data_url() -> str:
    payload = base64.b64encode(_VALID_CLIP.read_bytes()).decode("ascii")
    return f"data:video/mp4;base64,{payload}"


def _non_video_data_url() -> str:
    payload = base64.b64encode(b"not-a-video").decode("ascii")
    return f"data:video/mp4;base64,{payload}"


def _sse_error(raw: str) -> Dict:
    """The error object from the first SSE data frame that carries one."""
    for line in raw.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload.strip() == "[DONE]":
            continue
        event = json.loads(payload)
        if "error" in event:
            return event["error"]
    raise AssertionError(f"no SSE error frame; got: {raw}")


def _video_request(model_name: str, url: str, stream: bool) -> Dict:
    body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video."},
                    {"type": "video_url", "video_url": {"url": url}},
                ],
            }
        ],
    }
    if stream:
        body["stream"] = True
    return body


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
        if isinstance(user_message, list):
            await _decode_request_videos(user_message)
            user_message = ""
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
        prompt_logprobs = PromptLogprobsAccumulator(request)
        prompt_logprobs.update(
            {
                "engine_data": {
                    "prompt_logprobs": [
                        None,
                        {"17": {"logprob": -0.25, "rank": 1}},
                    ]
                }
            }
        )
        for i, char in enumerate(response_text):
            finish_reason = "stop" if i == len(response_text) - 1 else None
            chunk = {
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
            prompt_logprobs.attach(chunk)
            yield {"_dynamo_annotated": True, "data": chunk}
            await asyncio.sleep(0.01)


@pytest.mark.forked
def test_batch_endpoint_cannot_be_changed_at_runtime():
    service = HttpService()

    with pytest.raises(
        Exception,
        match="batch endpoint availability is fixed when the HTTP service is built",
    ):
        service.enable_endpoint("batch", True)


@pytest.fixture(scope="function", autouse=False)
async def http_server(request, unused_tcp_port: int, runtime: DistributedRuntime):
    """Fixture to start a mock HTTP server using HttpService, contributed by Baseten.

    Parametrize indirectly with a bool to set ``wait_for_first_item`` on the
    service; the default is False.
    """
    wait_for_first_item = getattr(request, "param", False)
    port = unused_tcp_port
    model_name = "test_model"
    start_done = asyncio.Event()
    checksum = "abc123"  # Checksum of ModelDeplomentCard for that model
    # Create service outside worker so we can shutdown
    service = HttpService(port=port, wait_for_first_item=wait_for_first_item)

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
            except OSError:
                # open_connection raises a bare OSError, not ConnectionError,
                # when every resolved address is refused; don't narrow this.
                raise_if_server_exited()
                await asyncio.sleep(0.1)
                continue
            raise_if_server_exited()  # a stale listener may answer; confirm ours bound
            return

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


WAIT_FOR_FIRST_ITEM = pytest.param(True, id="wait_for_first_item")
DEFAULT_SERVICE = pytest.param(False, id="default")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "http_server", [DEFAULT_SERVICE, WAIT_FOR_FIRST_ITEM], indirect=True
)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_chat_completion_success(http_server):
    """A streaming completion arrives in full, including its first chunk, whether
    or not the service waits for the first item before committing the status."""
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


HTTP_ERROR_CASES = (
    (MSG_CONTAINS_ERROR, 400, "Bad Request", "Bad Request"),
    (
        MSG_CONTAINS_STATUS_ERROR,
        415,
        "Unsupported Media Type",
        "Unsupported Media Type",
    ),
    (
        MSG_CONTAINS_INVALID_ARGUMENT,
        400,
        "Invalid request",
        "Bad Request",
    ),
    (
        MSG_CONTAINS_INTERNAL_ERROR,
        500,
        "Internal server error",
        "Internal Server Error",
    ),
)


def expected_error_body(status: int, message: str, error_type: str) -> Dict:
    body = {"message": message, "type": error_type, "code": status}
    if status == 500:
        body["details"] = {"backend_status": status}
    return body


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("trigger", "status", "expected_message", "expected_type"), HTTP_ERROR_CASES
)
@pytest.mark.timeout(60)
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
            assert await response.json() == expected_error_body(
                status, expected_message, expected_type
            )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_valid_video_clip_completes_over_http(http_server):
    """A real VP9 clip is a video, so the client gets a completion, not 415."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = _video_request(model_name, _valid_clip_data_url(), stream=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            body = await response.text()
            assert response.status == 200, body
            assert "Unsupported video media" not in body


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_valid_video_clip_streams_without_an_error_event(http_server):
    """The same clip on a stream returns content and no SSE error event."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = _video_request(model_name, _valid_clip_data_url(), stream=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == 200
            content = ""
            raw = ""
            async for line in response.content:
                raw += line.decode()
                if not line.startswith(b"data: "):
                    continue
                chunk_data = line[len(b"data: ") :]
                if chunk_data.strip() == b"[DONE]":
                    break
                chunk = json.loads(chunk_data)
                delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                content += delta.get("content") or ""
    assert content == "This is a mock response."
    assert '"error"' not in raw
    assert "Unsupported video media" not in raw


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_non_video_data_url_is_unsupported_over_http(http_server):
    """Bytes with no codec and no container are a client error, not a completion."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = _video_request(model_name, _non_video_data_url(), stream=False)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            body = await response.text()
            assert response.status == 415, body
            assert json.loads(body) == {
                "message": "Unsupported video media",
                "type": "Unsupported Media Type",
                "code": 415,
            }


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_non_video_data_url_is_unsupported_over_sse(http_server):
    """The same bytes on a stream arrive as an SSE error, not a completion."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = _video_request(model_name, _non_video_data_url(), stream=True)
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == 200
            assert response.content_type == "text/event-stream"
            raw = await response.text()

    assert _sse_error(raw) == {
        "message": "Unsupported video media",
        "type": "invalid_request_error",
        "code": 415,
    }
    assert "This is a mock response." not in raw


def test_video_public_message_matches_binding_allowlist():
    """The Python producer and the Rust allowlist must agree on the video 415 text.

    A reworded sentence on either side would silently fall back to the catalog
    text, so this checks the literal in both files."""
    repo = Path(__file__).resolve().parents[4]
    sentence = "Unsupported video media"
    loader_src = (
        repo / "components/src/dynamo/common/multimodal/video_loader.py"
    ).read_text()
    errors_src = (repo / "lib/bindings/python/rust/errors.rs").read_text()

    assert f'"{sentence}"' in loader_src
    allowlist = re.search(
        r"PUBLIC_HTTP_MESSAGES: &\[&str\] = &\[(.*?)\];", errors_src, re.DOTALL
    )
    assert allowlist is not None, "PUBLIC_HTTP_MESSAGES not found in errors.rs"
    assert f'"{sentence}"' in allowlist.group(1)


@pytest.mark.asyncio
@pytest.mark.parametrize("http_server", [WAIT_FOR_FIRST_ITEM], indirect=True)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_streaming_chat_completion_http_error_waits_for_first_item(http_server):
    """With wait_for_first_item, an exception raised before the generator's first
    yield maps to the same HTTP error response for a streaming request as for a
    non-streaming one.

    The status mapping itself belongs to test_chat_completion_http_error; this
    pairs with test_streaming_chat_completion_http_error_default_commits_200 on
    the same trigger, so the only difference is the option under test."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": MSG_CONTAINS_ERROR}],
        "stream": True,
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == 400
            assert await response.json() == expected_error_body(
                400, "Bad Request", "Bad Request"
            )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_streaming_chat_completion_http_error_default_commits_200(http_server):
    """Without wait_for_first_item, a streaming request commits HTTP 200 before
    the generator runs, so an exception raised before its first yield arrives
    inside the SSE stream instead."""
    base_url, model_name = http_server
    url = f"{base_url}/v1/chat/completions"
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": MSG_CONTAINS_ERROR}],
        "stream": True,
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(url, json=data) as response:
            assert response.status == 200
            assert response.content_type == "text/event-stream"
            body = await response.text()

    # The frame carries the sanitized stream error, not the backend's own
    # message: HTTP 200 is already committed, so there is no status left to
    # carry the 400, and the stream formatter does not forward backend text.
    assert (
        '"error"' in body
    ), f"the pre-yield failure must arrive as an SSE error frame; got: {body}"
    assert "mock response" not in body, (
        f"the generator raised before its first yield, so the stream must "
        f"carry no content; got: {body}"
    )
