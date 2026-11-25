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

import ast
import asyncio
import base64
import contextlib
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Dict
from unittest.mock import AsyncMock

import aiohttp
import numpy as np
import pytest
import pytest_asyncio
from aiohttp import web

from dynamo._core import (
    Context,
    DynamoException,
    InvalidArgument,
    WorkerDraining,
    _raise_dispatch_error,
    _raise_routed_http_error,
)
from dynamo.common.http import HttpConnectionError, HttpStatusError, HttpTimeoutError
from dynamo.frontend.utils import PromptLogprobsAccumulator
from dynamo.llm import HttpAsyncEngine, HttpError, HttpService
from dynamo.runtime import DistributedRuntime

try:
    from dynamo.common.multimodal.audio_loader import AudioLoader
    from dynamo.common.multimodal.image_loader import ImageLoader
    from dynamo.common.multimodal.video_loader import VideoLoader
except ImportError:
    # CPU text-only test images do not install all multimodal dependencies.
    AudioLoader = None
    ImageLoader = None
    VideoLoader = None

# Real VP9 clip. The probe names vp9, so this is a video, not a 415.
_VALID_CLIP = Path(__file__).resolve().parents[4] / "lib/llm/tests/data/media/2p_10.mp4"

MSG_CONTAINS_ERROR = "This message contains an 400error."
MSG_CONTAINS_STATUS_ERROR = "This message contains a 415 status error."
MSG_CONTAINS_INVALID_ARGUMENT = "This message contains an invalid argument."
MSG_CONTAINS_INTERNAL_ERROR = "This message contains an internal server error."
MSG_CONTAINS_ROUTED_ERROR = "This message contains a routed worker's error."
MSG_CONTAINS_LONG_ERROR = "This message contains a long HTTP error."
LONG_ERROR_MESSAGE = "Engine validation failure: " + "x" * 8192
MSG_SETS_RESPONSE_HEADER = "This message sets a response header."
AUTHORED_VALIDATION_MESSAGE = "Unknown template 'missing'. Choose a different template."
NONMEDIA_422_MESSAGE = "Output budget is invalid (parameter=max_tokens, value=7)"
CONTEXT_LENGTH_MESSAGE = (
    "This model's maximum context length is 8 tokens. However, your messages "
    "resulted in 10 tokens. Please reduce the length of the messages."
)
CONTEXT_LENGTH_DETAILS = {
    "type": "context_length",
    "limit": 8,
    "actual": 10,
    "prompt_tokens": 10,
}


def _retained_test_error(payload: dict, routed: bool) -> Exception:
    """Reuse the real relay converter with a captured or legacy worker payload."""
    try:
        _raise_routed_http_error(500, "PRIVATE_RELAY_DISPLAY", routed_engine=routed)
    except ValueError as error:
        attribute = "_dynamo_error" if routed else "_dynamo_relay_content"
        setattr(error, attribute, json.dumps(payload))
        return error
    raise AssertionError("the relay hook must raise")


def _raise_binding_regression_error(case: str) -> None:
    """Exercise Python-dependent conversions through client-visible responses."""
    parts = case.split(":")
    kind = parts[0]
    if kind == "unknown_runtime":
        raise RuntimeError("engine rejected the request")
    if kind == "programming":
        _, name, mode, translation = parts
        error_type = {
            "KeyError": KeyError,
            "TypeError": TypeError,
            "AttributeError": AttributeError,
        }[name]
        if mode == "outer":
            try:
                raise InvalidArgument(AUTHORED_VALIDATION_MESSAGE)
            except InvalidArgument:
                raise error_type("PRIVATE_OUTER_BUG") from None
        try:
            raise error_type(
                "PRIVATE_PROGRAMMING" * (1024 if translation == "long_copy" else 1)
            )
        except error_type as error:
            replacement = InvalidArgument(
                AUTHORED_VALIDATION_MESSAGE if translation == "authored" else str(error)
            )
            if mode == "explicit":
                raise replacement from error
            if mode == "suppressed":
                raise replacement from None
            raise replacement

    if kind == "cause":
        mode = parts[1]
        if mode in ("visible_programming", "suppressed_programming"):
            try:
                raise TypeError("PRIVATE_PROGRAMMING")
            except TypeError:
                if mode == "suppressed_programming":
                    raise ValueError("received replacement") from None
                raise ValueError("received replacement")
        if mode == "forked":
            error = InvalidArgument(AUTHORED_VALIDATION_MESSAGE)
            error.__cause__ = ValueError("visible cause")
            error.__context__ = HttpConnectionError("PRIVATE_HIDDEN_CONTEXT")
            error.__suppress_context__ = True
            raise error
        if mode.startswith("hidden_"):
            error_type = HttpConnectionError if "media" in mode else ConnectionError
            try:
                raise error_type("PRIVATE_HIDDEN_SOURCE")
            except error_type as error:
                replacement = InvalidArgument if mode.endswith("_typed") else ValueError
                raise replacement(str(error)) from None
        error = ValueError("PRIVATE_CYCLE_OR_DEPTH")
        if mode == "cycle":
            error.__cause__ = error
        elif mode == "hidden_cycle":
            error.__context__ = error
            error.__suppress_context__ = True
        elif mode == "excessive":
            for _ in range(20):
                wrapper = ValueError("PRIVATE_OUTER_CHAIN")
                wrapper.__cause__ = error
                error = wrapper
        else:
            raise AssertionError(f"unknown cause case: {case}")
        raise error

    if kind == "ancestry":
        if parts[1].startswith("requests"):
            from requests.exceptions import RequestException

            parent = RequestException
        else:
            parent = HttpConnectionError

        class CustomFetchError(parent):
            pass

        try:
            raise CustomFetchError("PRIVATE_SUBCLASS_FETCH")
        except CustomFetchError as error:
            if parts[1].endswith("wrapped"):
                raise ValueError(str(error)) from error
            raise

    if kind == "media_type":
        _, engine_type, parameter, value_kind = parts
        parent = type(
            "VLLMUnprocessableEntityError",
            (Exception,),
            {"__module__": "vllm.exceptions"},
        )
        error_type = {
            "typed": parent,
            "subclass": type("CustomVLLMError", (parent,), {}),
            "future": Exception,
        }[engine_type]
        error = error_type("received engine explanation")
        error.message = "received engine explanation"
        error.status_code = 422
        error.parameter = parameter
        error.value = {
            "https": "https://PRIVATE_MEDIA_HOST/image",
            "http": "HTTP://PRIVATE_MEDIA_HOST/video",
            "data": "data:audio/wav;base64,PRIVATE_MEDIA",
            "relative": "../PRIVATE_MEDIA_FILE.png",
            "file": "file:///PRIVATE_MEDIA_FILE.png",
            "opaque": "not a URL",
            "number": 7,
        }[value_kind]
        raise error

    if kind == "numeric":
        if parts[1] == "native":
            error_type = type("NativeError", (DynamoException,), {})
            error = error_type("native enum explanation")
            error.code = int(parts[2])
            error.message = "native enum explanation"
        else:
            error = Exception("HTTP API rejected the request")
            error.message = "HTTP API rejected the request"
            if parts[2] != "none":
                error.code = int(parts[2])
            error.status_code = int(parts[3])
        raise error

    if kind == "http_status":
        raise _StatusLikeError(int(parts[1]), "received HTTP explanation")

    if kind == "relay":
        _, route, mode = parts
        routed = route == "full"
        payload = {
            "class": "Internal",
            "reason": "runtime.internal",
            "message": "PRIVATE_RELAY_DISPLAY",
        }
        if mode.startswith("legacy_"):
            replacement = mode.removeprefix("legacy_").removeprefix("authored_")
            if replacement == "public":
                payload["public"] = {
                    "type": "message",
                    "message": "legacy public replacement",
                }
            elif replacement == "context":
                payload["public"] = CONTEXT_LENGTH_DETAILS
        else:
            payload["upstream_error"] = {
                "message": "received worker explanation",
                "http_status": 503,
            }
            if mode.startswith("media_"):
                payload["upstream_error"]["family"] = "media_fetch"
            elif mode == "context":
                payload["class"] = "InvalidRequest"
                payload["reason"] = "request.invalid"
                payload["public"] = CONTEXT_LENGTH_DETAILS
            elif mode.startswith("transport"):
                payload["class"] = "Unavailable"
                payload["reason"] = "transport.cannot_connect"
                if mode == "transport_explicit":
                    payload["upstream_error"]["family"] = "transport"
                elif mode == "transport_no_http":
                    payload["upstream_error"].pop("http_status")
            elif mode in ("worker", "invalid_worker"):
                worker = {
                    "class": "Unavailable",
                    "error_type": "WorkerUnavailable",
                    "reason": "backend.unavailable",
                    "retry_hint": "draining",
                    "upstream_error": {"message": "worker closed admission"},
                }
                payload["caused_by"] = worker
                payload.pop("upstream_error")
                if mode == "invalid_worker":
                    payload["reason"] = "request.invalid"
            elif mode.startswith("dispatch_public"):
                validation = mode.endswith("request")
                payload["class"] = "InvalidRequest" if validation else "Unavailable"
                payload["reason"] = (
                    "request.invalid" if validation else "backend.unavailable"
                )
                payload["public"] = {
                    "type": "message",
                    "message": "Authored dispatch replacement",
                }

        if mode == "worker" and routed:
            # This is the typed outer envelope emitted by routed_error_to_pyerr.
            payload = {
                "class": "Unavailable",
                "error_type": "WorkerUnavailable",
                "reason": "backend.worker_unavailable",
                "retry_hint": "draining",
                "message": "PRIVATE_OUTER_CONTEXT",
                "caused_by": payload,
            }
        error = _retained_test_error(payload, routed)
        if mode in ("worker", "invalid_worker") and routed:
            replacement = WorkerDraining("PRIVATE_OUTER_CONTEXT")
            replacement._dynamo_error = error._dynamo_error
            raise replacement
        if mode in (
            "context",
            "transport_http",
            "transport_explicit",
            "transport_no_http",
            "worker",
            "invalid_worker",
        ) or mode.startswith("dispatch_public"):
            if mode.startswith("dispatch_public"):
                replacement = Exception("PRIVATE_DISPATCH_WRAPPER")
                replacement._dynamo_relay_content = error._dynamo_relay_content
                raise replacement
            raise error
        try:
            raise error
        except ValueError as error:
            text = AUTHORED_VALIDATION_MESSAGE if "authored" in mode else str(error)
            if mode == "media_none":
                raise InvalidArgument(text) from None
            raise InvalidArgument(text) from error

    raise AssertionError(f"unknown binding regression case: {case}")


async def _raise_batch_media_fetch_error(media_kind: str, kind: str) -> None:
    """Exercise real batch aggregation with raw per-item coroutine failures.

    Per-item normalization differs by loader; the stand-ins target their
    aggregate fallback and both mixed orders without external networking.
    """
    loader_type = {"image": ImageLoader, "audio": AudioLoader, "video": VideoLoader}[
        media_kind
    ]
    loader = loader_type()
    detail = "PRIVATE_BATCH private-batch-host.internal errno=111 at 10.20.30.40"
    if kind in ("mixed_fetch_first", "mixed_decode_first"):
        failures = [
            HttpConnectionError(detail),
            RuntimeError("decoder unavailable"),
        ]
        if kind == "mixed_decode_first":
            failures.reverse()
    else:
        error_type = {
            "connection": HttpConnectionError,
            "timeout": HttpTimeoutError,
        }[kind]
        failures = [error_type(detail), error_type(detail)]
    setattr(loader, f"load_{media_kind}", AsyncMock(side_effect=failures))
    await getattr(loader, f"load_{media_kind}_batch")(
        [
            {"Url": "https://example.com/one.png"},
            {"Url": "https://example.com/two.png"},
        ]
    )


def _raise_context_or_media_error(case: str) -> None:
    """Produce real Python exception chains without a model or media decoder."""
    if ":" in case or case == "unknown_runtime":
        _raise_binding_regression_error(case)
    if case in ("authored_retained_422", "authored_retained_503"):
        status = 422 if case.endswith("422") else 503
        try:
            _raise_routed_http_error(status, "PRIVATE_RETAINED_SOURCE_STATUS")
        except (ValueError, InvalidArgument) as error:
            # Received HTTP metadata must not override the new local verdict.
            raise InvalidArgument(AUTHORED_VALIDATION_MESSAGE) from error

    if case == "authored_hidden_routed_status":
        try:
            _raise_routed_http_error(503, "PRIVATE_UNRELATED_RELAY")
        except ValueError:
            raise InvalidArgument(AUTHORED_VALIDATION_MESSAGE) from ValueError(
                "translation helper"
            )

    if case.startswith("authored_"):
        try:
            raise KeyError("missing")
        except KeyError as error:
            # The authored message includes the KeyError's short quoted text;
            # that substring does not make it a copied programming diagnostic.
            if case == "authored_explicit":
                raise InvalidArgument(AUTHORED_VALIDATION_MESSAGE) from error
            if case == "authored_from_none":
                raise InvalidArgument(AUTHORED_VALIDATION_MESSAGE) from None
            raise InvalidArgument(AUTHORED_VALIDATION_MESSAGE)

    if case.startswith("aiohttp_"):
        from aiohttp.client_reqrep import ConnectionKey

        key = ConnectionKey(
            host="PRIVATE_AIOHTTP_HOST",
            port=443,
            is_ssl=True,
            ssl=True,
            proxy=None,
            proxy_auth=None,
            proxy_headers_hash=None,
        )
        try:
            raise aiohttp.ClientConnectorError(key, OSError("PRIVATE_CONNECTOR"))
        except aiohttp.ClientConnectorError as error:
            if case == "aiohttp_wrapped":
                raise ValueError(str(error)) from error
            raise

    if case == "copied_media_from_none":
        try:
            raise HttpStatusError(
                403,
                "PRIVATE_MEDIA_FAILURE",
                "https://private-origin.example/image?token=PRIVATE_QUERY",
            )
        except HttpStatusError as error:
            raise InvalidArgument(str(error)) from None

    if case.startswith("retained_"):
        try:
            # A non-400 routed exception displays its private diagnostic. A
            # 400 would already display "Invalid request", hiding the bug.
            _raise_routed_http_error(500, "PRIVATE_RELAY_DIAGNOSTIC")
        except ValueError as error:
            payload = json.loads(error._dynamo_error)
            payload.pop("public", None)
            if case == "retained_legacy_from_none":
                # An older producer has only a private diagnostic, never a
                # received message eligible for default forwarding.
                payload.pop("upstream_error", None)
            else:
                # Model a received media error retained by a routed stream.
                payload["upstream_error"]["family"] = "media_fetch"
            error._dynamo_error = json.dumps(payload)
            raise InvalidArgument(str(error)) from None

    if case.startswith("vllm_"):
        from vllm.exceptions import VLLMUnprocessableEntityError

        from dynamo.vllm.errors import vllm_client_error_to_http_error

        media = case != "vllm_nonmedia_422"
        try:
            raise VLLMUnprocessableEntityError(
                "Unable to fetch media" if media else "Output budget is invalid",
                parameter="image_url" if media else "max_tokens",
                value=(
                    "https://private-origin.example/image?token=PRIVATE_VLLM_QUERY"
                    if media
                    else 7
                ),
            )
        except VLLMUnprocessableEntityError as error:
            # This is the production embedding handler's 422 conversion. The
            # typed wrapper itself carries media provenance without aiohttp.
            if case == "vllm_media_422_from_none":
                raise vllm_client_error_to_http_error(error) from None
            raise vllm_client_error_to_http_error(error) from error

    raise AssertionError(f"unknown context/media regression case: {case}")


class _OpenIfCodec:
    """Software-decode stand-in: a probed codec plays, anything else fails open."""

    def load_bytes(self, content: bytes):
        # Imported here: the multimodal package imports torch, which the
        # Triton CPU image does not install.
        from dynamo.common.multimodal import video_loader as video_loader_module

        if video_loader_module.probe_video_codec(content) is None:
            raise ValueError("Could not open video stream")
        frames = np.zeros((1, 2, 2, 3), dtype=np.uint8)
        return frames, {"fps": 1.0, "frames_indices": [0]}


async def _decode_request_media(content: list) -> None:
    """Load media through the real loaders before the mock completion."""
    from dynamo.common.multimodal.media_source import decode_data_uri
    from dynamo.common.multimodal.video_loader import VideoLoader

    loader = VideoLoader()
    for part in content:
        if isinstance(part, dict) and part.get("type") == "image_url":
            await ImageLoader().load_image(part["image_url"]["url"])
            continue
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

    def generate(self, request: Dict, context) -> AsyncGenerator[Dict, None]:
        # Fail at callable startup so the raw HttpAsyncEngine adapter runs.
        for message in request.get("messages", []):
            content = message.get("content", "")
            if isinstance(content, str) and content.startswith("http-dispatch-test:"):
                status = int(content.split(":", 1)[1])
                error = HttpStatusError(
                    status,
                    "PRIVATE_HTTP_SOURCE\r\nprivate continuation",
                    "https://private-origin.example/image",
                )
                error.code = status
                error.status_code = 503
                raise error
        return self._generate(request, context)

    async def _generate(self, request: Dict, context) -> AsyncGenerator[Dict, None]:
        """
        Raises the requested exception, otherwise streams a mock response.
        """
        user_message = ""
        for message in request.get("messages", []):
            if message.get("role") == "user":
                user_message = message.get("content", "")
                break
        if isinstance(user_message, list):
            await _decode_request_media(user_message)
            user_message = ""
        # verifies that cancellation is propagated
        if context.is_stopped():
            print(f"Request {context.id()} was cancelled before starting.")
            return

        if MSG_SETS_RESPONSE_HEADER.lower() in user_message.lower():
            context.set_response_header("x-mock-engine", "set")

        if user_message.startswith("public-http-test:"):
            status, message = json.loads(user_message.split(":", 1)[1])
            raise HttpStatusError(status, message, "https://example.com/image")
        if user_message.startswith("context-media-test:"):
            _raise_context_or_media_error(user_message.split(":", 1)[1])
        if user_message.startswith("batch-image-fetch-test:"):
            await _raise_batch_media_fetch_error("image", user_message.split(":", 1)[1])
        if user_message.startswith("batch-media-fetch-test:"):
            _, media_kind, kind = user_message.split(":", 2)
            await _raise_batch_media_fetch_error(media_kind, kind)
        if user_message.startswith("dispatch-error-test:"):
            case = user_message.split(":", 1)[1]
            _raise_dispatch_error(
                "TCP timeout connecting to 10.20.30.40:4321 PRIVATE_DISPATCH",
                typed=case != "untyped",
                received=case == "received",
            )
        if MSG_CONTAINS_LONG_ERROR.lower() in user_message.lower():
            raise HttpError(code=422, message=LONG_ERROR_MESSAGE)
        if MSG_CONTAINS_ERROR.lower() in user_message.lower():
            raise HttpError(code=400, message=MSG_CONTAINS_ERROR)
        elif MSG_CONTAINS_STATUS_ERROR.lower() in user_message.lower():
            raise _StatusLikeError(status=415, message=MSG_CONTAINS_STATUS_ERROR)
        elif MSG_CONTAINS_INVALID_ARGUMENT.lower() in user_message.lower():
            raise ValueError(MSG_CONTAINS_INVALID_ARGUMENT)
        elif MSG_CONTAINS_INTERNAL_ERROR.lower() in user_message.lower():
            raise RuntimeError("Simulated internal error")
        elif MSG_CONTAINS_ROUTED_ERROR.lower() in user_message.lower():
            # A worker's HttpError(503) as a routed engine raises it; escaping
            # the engine keeps the worker's status.
            _raise_routed_http_error(503, MSG_CONTAINS_ROUTED_ERROR)

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
async def http_server(
    request, unused_tcp_port: int, runtime: DistributedRuntime, monkeypatch
):
    async for server in _serve_mock_http(
        request, unused_tcp_port, runtime, monkeypatch
    ):
        yield server


@pytest_asyncio.fixture
async def pixel_limit_http_server(request, unused_tcp_port, tmp_path, monkeypatch):
    monkeypatch.setenv("DYN_FILE_KV", str(tmp_path))
    runtime = DistributedRuntime(
        asyncio.get_running_loop(), "file", "tcp", event_plane="zmq"
    )
    try:
        async for server in _serve_mock_http(
            request, unused_tcp_port, runtime, monkeypatch
        ):
            yield server
    finally:
        runtime.shutdown()


async def _serve_mock_http(
    request, unused_tcp_port: int, runtime: DistributedRuntime, monkeypatch
):
    """Fixture to start a mock HTTP server using HttpService, contributed by Baseten.

    Parametrize indirectly with ``wait_for_first_item`` or ``peek_ms`` settings.
    """
    settings = getattr(request, "param", {})
    if "peek_ms" in settings:
        monkeypatch.setenv(
            "DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS", str(settings["peek_ms"])
        )
    else:
        monkeypatch.delenv("DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS", raising=False)
    wait_for_first_item = settings.get("wait_for_first_item", False)
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


WAIT_FOR_FIRST_ITEM = pytest.param(
    {"wait_for_first_item": True}, id="wait_for_first_item"
)
DEFAULT_SERVICE = pytest.param({}, id="default")


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
        MSG_CONTAINS_INVALID_ARGUMENT,
        "Bad Request",
    ),
    (
        MSG_CONTAINS_INTERNAL_ERROR,
        500,
        "Simulated internal error",
        "Internal Server Error",
    ),
    (
        MSG_CONTAINS_ROUTED_ERROR,
        503,
        MSG_CONTAINS_ROUTED_ERROR,
        "Service Unavailable",
    ),
    pytest.param(
        MSG_CONTAINS_LONG_ERROR,
        400,
        # Main's semantic class remains InvalidRequest when the legacy JSON
        # diagnostic is truncated; received text has its own 4096-byte bound.
        LONG_ERROR_MESSAGE[:4082] + "...[truncated]",
        "Bad Request",
        id="long_message_keeps_main_class_status",
    ),
)


def expected_error_body(
    status: int,
    message: str,
    error_type: str,
    *,
    backend_status: int | None = None,
) -> Dict:
    body = {"message": message, "type": error_type, "code": status}
    if backend_status is not None:
        body["details"] = {"backend_status": backend_status}
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
                status,
                expected_message,
                expected_type,
                backend_status=500 if status == 500 else None,
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
@pytest.mark.parametrize("http_server", [{"peek_ms": 0}], indirect=True)
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
    """The Python producer's sentence retains its actual Rust public projection."""
    repo = Path(__file__).resolve().parents[4]
    loader_source = repo / "components/src/dynamo/common/multimodal/video_loader.py"
    loader = ast.parse(loader_source.read_text())
    producer = next(
        node
        for node in loader.body
        if isinstance(node, ast.FunctionDef) and node.name == "_unsupported_video_input"
    )
    returned = next(node for node in ast.walk(producer) if isinstance(node, ast.Return))
    assert isinstance(returned.value, ast.Call)
    assert ast.literal_eval(returned.value.args[0]) == 415
    sentence = ast.literal_eval(returned.value.args[1])
    assert sentence == "Unsupported video media"
    with pytest.raises(ValueError) as caught:
        _raise_routed_http_error(415, sentence)
    payload = json.loads(caught.value._dynamo_error)
    assert payload["public"] == {"type": "message", "message": sentence}


@pytest.mark.asyncio
@pytest.mark.parametrize("pixel_limit_http_server", [{"peek_ms": 0}], indirect=True)
@pytest.mark.parametrize("stream", [False, True], ids=["http", "sse"])
@pytest.mark.parametrize(
    ("case", "unary_status", "stream_status", "expected_message"),
    [
        pytest.param(*row, id=row[0].replace(":", "_"))
        for row in [
            ("authored_implicit", 400, 400, AUTHORED_VALIDATION_MESSAGE),
            ("authored_explicit", 400, 400, AUTHORED_VALIDATION_MESSAGE),
            ("authored_from_none", 400, 400, AUTHORED_VALIDATION_MESSAGE),
            ("authored_hidden_routed_status", 400, 400, AUTHORED_VALIDATION_MESSAGE),
            ("authored_retained_422", 400, 400, AUTHORED_VALIDATION_MESSAGE),
            ("authored_retained_503", 400, 400, AUTHORED_VALIDATION_MESSAGE),
            ("aiohttp_connector", 500, 500, "Internal server error"),
            ("aiohttp_wrapped", 400, 400, "Invalid request"),
            ("copied_media_from_none", 400, 400, "Invalid request"),
            ("retained_media_from_none", 400, 400, "Invalid request"),
            ("retained_legacy_from_none", 400, 400, "Invalid request"),
            ("vllm_media_422", 422, 400, "Invalid request"),
            ("vllm_media_422_from_none", 422, 400, "Invalid request"),
            ("vllm_nonmedia_422", 422, 400, NONMEDIA_422_MESSAGE),
            ("unknown_runtime", 500, 500, "engine rejected the request"),
            *[
                (
                    f"programming:{kind}:{mode}:{translation}",
                    400,
                    400,
                    AUTHORED_VALIDATION_MESSAGE
                    if translation == "authored"
                    else "Invalid request",
                )
                for kind in ("TypeError", "AttributeError", "KeyError")
                for mode in ("implicit", "explicit", "suppressed")
                for translation in ("authored", "copy")
                if not (kind == "KeyError" and translation == "authored")
            ],
            *[
                (f"programming:{kind}:outer:copy", status, status, message)
                for kind, status, message in (
                    ("TypeError", 400, "Invalid request"),
                    ("AttributeError", 500, "Internal server error"),
                    ("KeyError", 500, "Internal server error"),
                )
            ],
            ("programming:TypeError:suppressed:long_copy", 400, 400, "Invalid request"),
            ("cause:visible_programming", 400, 400, "Invalid request"),
            ("cause:suppressed_programming", 400, 400, "received replacement"),
            *[
                (f"cause:{case}", 400, 400, "Invalid request")
                for case in (
                    "hidden_media_value",
                    "hidden_media_typed",
                    "hidden_transport_value",
                    "hidden_transport_typed",
                    "forked",
                    "cycle",
                    "hidden_cycle",
                    "excessive",
                )
            ],
            *[
                (f"ancestry:{case}", status, status, message)
                for case, status, message in (
                    ("requests", 500, "Internal server error"),
                    ("requests_wrapped", 400, "Invalid request"),
                    ("dynamo", 500, "Internal server error"),
                    ("dynamo_wrapped", 400, "Invalid request"),
                )
            ],
            *[
                (
                    f"media_type:{kind}:{parameter}:{value}",
                    500,
                    500,
                    "Internal server error"
                    if masked
                    else "received engine explanation",
                )
                for kind, parameter, value, masked in (
                    ("typed", "image_url", "https", True),
                    ("subclass", "video_url", "http", True),
                    ("typed", "audio_url", "data", True),
                    ("typed", "max_tokens", "https", False),
                    ("typed", "image_url", "opaque", True),
                    ("typed", "image_url", "relative", True),
                    ("typed", "image_url", "file", True),
                    ("future", "image_url", "https", False),
                    ("typed", "image_url", "number", False),
                )
            ],
            *[
                (f"numeric:native:{code}", 500, 500, "native enum explanation")
                for code in (7, 422)
            ],
            *[
                (
                    f"numeric:sdk:{code}:{status}",
                    422 if code == "422" else 500,
                    400 if code == "422" else 500,
                    "HTTP API rejected the request",
                )
                for code, status in (("7", 422), ("7", 7), ("none", 422), ("422", 503))
            ],
            *[
                (
                    f"http_status:{status}",
                    status,
                    committed,
                    "received HTTP explanation",
                )
                for status, committed in (
                    (408, 504),
                    (418, 400),
                    (422, 400),
                    (431, 413),
                    (503, 503),
                    (529, 529),
                )
            ],
            *[
                (f"relay:{route}:legacy_{mode}", 400, 400, message)
                for route in ("full", "worker")
                for mode, message in (
                    ("none", "Invalid request"),
                    ("public", "legacy public replacement"),
                    ("context", CONTEXT_LENGTH_MESSAGE),
                    ("authored_none", AUTHORED_VALIDATION_MESSAGE),
                    ("authored_public", AUTHORED_VALIDATION_MESSAGE),
                    ("authored_context", AUTHORED_VALIDATION_MESSAGE),
                )
            ],
            *[
                (f"relay:{route}:{mode}", 400, 400, "Invalid request")
                for route in ("full", "worker")
                for mode in ("media_copy", "media_none")
            ],
            *[
                (f"relay:{route}:context", 400, 400, CONTEXT_LENGTH_MESSAGE)
                for route in ("full", "worker")
            ],
            ("relay:full:worker", 503, 503, "worker closed admission"),
            ("relay:full:invalid_worker", 500, 500, "Internal server error"),
            ("relay:worker:worker", 400, 400, "worker closed admission"),
            ("relay:worker:invalid_worker", 400, 400, "Invalid request"),
            ("relay:worker:transport_http", 400, 400, "received worker explanation"),
            ("relay:worker:transport_explicit", 400, 400, "Invalid request"),
            ("relay:worker:transport_no_http", 400, 400, "Invalid request"),
            *[
                (
                    f"relay:worker:dispatch_public_{source}",
                    500,
                    500,
                    "Authored dispatch replacement",
                )
                for source in ("request", "worker")
            ],
            ("http_dispatch:7", 500, 500, "Internal server error"),
            ("http_dispatch:422", 422, 422, "Invalid request"),
            ("http_dispatch:502", 500, 500, "Internal server error"),
            ("http_dispatch:504", 500, 500, "Internal server error"),
        ]
    ],
)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_python_error_context_and_media_privacy_over_http_and_sse(
    pixel_limit_http_server, case, unary_status, stream_status, expected_message, stream
):
    """Preserve authored validation while masking copied media and relay text."""
    if case.startswith("vllm_"):
        pytest.importorskip("vllm.exceptions")
    status = stream_status if stream else unary_status
    committed = stream and not case.startswith("http_dispatch:")
    base_url, model_name = pixel_limit_http_server
    data = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": (
                    "http-dispatch-test:" + case.split(":", 1)[1]
                    if case.startswith("http_dispatch:")
                    else f"context-media-test:{case}"
                ),
            }
        ],
        "stream": stream,
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=data
        ) as response:
            raw = await response.text()
            assert response.status == (200 if committed else status), raw
            if committed:
                assert response.content_type == "text/event-stream"
            else:
                assert response.content_type == "application/json"
    error = _sse_error(raw) if committed else json.loads(raw)
    assert error["code"] == status, raw
    assert error["message"] == expected_message, raw
    if committed:
        expected_type = {
            400: "invalid_request_error",
            413: "invalid_request_error",
            500: "internal_server_error",
            503: "service_unavailable",
            504: "timeout_error",
            529: "service_unavailable",
        }[status]
    else:
        expected_type = {
            400: "Bad Request",
            408: "Request Timeout",
            413: "Payload Too Large",
            418: "I'm a teapot",
            422: "Unprocessable Entity",
            431: "Request Header Fields Too Large",
            500: "Internal Server Error",
            503: "Service Unavailable",
            529: "Overloaded",
        }[status]
    assert error["type"] == expected_type, raw
    if not committed and case in ("http_dispatch:502", "http_dispatch:504"):
        assert error["details"] == {"backend_status": int(case.split(":")[1])}, raw
    elif not committed and (
        case in ("http_dispatch:7", "relay:full:invalid_worker")
        or case.startswith("relay:worker:dispatch_public_")
    ):
        assert "details" not in error, raw
    elif not committed and (
        case.endswith(":context") or case.endswith(":legacy_context")
    ):
        assert error["details"] == CONTEXT_LENGTH_DETAILS, raw
    elif not committed and status == 500:
        assert error["details"] == {"backend_status": 500}, raw
    else:
        assert "details" not in error, raw
    assert "PRIVATE_" not in raw
    assert "private-origin.example" not in raw
    assert "This is a mock response." not in raw


@pytest.mark.asyncio
@pytest.mark.parametrize("pixel_limit_http_server", [{"peek_ms": 0}], indirect=True)
@pytest.mark.parametrize("stream", [False, True], ids=["http", "sse"])
@pytest.mark.parametrize(
    "trigger",
    [
        *[
            pytest.param(
                f"batch-media-fetch-test:{media_kind}:{kind}",
                id=f"batch_{media_kind}_{kind}",
            )
            for media_kind in ("image", "audio", "video")
            for kind in (
                "connection",
                "timeout",
                "mixed_fetch_first",
                "mixed_decode_first",
            )
        ],
        pytest.param("dispatch-error-test:received", id="dispatch_received"),
        pytest.param("dispatch-error-test:no_received", id="dispatch_no_received"),
        pytest.param("dispatch-error-test:untyped", id="dispatch_untyped"),
    ],
)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_batch_media_and_dispatch_privacy_over_http_and_sse(
    pixel_limit_http_server, trigger, stream
):
    """Mask aggregate/dispatch diagnostics and keep the service usable afterward."""
    if trigger.startswith("batch-media-fetch-test:") and ImageLoader is None:
        pytest.skip("multimodal loader dependencies are not installed")
    base_url, model_name = pixel_limit_http_server
    data = {
        "model": model_name,
        "messages": [{"role": "user", "content": trigger}],
        "stream": stream,
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=data
        ) as response:
            raw = await response.text()
            assert response.status == (200 if stream else 500), raw
            if stream:
                assert response.content_type == "text/event-stream"
        error = _sse_error(raw) if stream else json.loads(raw)
        assert error == expected_error_body(
            500,
            "Internal server error",
            "internal_server_error" if stream else "Internal Server Error",
            backend_status=None if stream else 500,
        ), raw
        assert "PRIVATE_" not in raw
        assert "10.20.30." not in raw
        assert "private-batch-host" not in raw
        assert "errno" not in raw

        data["messages"][0]["content"] = "Reply normally after the rejected request."
        data["stream"] = False
        async with session.post(
            f"{base_url}/v1/chat/completions", json=data
        ) as response:
            recovered = await response.text()
            assert response.status == 200, recovered
            assert "This is a mock response." in recovered


@pytest.mark.asyncio
@pytest.mark.skipif(
    ImageLoader is None, reason="multimodal image dependencies are not installed"
)
@pytest.mark.parametrize(
    "http_server,origin_status,after_commit",
    [({"wait_for_first_item": True}, 403, False), ({"peek_ms": 20}, 404, True)],
    indirect=["http_server"],
)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_image_fetch_recovery_advice_survives_http_and_sse(
    http_server, unused_tcp_port_factory, monkeypatch, origin_status, after_commit
):
    monkeypatch.setenv("DYN_MM_ALLOW_INTERNAL", "1")
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    origin_port = unused_tcp_port_factory()
    release_origin = asyncio.Event()

    async def refuse_image(request):
        await asyncio.wait_for(release_origin.wait(), timeout=10)
        return web.Response(status=origin_status, text="PRIVATE_ORIGIN_DETAILS")

    app = web.Application()
    app.router.add_get("/image.png", refuse_image)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", origin_port).start()
        if not after_commit:
            release_origin.set()
        base_url, model_name = http_server
        data = {
            "model": model_name,
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"http://127.0.0.1:{origin_port}/image.png?token=SECRET"
                            },
                        }
                    ],
                }
            ],
        }
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        ) as session:
            async with session.post(
                f"{base_url}/v1/chat/completions", json=data
            ) as response:
                assert response.status == (200 if after_commit else 400)
                # The origin cannot fail until HTTP 200 has been committed.
                release_origin.set()
                raw = await response.text()
        error = _sse_error(raw) if after_commit else json.loads(raw)
        assert error["code"] == 400
        assert error["message"].startswith(
            f"Image download returned HTTP {origin_status}."
        )
        assert "resend the image data before retrying" in error["message"]
        assert "SECRET" not in raw
        assert "PRIVATE_ORIGIN_DETAILS" not in raw
        assert "This is a mock response." not in raw
    finally:
        release_origin.set()
        await runner.cleanup()


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
                400, MSG_CONTAINS_ERROR, "Bad Request"
            )


@pytest.mark.asyncio
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_streaming_chat_completion_http_error_default_commits_200(http_server):
    """The default streaming service reports backend errors inside HTTP 200."""
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

    # HTTP 200 is already committed; the SSE frame retains the error's status
    # and the same forwarded engine message as the unary response.
    error = _sse_error(body)
    assert error["message"] == MSG_CONTAINS_ERROR
    assert error["code"] == 400
    assert "mock response" not in body, (
        f"the generator raised before its first yield, so the stream must "
        f"carry no content; got: {body}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("http_server", "stream", "trigger", "status", "has_header"),
    [
        pytest.param({}, False, "", 200, True, id="unary"),
        pytest.param({}, False, MSG_CONTAINS_ERROR, 400, True, id="unary_error"),
        pytest.param(
            {"wait_for_first_item": True},
            True,
            "",
            200,
            True,
            id="stream_wait_for_first_item",
        ),
        pytest.param({}, True, "", 200, False, id="stream_default"),
        pytest.param({"peek_ms": 1000}, True, "", 200, True, id="stream_peek_enabled"),
    ],
    indirect=["http_server"],
)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_chat_engine_response_header(
    http_server, stream: bool, trigger: str, status: int, has_header: bool
):
    """A header the engine sets through its context is on the response, error
    responses included, when the service reads the engine's stream before it
    commits the response. Disabling the peek commits streaming headers first."""
    base_url, model_name = http_server
    data = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": f"{MSG_SETS_RESPONSE_HEADER} {trigger}"}
        ],
        "stream": stream,
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=data
        ) as response:
            assert response.status == status
            assert response.headers.get("x-mock-engine") == (
                "set" if has_header else None
            )
            await response.read()


def test_set_response_header_without_an_http_response():
    # A context with no HTTP response behind it reports that nothing was set.
    assert Context().set_response_header("x-mock-engine", "set") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize(
    ("status", "message", "public"),
    [
        (413, "Image exceeds the decoded pixel limit", True),
        (
            413,
            "Image exceeds the decoded pixel limit https://example.com/?secret=private",
            False,
        ),
        (413, "private https://example.com/?secret=private", False),
        (400, "Image exceeds the decoded pixel limit", True),
    ],
)
@pytest.mark.timeout(60)
@pytest.mark.forked
async def test_image_pixel_limit_public_response(
    pixel_limit_http_server, stream, status, message, public
):
    base_url, model_name = pixel_limit_http_server
    data = {
        "model": model_name,
        "stream": stream,
        "messages": [
            {
                "role": "user",
                "content": "public-http-test:" + json.dumps([status, message]),
            }
        ],
    }
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        async with session.post(
            f"{base_url}/v1/chat/completions", json=data
        ) as response:
            assert response.status == (200 if stream else status)
            raw = await response.text()
    error = _sse_error(raw) if stream else json.loads(raw)
    assert error["code"] == status
    if public:
        error_type = {400: "Bad Request", 413: "Payload Too Large"}[status]
        assert error == {
            "message": "Image exceeds the decoded pixel limit",
            "type": "invalid_request_error" if stream else error_type,
            "code": status,
        }
    else:
        assert error["message"] != "Image exceeds the decoded pixel limit"
        assert "private" not in raw
        assert "secret" not in raw
