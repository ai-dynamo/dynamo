# SPDX-FileCopyrightText: Copyright (c) 2026 doubleword.ai
# SPDX-License-Identifier: MIT

"""Register a Dynamo text worker that forwards OpenAI requests to a local engine."""

import argparse
import asyncio
import contextlib
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Optional, TypeVar
from urllib.parse import urlsplit

import httpx
import uvloop
from httpx_sse import aconnect_sse

from dynamo.common.utils.graceful_shutdown import install_signal_handlers
from dynamo.llm import HttpError, ModelInput, ModelType, register_model
from dynamo.runtime import DistributedRuntime, Endpoint, dynamo_worker

LOGGER = logging.getLogger("dynamo.openai_backend.worker")

DEFAULT_UPSTREAM_BASE_URL = "http://127.0.0.1:30000/v1"
DEFAULT_UPSTREAM_HEALTH_PATH = "/health"
DEFAULT_CONNECT_TIMEOUT_SECONDS = 30.0
DEFAULT_WRITE_TIMEOUT_SECONDS = 100.0
DEFAULT_MAX_KEEPALIVE_CONNECTIONS = 20

_SHUTDOWN_EVENT = asyncio.Event()
_WORKER_ARGV: list[str] | None = None
T = TypeVar("T")


@dataclass
class Config:
    model: str
    served_model_name: Optional[str]
    upstream_base_url: str
    upstream_health_path: str
    upstream_api_key: Optional[str]
    connect_timeout_seconds: float
    write_timeout_seconds: float


@dataclass
class _BufferedToolCall:
    id: Optional[str] = None
    type: Optional[str] = None
    name: Optional[str] = None
    arguments: str = ""


class _ToolCallCoalescer:
    """Merge OpenAI-style streamed tool-call deltas into complete chunks."""

    def __init__(self) -> None:
        self._calls: dict[tuple[int, int], _BufferedToolCall] = {}

    def push(self, chunk: dict[str, Any]) -> list[dict[str, Any]]:
        choices = chunk.get("choices")
        if not isinstance(choices, list):
            return [chunk]

        output_choices = []

        for choice in choices:
            if not isinstance(choice, dict):
                output_choices.append(choice)
                continue

            delta = choice.get("delta")
            if not isinstance(delta, dict):
                output_choices.append(choice)
                continue

            choice_index = self._choice_index(choice)
            finish_reason = choice.get("finish_reason")
            tool_calls = delta.get("tool_calls")

            if isinstance(tool_calls, list) and tool_calls:
                self._accumulate(choice_index, tool_calls)
                continue

            if self._has_pending_choice(choice_index) and finish_reason is not None:
                output_choices.append(
                    self._build_choice(choice, choice_index, finish_reason)
                )
                self._clear_choice(choice_index)
                continue

            output_choices.append(choice)

        if not output_choices:
            return []

        output_chunk = dict(chunk)
        output_chunk["choices"] = output_choices
        return [output_chunk]

    @staticmethod
    def _choice_index(choice: dict[str, Any]) -> int:
        index = choice.get("index", 0)
        return index if isinstance(index, int) else 0

    @staticmethod
    def _tool_call_index(tool_call: dict[str, Any]) -> int:
        index = tool_call.get("index", 0)
        return index if isinstance(index, int) else 0

    def _has_pending_choice(self, choice_index: int) -> bool:
        return any(
            state_choice_index == choice_index for state_choice_index, _ in self._calls
        )

    def _accumulate(self, choice_index: int, tool_calls: list[Any]) -> None:
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue

            key = (choice_index, self._tool_call_index(tool_call))
            buffered = self._calls.setdefault(key, _BufferedToolCall())

            tool_call_id = tool_call.get("id")
            if isinstance(tool_call_id, str) and tool_call_id:
                buffered.id = buffered.id or tool_call_id

            tool_call_type = tool_call.get("type")
            if isinstance(tool_call_type, str) and tool_call_type:
                buffered.type = buffered.type or tool_call_type

            function = tool_call.get("function")
            if not isinstance(function, dict):
                continue

            name = function.get("name")
            if isinstance(name, str) and name:
                buffered.name = buffered.name or name

            arguments = function.get("arguments")
            if isinstance(arguments, str):
                buffered.arguments += arguments

    def _build_choice(
        self,
        source_choice: dict[str, Any],
        choice_index: int,
        finish_reason: Any,
    ) -> dict[str, Any]:
        choice = {
            key: value
            for key, value in source_choice.items()
            if key not in {"delta", "finish_reason"}
        }
        choice["index"] = choice_index
        choice["delta"] = {
            "role": "assistant",
            "tool_calls": self._complete_tool_calls(choice_index),
        }
        choice["finish_reason"] = finish_reason
        return choice

    def _complete_tool_calls(self, choice_index: int) -> list[dict[str, Any]]:
        completed = []
        for (state_choice_index, tool_call_index), buffered in sorted(
            self._calls.items()
        ):
            if state_choice_index != choice_index:
                continue

            completed.append(
                {
                    "index": tool_call_index,
                    "id": buffered.id or f"call_{choice_index}_{tool_call_index}",
                    "type": buffered.type or "function",
                    "function": {
                        "name": buffered.name or "",
                        "arguments": buffered.arguments,
                    },
                }
            )
        return completed

    def _clear_choice(self, choice_index: int) -> None:
        for key in [key for key in self._calls if key[0] == choice_index]:
            del self._calls[key]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamo worker that forwards requests to a local OpenAI-compatible "
            "upstream."
        )
    )
    parser.add_argument("--model", required=True, help="Model identifier to register.")
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Optional public model name to register with Dynamo.",
    )
    parser.add_argument(
        "--upstream-base-url",
        default=DEFAULT_UPSTREAM_BASE_URL,
        help=(
            "Base URL for the raw engine, including the API prefix, with no "
            f"trailing slash. Default: {DEFAULT_UPSTREAM_BASE_URL}"
        ),
    )
    parser.add_argument(
        "--upstream-health-path",
        default=DEFAULT_UPSTREAM_HEALTH_PATH,
        help=(
            "Health path on the upstream engine in strict '/path' format. "
            f"Default: {DEFAULT_UPSTREAM_HEALTH_PATH}"
        ),
    )
    parser.add_argument(
        "--upstream-api-key",
        default=None,
        help="Optional API key to send to the upstream engine.",
    )
    parser.add_argument(
        "--connect-timeout-seconds",
        type=float,
        default=DEFAULT_CONNECT_TIMEOUT_SECONDS,
        help="HTTP connect timeout for upstream calls.",
    )
    parser.add_argument(
        "--write-timeout-seconds",
        type=float,
        default=DEFAULT_WRITE_TIMEOUT_SECONDS,
        help="HTTP write timeout for upstream calls.",
    )
    return parser


def cmd_line_args(argv: Sequence[str] | None = None) -> Config:
    args = _build_parser().parse_args(list(argv) if argv is not None else None)

    return Config(
        model=args.model,
        served_model_name=args.served_model_name,
        upstream_base_url=args.upstream_base_url,
        upstream_health_path=args.upstream_health_path,
        upstream_api_key=args.upstream_api_key,
        connect_timeout_seconds=args.connect_timeout_seconds,
        write_timeout_seconds=args.write_timeout_seconds,
    )


def _normalize_chat_template_kwargs(request: dict[str, Any]) -> None:
    chat_template_args = request.pop("chat_template_args", None)
    if "chat_template_kwargs" not in request and chat_template_args is not None:
        request["chat_template_kwargs"] = chat_template_args


class UpstreamClient:
    def __init__(self, config: Config):
        headers = {}
        if config.upstream_api_key:
            headers["Authorization"] = f"Bearer {config.upstream_api_key}"

        timeout = httpx.Timeout(
            connect=config.connect_timeout_seconds,
            read=None,
            write=config.write_timeout_seconds,
            pool=None,
        )
        limits = httpx.Limits(
            max_connections=None,
            max_keepalive_connections=DEFAULT_MAX_KEEPALIVE_CONNECTIONS,
        )
        self._config = config
        self._client = httpx.AsyncClient(
            headers=headers,
            timeout=timeout,
            limits=limits,
        )
        split_result = urlsplit(config.upstream_base_url)
        self._origin = f"{split_result.scheme}://{split_result.netloc}"
        self._api_prefix = split_result.path.rstrip("/")

    async def aclose(self) -> None:
        await self._client.aclose()

    async def wait_until_ready(self) -> None:
        while True:
            if _SHUTDOWN_EVENT.is_set():
                raise RuntimeError("shutdown requested while waiting for upstream")

            try:
                timeout = httpx.Timeout(
                    connect=self._config.connect_timeout_seconds,
                    read=self._config.connect_timeout_seconds,
                    write=self._config.connect_timeout_seconds,
                    pool=None,
                )
                response = await self._client.get(self._health_url(), timeout=timeout)
                if response.is_success:
                    LOGGER.info(
                        "Upstream became healthy at %s%s",
                        self._config.upstream_base_url,
                        self._config.upstream_health_path,
                    )
                    return
            except Exception:
                LOGGER.debug("Upstream is not ready yet", exc_info=True)

            await asyncio.sleep(2.0)

    async def forward(
        self,
        request: dict[str, Any],
        context: Optional[Any] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        try:
            path = self._resolve_upstream_path(request)
            forwarded_request = dict(request)
            forwarded_request["stream"] = True
            _normalize_chat_template_kwargs(forwarded_request)

            tool_call_coalescer = (
                _ToolCallCoalescer()
                if path == "/chat/completions" and request.get("tools")
                else None
            )

            async for chunk in self._stream_request(path, forwarded_request, context):
                if tool_call_coalescer is None:
                    yield chunk
                    continue

                for output_chunk in tool_call_coalescer.push(chunk):
                    yield output_chunk
        except asyncio.CancelledError:
            LOGGER.info("Dropping cancelled request")
            return

    def _resolve_upstream_path(self, request: dict[str, Any]) -> str:
        if "messages" in request:
            return "/chat/completions"
        if "prompt" in request:
            return "/completions"
        raise HttpError(
            400,
            "OpenAI backend worker expected either a chat-completions or completions request.",
        )

    def _check_runtime_state(self, context: Optional[Any]) -> None:
        if _SHUTDOWN_EVENT.is_set():
            raise GeneratorExit("worker shutting down; request can be migrated")
        if context is not None and context.is_stopped():
            raise asyncio.CancelledError("request was cancelled")

    def _health_url(self) -> str:
        return f"{self._origin}{self._config.upstream_health_path}"

    def _request_url(self, path: str) -> str:
        return f"{self._origin}{self._api_prefix}{path}"

    @staticmethod
    async def _cancel_future(future: asyncio.Future[Any]) -> None:
        if future.done():
            return
        future.cancel()
        try:
            await future
        except asyncio.CancelledError:
            pass

    async def _await_with_runtime_cancellation(
        self,
        awaitable: Awaitable[T],
        context: Optional[Any],
    ) -> T:
        operation = asyncio.ensure_future(awaitable)
        shutdown_wait = asyncio.create_task(_SHUTDOWN_EVENT.wait())
        cancellation_wait = None
        operation_finished = False
        worker_shutdown = False
        request_cancelled = False

        try:
            wait_for: list[asyncio.Future[Any]] = [operation, shutdown_wait]
            if context is not None:
                cancellation_wait = asyncio.ensure_future(
                    context.async_killed_or_stopped()
                )
                wait_for.append(cancellation_wait)

            done, _ = await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )

            operation_finished = operation in done
            worker_shutdown = shutdown_wait in done
            request_cancelled = (
                cancellation_wait is not None and cancellation_wait in done
            )

            if operation_finished:
                return operation.result()

            await self._cancel_future(operation)

            if worker_shutdown:
                raise GeneratorExit("worker shutting down; request can be migrated")

            if request_cancelled:
                raise asyncio.CancelledError("request was cancelled")

            raise RuntimeError("unexpected runtime cancellation state")
        finally:
            if not worker_shutdown:
                await self._cancel_future(shutdown_wait)
            if cancellation_wait is not None and not request_cancelled:
                await self._cancel_future(cancellation_wait)

    @contextlib.asynccontextmanager
    async def _open_cancellable_sse(
        self,
        path: str,
        request: dict[str, Any],
        context: Optional[Any],
    ) -> AsyncGenerator[Any, None]:
        cm = aconnect_sse(
            self._client,
            "POST",
            self._request_url(path),
            json=request,
            headers={"Content-Type": "application/json"},
        )
        event_source = await self._await_with_runtime_cancellation(
            cm.__aenter__(),
            context,
        )
        try:
            yield event_source
        except BaseException as exc:
            suppress = await cm.__aexit__(type(exc), exc, exc.__traceback__)
            if not suppress:
                raise
        else:
            await cm.__aexit__(None, None, None)

    async def _stream_request(
        self,
        path: str,
        request: dict[str, Any],
        context: Optional[Any],
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._check_runtime_state(context)

        try:
            async with self._open_cancellable_sse(path, request, context) as event_source:
                if event_source.response.status_code >= 400:
                    raise await self._as_http_error(event_source.response)

                sse_iterator = event_source.aiter_sse()
                while True:
                    try:
                        sse = await self._await_with_runtime_cancellation(
                            sse_iterator.__anext__(),
                            context,
                        )
                    except StopAsyncIteration:
                        return

                    self._check_runtime_state(context)
                    if sse.data == "[DONE]":
                        return
                    yield self._decode_sse_payload(sse.data)
        except httpx.HTTPError as exc:
            raise HttpError(502, f"Upstream streaming request failed: {exc}") from exc

    @staticmethod
    def _decode_sse_payload(payload: str) -> dict[str, Any]:
        try:
            decoded = json.loads(payload)
        except ValueError as exc:
            raise HttpError(
                502,
                f"Upstream returned invalid JSON in a streaming chunk: {payload[:200]}",
            ) from exc

        if not isinstance(decoded, dict):
            raise HttpError(
                502,
                f"Upstream returned {type(decoded).__name__} in a streaming chunk; expected a JSON object.",
            )

        return decoded

    async def _as_http_error(self, response: httpx.Response) -> HttpError:
        message = None

        try:
            payload = await response.aread()
        except httpx.HTTPError as exc:
            return HttpError(
                502,
                f"Upstream request failed while reading the error response: {exc}",
            )

        if payload:
            try:
                decoded = json.loads(payload)
            except ValueError:
                message = payload.decode("utf-8", "replace").strip()
            else:
                if isinstance(decoded, dict):
                    error_value = decoded.get("error")
                    if isinstance(error_value, dict):
                        message = error_value.get("message")
                    if message is None:
                        top_level_message = decoded.get("message")
                        if isinstance(top_level_message, str):
                            message = top_level_message

        if not message:
            message = (
                f"Upstream returned HTTP {response.status_code} with an empty error body."
            )

        return HttpError(response.status_code, message)


class RequestHandler:
    def __init__(self, upstream: UpstreamClient):
        self._upstream = upstream

    async def generate(
        self,
        request: dict[str, Any],
        context: Optional[Any] = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        async for chunk in self._upstream.forward(request, context):
            yield chunk


def _configure_logging() -> None:
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _runtime_endpoint_name() -> str:
    namespace = os.environ.get("DYN_NAMESPACE", "dynamo")
    return f"{namespace}.backend.generate"


def _enable_nats_from_env() -> bool:
    return os.environ.get("DYN_REQUEST_PLANE", "tcp") == "nats"


@dynamo_worker(enable_nats=_enable_nats_from_env())
async def worker(runtime: DistributedRuntime) -> None:
    _configure_logging()
    _SHUTDOWN_EVENT.clear()

    config = cmd_line_args(_WORKER_ARGV)
    endpoint_name = _runtime_endpoint_name()
    endpoint = runtime.endpoint(endpoint_name)

    install_signal_handlers(
        asyncio.get_running_loop(),
        runtime,
        [endpoint],
        shutdown_event=_SHUTDOWN_EVENT,
    )
    await init(runtime, config, endpoint, endpoint_name)


async def init(
    runtime: DistributedRuntime,
    config: Config,
    endpoint: Endpoint,
    endpoint_name: str,
) -> None:
    upstream = UpstreamClient(config)

    try:
        await upstream.wait_until_ready()

        await register_model(
            ModelInput.Text,
            ModelType.Chat | ModelType.Completions,
            endpoint,
            config.model,
            model_name=config.served_model_name,
        )

        LOGGER.info(
            "Registered OpenAI backend worker for model '%s' on endpoint '%s'",
            config.served_model_name or config.model,
            endpoint_name,
        )

        await endpoint.serve_endpoint(RequestHandler(upstream).generate)
    finally:
        await upstream.aclose()


def worker_main(argv: Sequence[str] | None = None) -> None:
    global _WORKER_ARGV

    _WORKER_ARGV = list(argv) if argv is not None else None
    uvloop.install()
    asyncio.run(worker())
