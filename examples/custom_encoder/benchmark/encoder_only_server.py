# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a custom vision encoder through the OpenAI chat-completions wire format."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass
from typing import Any, AsyncIterator, Protocol

from aiohttp import ContentTypeError, web

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_REQUEST_SIZE_MIB = 64
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.qwen2_5_vl_benchmark_encoder." "Qwen2_5VLBenchmarkEncoder"
)
_DUMMY_CONTENT = "ok"


class Encoder(Protocol):
    async def encode(self, raws: list[str]) -> list[Any]:
        ...

    def shutdown(self) -> None:
        ...


@dataclass(frozen=True)
class EncoderRequest:
    image_url: str
    model: str
    stream: bool
    include_usage: bool


class RequestValidationError(ValueError):
    """A client request that cannot be represented as one encoder call."""


ENCODER_KEY = web.AppKey("encoder", Encoder)
MODEL_KEY = web.AppKey("model", str)


def _extract_image_url(messages: Any) -> str:
    if not isinstance(messages, list) or not messages:
        raise RequestValidationError("'messages' must be a non-empty array")

    image_urls: list[str] = []
    for message in messages:
        if not isinstance(message, dict):
            raise RequestValidationError("every message must be an object")
        content = message.get("content")
        if isinstance(content, str) or content is None:
            continue
        if not isinstance(content, list):
            raise RequestValidationError(
                "message 'content' must be a string or content-part array"
            )
        for part in content:
            if not isinstance(part, dict):
                raise RequestValidationError("every content part must be an object")
            if part.get("type") != "image_url":
                continue
            image_url = part.get("image_url")
            if not isinstance(image_url, dict) or not isinstance(
                image_url.get("url"), str
            ):
                raise RequestValidationError(
                    "an image_url part must contain image_url.url"
                )
            image_urls.append(image_url["url"])

    if len(image_urls) != 1:
        raise RequestValidationError(
            f"exactly one image_url is required; received {len(image_urls)}"
        )

    image_url = image_urls[0]
    header, separator, payload = image_url.partition(",")
    if (
        not separator
        or not header.startswith("data:image/")
        or not header.endswith(";base64")
        or not payload
    ):
        raise RequestValidationError(
            "image_url.url must be an inline base64 image data URI"
        )
    return image_url


def parse_request(body: Any, default_model: str) -> EncoderRequest:
    if not isinstance(body, dict):
        raise RequestValidationError("request body must be a JSON object")

    model = body.get("model", default_model)
    if not isinstance(model, str) or not model:
        raise RequestValidationError("'model' must be a non-empty string")

    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise RequestValidationError("'stream' must be a boolean")

    stream_options = body.get("stream_options")
    if stream_options is not None and not isinstance(stream_options, dict):
        raise RequestValidationError("'stream_options' must be an object")
    include_usage = bool(stream_options and stream_options.get("include_usage", False))

    return EncoderRequest(
        image_url=_extract_image_url(body.get("messages")),
        model=model,
        stream=stream,
        include_usage=include_usage,
    )


def _usage() -> dict[str, int]:
    return {"prompt_tokens": 0, "completion_tokens": 1, "total_tokens": 1}


def _error_response(message: str, status: int, error_type: str) -> web.Response:
    return web.json_response(
        {
            "error": {
                "message": message,
                "type": error_type,
                "param": None,
                "code": None,
            }
        },
        status=status,
    )


def _completion_payload(request_id: str, created: int, model: str) -> dict[str, Any]:
    return {
        "id": request_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": _DUMMY_CONTENT},
                "finish_reason": "stop",
            }
        ],
        "usage": _usage(),
    }


def _stream_chunks(
    request_id: str, created: int, model: str, include_usage: bool
) -> list[dict[str, Any]]:
    chunks = [
        {
            "id": request_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": _DUMMY_CONTENT},
                    "finish_reason": "stop",
                }
            ],
        }
    ]
    if include_usage:
        chunks.append(
            {
                "id": request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [],
                "usage": _usage(),
            }
        )
    return chunks


async def _write_stream(
    request: web.Request,
    request_id: str,
    created: int,
    encoder_request: EncoderRequest,
) -> web.StreamResponse:
    response = web.StreamResponse(
        status=200,
        headers={
            "Cache-Control": "no-cache",
            "Content-Type": "text/event-stream",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    for chunk in _stream_chunks(
        request_id,
        created,
        encoder_request.model,
        encoder_request.include_usage,
    ):
        encoded = json.dumps(chunk, separators=(",", ":")).encode("utf-8")
        await response.write(b"data: " + encoded + b"\n\n")
    await response.write(b"data: [DONE]\n\n")
    await response.write_eof()
    return response


async def handle_chat_completion(request: web.Request) -> web.StreamResponse:
    try:
        body = await request.json()
    except (ContentTypeError, json.JSONDecodeError, UnicodeDecodeError):
        return _error_response(
            "request body must contain valid JSON", 400, "invalid_request_error"
        )

    try:
        encoder_request = parse_request(body, request.app[MODEL_KEY])
    except RequestValidationError as error:
        return _error_response(str(error), 400, "invalid_request_error")

    request_id = f"chatcmpl-encoder-{uuid.uuid4().hex}"
    created = int(time.time())
    start = time.perf_counter()
    try:
        outputs = await request.app[ENCODER_KEY].encode([encoder_request.image_url])
        if len(outputs) != 1:
            raise RuntimeError(
                f"custom encoder returned {len(outputs)} outputs for one image"
            )
    except Exception as error:
        logger.exception("Custom encoder request %s failed", request_id)
        return _error_response(str(error), 500, "server_error")
    logger.debug(
        "Custom encoder request %s completed in %.3f ms",
        request_id,
        (time.perf_counter() - start) * 1000,
    )

    if encoder_request.stream:
        return await _write_stream(request, request_id, created, encoder_request)
    return web.json_response(
        _completion_payload(request_id, created, encoder_request.model)
    )


async def handle_health(request: web.Request) -> web.Response:
    del request
    return web.json_response({"status": "ready"})


async def _encoder_lifecycle(app: web.Application) -> AsyncIterator[None]:
    yield
    app[ENCODER_KEY].shutdown()


def create_app(
    encoder: Encoder,
    model: str = DEFAULT_MODEL,
    max_request_size_mib: int = DEFAULT_MAX_REQUEST_SIZE_MIB,
) -> web.Application:
    if max_request_size_mib <= 0:
        raise ValueError("max_request_size_mib must be positive")
    app = web.Application(client_max_size=max_request_size_mib * 1024**2)
    app[ENCODER_KEY] = encoder
    app[MODEL_KEY] = model
    app.router.add_get("/health", handle_health)
    app.router.add_post("/v1/chat/completions", handle_chat_completion)
    app.cleanup_ctx.append(_encoder_lifecycle)
    return app


def _custom_encoder_api() -> Any:
    return importlib.import_module("dynamo.vllm.multimodal_utils.custom_encoder")


def _load_backend(class_path: str) -> Any:
    module_path, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError("--custom-encoder-class must be a dotted module.Class path")
    backend_class = getattr(importlib.import_module(module_path), class_name)
    backend_base = _custom_encoder_api().VisionEncoderBackend
    if not (
        isinstance(backend_class, type) and issubclass(backend_class, backend_base)
    ):
        raise TypeError(
            f"--custom-encoder-class {class_path!r} must resolve to a "
            f"VisionEncoderBackend subclass, got {backend_class!r}"
        )
    return backend_class()


def _positive_port(value: str) -> int:
    port = int(value)
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be nonnegative")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _qwen_benchmark_defaults() -> None:
    os.environ.setdefault("DYN_QWEN2_VL_ENCODER_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    os.environ.setdefault("DYN_QWEN2_VL_OUTPUT_HIDDEN_SIZE", "1536")
    os.environ.setdefault("DYN_QWEN2_VL_PREPROCESS_CONCURRENCY", "64")
    os.environ.setdefault("DYN_QWEN2_VL_MAX_BATCH_COST", "64")
    os.environ.setdefault("DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS", "1,2,4,8,16,32,64")
    os.environ.setdefault("DYN_QWEN2_VL_GRAPH_IMAGE_SIZES", "500x500")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.environ.get("DYN_HTTP_HOST", "0.0.0.0"))
    parser.add_argument(
        "--port",
        type=_positive_port,
        default=_positive_port(os.environ.get("DYN_HTTP_PORT", "8000")),
    )
    parser.add_argument(
        "--model", default=os.environ.get("DYN_ENCODER_ONLY_MODEL", DEFAULT_MODEL)
    )
    parser.add_argument(
        "--custom-encoder-class",
        default=os.environ.get("DYN_CUSTOM_ENCODER_CLASS", DEFAULT_ENCODER_CLASS),
    )
    parser.add_argument(
        "--max-queue-delay-us",
        type=_nonnegative_int,
        default=_nonnegative_int(
            os.environ.get("DYN_CUSTOM_ENCODER_MAX_QUEUE_DELAY_US", "0")
        ),
    )
    parser.add_argument(
        "--max-request-size-mib",
        type=_positive_int,
        default=_positive_int(os.environ.get("DYN_HTTP_MAX_REQUEST_SIZE_MIB", "64")),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=os.environ.get("DYN_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.custom_encoder_class == DEFAULT_ENCODER_CLASS:
        _qwen_benchmark_defaults()

    backend = _load_backend(args.custom_encoder_class)
    encoder = _custom_encoder_api().AsyncVisionEncoder(
        backend,
        max_queue_delay_us=args.max_queue_delay_us,
        name="encoder-only",
    )
    encoder.load(args.model)
    logger.info(
        "Loaded custom encoder %s for %s; serving http://%s:%d",
        args.custom_encoder_class,
        args.model,
        args.host,
        args.port,
    )
    try:
        web.run_app(
            create_app(encoder, args.model, args.max_request_size_mib),
            host=args.host,
            port=args.port,
        )
    finally:
        encoder.shutdown()


if __name__ == "__main__":
    main()
