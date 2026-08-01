# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Serve a custom vision encoder through a minimal multipart HTTP endpoint."""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import time
from collections.abc import AsyncIterator
from typing import Any, Protocol

from aiohttp import web

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_MAX_REQUEST_SIZE_MIB = 64
DEFAULT_ENCODER_CLASS = (
    "examples.custom_encoder.qwen2_5_vl_benchmark_encoder." "Qwen2_5VLBenchmarkEncoder"
)
DUMMY_RESPONSE = b"encoder-ok"


class Encoder(Protocol):
    async def encode(self, raws: list[bytes]) -> list[Any]:
        ...

    def shutdown(self) -> None:
        ...


class RequestValidationError(ValueError):
    """A client request that cannot be represented as one encoder call."""


ENCODER_KEY = web.AppKey("encoder", Encoder)


def _error_response(message: str, status: int) -> web.Response:
    return web.Response(text=message, status=status, content_type="text/plain")


async def _read_jpeg(request: web.Request) -> bytes:
    if request.content_type != "multipart/form-data":
        raise RequestValidationError("request must use multipart/form-data")
    try:
        reader = await request.multipart()
        part = await reader.next()
    except (AssertionError, ValueError) as error:
        raise RequestValidationError(
            "request contains invalid multipart data"
        ) from error
    if part is None or part.name != "image":
        raise RequestValidationError(
            "exactly one multipart field named 'image' is required"
        )
    content_type = part.headers.get("Content-Type", "").partition(";")[0].lower()
    if content_type != "image/jpeg":
        raise RequestValidationError("the 'image' field must use image/jpeg")
    image = bytes(await part.read(decode=False))
    if not image:
        raise RequestValidationError("the 'image' field must not be empty")
    if await reader.next() is not None:
        raise RequestValidationError(
            "exactly one multipart field named 'image' is required"
        )
    return image


async def handle_encode(request: web.Request) -> web.Response:
    try:
        image = await _read_jpeg(request)
    except RequestValidationError as error:
        return _error_response(str(error), 400)

    started = time.perf_counter()
    try:
        outputs = await request.app[ENCODER_KEY].encode([image])
        if len(outputs) != 1:
            raise RuntimeError(
                f"custom encoder returned {len(outputs)} outputs for one image"
            )
    except Exception as error:
        logger.exception("Custom encoder request failed")
        return _error_response(str(error), 500)
    logger.debug(
        "Custom encoder request completed in %.3f ms",
        (time.perf_counter() - started) * 1000,
    )
    return web.Response(body=DUMMY_RESPONSE, content_type="text/plain")


async def handle_health(request: web.Request) -> web.Response:
    del request
    return web.json_response({"status": "ready"})


async def _encoder_lifecycle(app: web.Application) -> AsyncIterator[None]:
    yield
    app[ENCODER_KEY].shutdown()


def create_app(
    encoder: Encoder,
    max_request_size_mib: int = DEFAULT_MAX_REQUEST_SIZE_MIB,
) -> web.Application:
    if max_request_size_mib <= 0:
        raise ValueError("max_request_size_mib must be positive")
    app = web.Application(client_max_size=max_request_size_mib * 1024**2)
    app[ENCODER_KEY] = encoder
    app.router.add_get("/health", handle_health)
    app.router.add_post("/encode", handle_encode)
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
    os.environ.setdefault("DYN_QWEN2_VL_MAX_BATCH_PATCHES", str(64 * 36 * 36))
    os.environ.setdefault("DYN_QWEN2_VL_MAX_BATCH_ITEMS", "64")
    os.environ.setdefault("DYN_QWEN2_VL_GRAPH_BATCH_BUCKETS", "1,2,4,8,16,32,64")
    os.environ.setdefault("DYN_QWEN2_VL_GRAPH_IMAGE_SIZES", "300x300,500x500")


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
        "Loaded custom encoder %s for %s; serving http://%s:%d/encode",
        args.custom_encoder_class,
        args.model,
        args.host,
        args.port,
    )
    web.run_app(
        create_app(encoder, args.max_request_size_mib),
        host=args.host,
        port=args.port,
    )


if __name__ == "__main__":
    main()
