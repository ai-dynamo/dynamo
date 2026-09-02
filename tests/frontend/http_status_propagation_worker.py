# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workers for the client-error propagation e2e tests.

Two models are registered from one process because the two tests need two
different frontend pipelines:

``MODEL_NAME`` (``ModelInput.Tokens``)
    The Rust preprocessor runs. Used to prove a duck-typed ``.status=415``
    exception survives the wire.

``PASSTHROUGH_MODEL_NAME`` (``ModelInput.Text``)
    The frontend forwards the OpenAI request without Rust token or media
    preprocessing. This lets the test exercise Python's ``ImageLoader``.
"""

from __future__ import annotations

import asyncio

import uvloop

from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.frontend.utils import extract_mm_urls
from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_http_status_propagation import (
    ENDPOINT_PATH,
    EXPECTED_MESSAGE,
    EXPECTED_STATUS,
    MODEL_NAME,
    PASSTHROUGH_ENDPOINT_PATH,
    PASSTHROUGH_MODEL_NAME,
)
from tests.utils.constants import QWEN


class _StatusLikeError(Exception):
    """Duck-typed `.status` + `.message` — same shape as
    `dynamo.common.http.HttpStatusError`."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


async def generate(request, context):
    raise _StatusLikeError(status=EXPECTED_STATUS, message=EXPECTED_MESSAGE)
    yield  # unreachable; needed to make this an async generator


async def generate_passthrough(request, context):
    # Decode media in Python using the same logic as a URL-passthrough backend.
    mm_data, _ = extract_mm_urls(request.get("messages") or [])
    image_items = (mm_data or {}).get("image_url")
    if image_items:
        # Exercise the real Python backend-decoding batch boundary. A blocked
        # URL must raise before this generator can produce or delegate work.
        await ImageLoader().load_image_batch(image_items)
        raise AssertionError("blocked media URL unexpectedly passed validation")
    raise _StatusLikeError(status=EXPECTED_STATUS, message=EXPECTED_MESSAGE)
    yield  # unreachable; needed to make this an async generator


async def main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "tcp")
    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        QWEN,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )

    passthrough_endpoint = runtime.endpoint(PASSTHROUGH_ENDPOINT_PATH)
    await register_model(
        ModelInput.Text,
        ModelType.Chat,
        passthrough_endpoint,
        QWEN,
        model_name=PASSTHROUGH_MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )

    await asyncio.gather(
        endpoint.serve_endpoint(generate),
        passthrough_endpoint.serve_endpoint(generate_passthrough),
    )


if __name__ == "__main__":
    uvloop.run(main())
