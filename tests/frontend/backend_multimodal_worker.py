# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub worker for the backend-multimodal transport e2e test.

The worker replaces an engine entirely: it inspects the ``PreprocessedRequest``
the frontend delivered and reports what it saw. That is the whole claim under
test — the completions schema admits a custom-modality payload and the wire
carries it to the worker unchanged.

Two models are registered from one process:

``MODEL_NAME``
    Must receive ``backend_multi_modal_data`` byte-identical to
    :data:`EXPECTED_PAYLOAD`, and must not receive the frontend-owned
    ``multi_modal_data`` media map.

``PLAIN_MODEL_NAME``
    Must receive no ``backend_multi_modal_data`` at all, proving the field is
    absent rather than defaulted when the client omits it.

A mismatch is reported as a typed 422 whose message carries what actually
arrived, so a failure is diagnosable from the response body alone.
"""

from __future__ import annotations

import asyncio
import json

import uvloop

from dynamo.llm import ModelInput, ModelType, WorkerType, register_model
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_backend_multimodal_data import (
    ENDPOINT_PATH,
    EXPECTED_PAYLOAD,
    MISMATCH_STATUS,
    MODEL_NAME,
    PLAIN_ENDPOINT_PATH,
    PLAIN_MODEL_NAME,
    RESPONSE_TOKEN_IDS,
)
from tests.utils.constants import QWEN


class _StatusLikeError(Exception):
    """Duck-typed `.status` + `.message`, as `HttpStatusError` provides."""

    def __init__(self, status: int, message: str):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.message = message


def _mismatch(reason: str, received: object) -> _StatusLikeError:
    return _StatusLikeError(
        MISMATCH_STATUS,
        f"{reason}: {json.dumps(received, sort_keys=True, default=repr)}",
    )


async def generate(request, context):
    received = request.get("backend_multi_modal_data")
    if received != EXPECTED_PAYLOAD:
        raise _mismatch("backend_multi_modal_data did not survive the wire", received)
    # The backend payload must stay out of the frontend-owned media map, whose
    # entries the preprocessor materializes from URL and RDMA descriptors.
    if request.get("multi_modal_data") is not None:
        raise _mismatch(
            "frontend multi_modal_data was populated",
            request.get("multi_modal_data"),
        )
    yield {"token_ids": list(RESPONSE_TOKEN_IDS), "finish_reason": "stop"}


async def generate_plain(request, context):
    received = request.get("backend_multi_modal_data")
    if received is not None:
        raise _mismatch(
            "backend_multi_modal_data present without a request field", received
        )
    yield {"token_ids": list(RESPONSE_TOKEN_IDS), "finish_reason": "stop"}


async def main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "tcp")

    endpoint = runtime.endpoint(ENDPOINT_PATH)
    await register_model(
        ModelInput.Tokens,
        ModelType.Completions,
        endpoint,
        QWEN,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )

    plain_endpoint = runtime.endpoint(PLAIN_ENDPOINT_PATH)
    await register_model(
        ModelInput.Tokens,
        ModelType.Completions,
        plain_endpoint,
        QWEN,
        model_name=PLAIN_MODEL_NAME,
        worker_type=WorkerType.Aggregated,
    )

    await asyncio.gather(
        endpoint.serve_endpoint(generate),
        plain_endpoint.serve_endpoint(generate_plain),
    )


if __name__ == "__main__":
    uvloop.run(main())
