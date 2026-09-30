# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stub worker asserting JSON and media share the existing multi_modal_data wire field."""

from __future__ import annotations

import asyncio
import json

import uvloop

from dynamo.llm import (
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    register_model,
)
from dynamo.runtime import DistributedRuntime
from tests.frontend.test_json_multimodal_content import (
    ENDPOINT_PATH,
    EXPECTED_PAYLOAD,
    IMAGE_URL,
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
    received = request.get("multi_modal_data")
    expected = {kind: [{"Json": value}] for kind, value in EXPECTED_PAYLOAD.items()}
    if request.get("token_ids") == [1, 2, 4]:
        expected["image_url"] = [{"Url": IMAGE_URL}]
    if received != expected:
        raise _mismatch("multi_modal_data did not survive the wire", received)
    if "data_base64" in json.dumps((request.get("extra_args") or {}).get("messages")):
        raise _mismatch(
            "payload duplicated in auxiliary messages", request["extra_args"]
        )
    prompt = (request.get("extra_args") or {}).get("formatted_prompt")
    if (
        request.get("token_ids") not in ([1, 2, 3], [1, 2, 4])
        and prompt is not None
        and (
            "Predict this" not in prompt
            or EXPECTED_PAYLOAD["custom_input"]["data_base64"] in prompt
        )
    ):
        raise _mismatch(
            "chat template did not preserve text and exclude payload", prompt
        )
    yield {"token_ids": list(RESPONSE_TOKEN_IDS), "finish_reason": "stop"}


async def generate_plain(request, context):
    received = request.get("multi_modal_data")
    if received is not None:
        raise _mismatch("multi_modal_data present without content", received)
    yield {"token_ids": list(RESPONSE_TOKEN_IDS), "finish_reason": "stop"}


async def main():
    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "tcp")

    endpoint = runtime.endpoint(ENDPOINT_PATH)
    runtime_config = ModelRuntimeConfig()
    runtime_config.set_engine_specific("json_multimodal", json.dumps(True))
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
        endpoint,
        QWEN,
        model_name=MODEL_NAME,
        worker_type=WorkerType.Aggregated,
        runtime_config=runtime_config,
    )

    plain_endpoint = runtime.endpoint(PLAIN_ENDPOINT_PATH)
    await register_model(
        ModelInput.Tokens,
        ModelType.Chat,
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
