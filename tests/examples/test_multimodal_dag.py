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

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import pytest
import torch

from dynamo.vllm.multimodal_utils.external_qwen_artifact import ExternalQwenArtifact
from examples.custom_backend.multimodal_dag.classifier_worker import classify_artifact
from examples.custom_backend.multimodal_dag.orchestrator_worker import (
    OrchestratorHandler,
)
from examples.custom_backend.multimodal_dag.protocol import validate_chat_request
from examples.custom_backend.multimodal_dag.vision_encoder_worker import (
    render_unexpanded_prompt_token_ids,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.multimodal,
]

_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
_BASE_REQUEST = {
    "model": "multimodal-dag",
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
            ],
        }
    ],
    "stream": False,
    "n": 1,
    "max_completion_tokens": 8,
    "temperature": 0.0,
    "top_p": 0.9,
    "stop": ["END"],
    "seed": 3,
}


def _artifact(values: list[float] | None = None) -> dict[str, Any]:
    image_embeds = torch.tensor(
        [values or [3.0, 1.0, 0.0, 0.0]],
        dtype=torch.bfloat16,
    )
    return ExternalQwenArtifact.create(
        model=_MODEL,
        prompt_token_ids=[100, 101, 102],
        image_embeds=image_embeds,
        image_grid_thw=[[1, 2, 2]],
    ).to_dict()


def test_request_parser_accepts_supported_subset() -> None:
    parsed = validate_chat_request(_BASE_REQUEST)

    assert parsed.image_url.startswith("data:image/png")
    assert parsed.max_tokens == 8
    assert parsed.temperature == 0.0
    assert parsed.top_p == 0.9
    assert parsed.stop == ["END"]
    assert parsed.seed == 3
    assert parsed.processor_messages[0]["content"][1] == {
        "type": "image",
        "image": "data:image/png;base64,AAAA",
    }


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ([{"type": "text", "text": "text only"}], "exactly one image_url"),
        (
            [
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                }
            ],
            "at least one text",
        ),
        (
            [
                {"type": "text", "text": "two images"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,AAAA"},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,BBBB"},
                },
            ],
            "exactly one image_url",
        ),
    ],
)
def test_request_parser_rejects_missing_or_multiple_inputs(content, match) -> None:
    request = copy.deepcopy(_BASE_REQUEST)
    request["messages"][0]["content"] = content

    with pytest.raises(ValueError, match=match):
        validate_chat_request(request)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("stream", True, "stream=false"),
        ("n", 2, "n=1"),
        ("tools", [{"type": "function"}], "unsupported request fields"),
        ("logprobs", True, "unsupported request fields"),
    ],
)
def test_request_parser_rejects_unsupported_options(field, value, match) -> None:
    request = copy.deepcopy(_BASE_REQUEST)
    request[field] = value

    with pytest.raises(ValueError, match=match):
        validate_chat_request(request)


def test_dummy_classifier_is_deterministic_and_consumes_tensor() -> None:
    class_zero = _artifact([5.0, 0.0, 0.0, 0.0])
    class_one = _artifact([0.0, 5.0, 0.0, 0.0])

    first = classify_artifact(class_zero)
    assert classify_artifact(class_zero) == first
    assert first["label"] == "class_0"
    assert first["embedding_shape"] == [1, 4]
    assert classify_artifact(class_one)["label"] == "class_1"


def test_prompt_rendering_keeps_one_unexpanded_vision_triple() -> None:
    class Tokenizer:
        @staticmethod
        def encode(prompt: str, *, add_special_tokens: bool) -> list[int]:
            assert not add_special_tokens
            assert prompt == "<vision_start><image_pad><vision_end>"
            return [100, 101, 102]

    class Processor:
        tokenizer = Tokenizer()

        @staticmethod
        def apply_chat_template(
            messages: list[dict[str, Any]],
            *,
            tokenize: bool,
            add_generation_prompt: bool,
        ) -> str:
            assert messages
            assert not tokenize
            assert add_generation_prompt
            return "<vision_start><image_pad><vision_end>"

    token_ids = render_unexpanded_prompt_token_ids(
        Processor(),
        [{"role": "user", "content": [{"type": "image", "image": "data:"}]}],
    )

    assert token_ids == [100, 101, 102]


class _Response:
    def __init__(self, payload: Mapping[str, Any]) -> None:
        self._payload = payload

    def data(self) -> Mapping[str, Any]:
        return self._payload


class _FakeClient:
    def __init__(
        self,
        callback: Callable[
            [dict[str, Any], Any],
            Awaitable[list[Mapping[str, Any]]],
        ],
    ) -> None:
        self._callback = callback
        self.calls: list[tuple[dict[str, Any], Any]] = []

    async def generate(self, request: dict[str, Any], context: Any):
        self.calls.append((request, context))
        payloads = await self._callback(request, context)

        async def responses():
            for payload in payloads:
                yield _Response(payload)

        return responses()


class _Tokenizer:
    @staticmethod
    def decode(token_ids: list[int], *, skip_special_tokens: bool) -> str:
        assert skip_special_tokens
        assert token_ids == [11, 12]
        return "model output"


@pytest.mark.asyncio
async def test_orchestrator_encodes_then_starts_parallel_branches() -> None:
    sequence: list[str] = []
    branches_started = asyncio.Event()
    context = object()
    artifact = _artifact()

    async def encoder_callback(request, received_context):
        assert received_context is context
        sequence.append("encoder")
        return [artifact]

    async def classifier_callback(request, received_context):
        assert received_context is context
        assert sequence == ["encoder"] or sequence == ["encoder", "vllm"]
        sequence.append("classifier")
        if "vllm" in sequence:
            branches_started.set()
        await branches_started.wait()
        return [classify_artifact(request)]

    async def vllm_callback(request, received_context):
        assert received_context is context
        assert sequence == ["encoder"] or sequence == ["encoder", "classifier"]
        assert request["external_mm_data"] == artifact
        sequence.append("vllm")
        if "classifier" in sequence:
            branches_started.set()
        await branches_started.wait()
        return [
            {"index": 0, "token_ids": [11]},
            {
                "index": 0,
                "token_ids": [12],
                "finish_reason": "stop",
                "completion_usage": {
                    "prompt_tokens": 3,
                    "completion_tokens": 2,
                    "total_tokens": 5,
                },
            },
        ]

    encoder = _FakeClient(encoder_callback)
    classifier = _FakeClient(classifier_callback)
    vllm = _FakeClient(vllm_callback)
    handler = OrchestratorHandler(
        backend_model=_MODEL,
        tokenizer=_Tokenizer(),
        eos_token_ids=[102],
        encoder_client=encoder,
        classifier_client=classifier,
        vllm_client=vllm,
    )

    chunks = [
        chunk async for chunk in handler.generate(copy.deepcopy(_BASE_REQUEST), context)
    ]

    assert sequence[0] == "encoder"
    assert set(sequence[1:]) == {"classifier", "vllm"}
    assert len(chunks) == 2
    assert chunks[0]["choices"][0]["delta"]["content"] == "model output"
    assert chunks[1]["choices"][0]["finish_reason"] == "stop"
    assert chunks[1]["nvext"]["classifier"]["embedding_shape"] == [1, 4]
    assert chunks[1]["usage"]["total_tokens"] == 5


@pytest.mark.asyncio
async def test_orchestrator_cancels_sibling_and_returns_no_partial_result() -> None:
    context = object()
    vllm_started = asyncio.Event()
    vllm_cancelled = asyncio.Event()

    async def encoder_callback(request, received_context):
        return [_artifact()]

    async def classifier_callback(request, received_context):
        await vllm_started.wait()
        raise RuntimeError("classifier failed")

    async def vllm_callback(request, received_context):
        vllm_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            vllm_cancelled.set()
            raise
        return []

    handler = OrchestratorHandler(
        backend_model=_MODEL,
        tokenizer=_Tokenizer(),
        eos_token_ids=[102],
        encoder_client=_FakeClient(encoder_callback),
        classifier_client=_FakeClient(classifier_callback),
        vllm_client=_FakeClient(vllm_callback),
    )
    chunks: list[dict[str, Any]] = []

    with pytest.raises(RuntimeError, match="classifier failed"):
        async for chunk in handler.generate(copy.deepcopy(_BASE_REQUEST), context):
            chunks.append(chunk)

    assert chunks == []
    assert vllm_cancelled.is_set()
