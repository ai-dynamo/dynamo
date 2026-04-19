#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

from collections.abc import AsyncGenerator
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from transformers import AutoTokenizer

from dynamo.frontend.prepost import _prepare_request
from dynamo.frontend.vllm_processor import VllmProcessor

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

# Needs vllm packages (gpu_1 container).  No need for parallel marker.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.pre_merge,
]

MODEL = "Qwen/Qwen3-0.6B"

TOOL_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "Hello"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                },
            },
        }
    ],
}


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


# ---------------------------------------------------------------------------
# _prepare_request: tool_choice=none tool-stripping
# ---------------------------------------------------------------------------


class TestPrepareRequestToolStripping:
    """Test that _prepare_request strips/keeps tools based on the flag."""

    def test_tool_choice_none_strips_tools_from_template(self, tokenizer):
        """When exclude flag is on and tool_choice=none, tools are excluded from template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert (
            chat_params.chat_template_kwargs["tools"] is None
        ), "tool_choice=none with exclude flag should strip tools from template"

    def test_tool_choice_none_keeps_tools_when_flag_off(self, tokenizer):
        """When exclude flag is off, tool_choice=none still includes tools in template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "none"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=False,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "tool_choice=none with flag off should keep tools in template"

    def test_tool_choice_auto_keeps_tools(self, tokenizer):
        """tool_choice=auto should always include tools regardless of flag."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "auto"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "tool_choice=auto should keep tools in template"

    def test_tool_choice_required_keeps_tools(self, tokenizer):
        """tool_choice=required should always include tools regardless of flag."""
        _, _, _, _, chat_params = _prepare_request(
            {**TOOL_REQUEST, "tool_choice": "required"},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        tools = chat_params.chat_template_kwargs["tools"]
        assert (
            tools is not None and len(tools) == 1
        ), "tool_choice=required should keep tools in template"

    def test_no_tools_in_request(self, tokenizer):
        """Request without tools should produce None tools in template kwargs."""
        _, _, _, _, chat_params = _prepare_request(
            {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]},
            tokenizer=tokenizer,
            tool_parser_class=None,
            exclude_tools_when_tool_choice_none=True,
        )
        assert (
            chat_params.chat_template_kwargs["tools"] is None
        ), "No tools in request should produce None tools in template"


async def _single_chunk_stream(
    chunk: dict[str, object],
) -> AsyncGenerator[dict[str, object], None]:
    yield chunk


@pytest.mark.asyncio
async def test_generate_and_stream_emits_nvext_routed_experts():
    processor = VllmProcessor.__new__(VllmProcessor)
    processor.router = SimpleNamespace(
        generate=AsyncMock(
            return_value=_single_chunk_stream(
                {
                    "token_ids": [11],
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                    "disaggregated_params": {
                        "routed_experts": "encoded-routed-experts"
                    },
                }
            )
        )
    )
    processor.is_kv_router = False
    processor.output_processor = SimpleNamespace(
        add_request=Mock(),
        process_outputs=Mock(
            return_value=SimpleNamespace(
                reqs_to_abort=[],
                request_outputs=[
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                index=0,
                                token_ids=[11],
                                text="hi",
                                finish_reason="stop",
                                logprobs=None,
                            )
                        ]
                    )
                ],
            )
        ),
        request_states={"req-1": object()},
        abort_requests=Mock(),
    )

    post = SimpleNamespace(
        process_output=Mock(
            return_value={
                "index": 0,
                "delta": {"role": "assistant", "content": "hi"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        )
    )
    vllm_preproc = SimpleNamespace(request_id="req-1")

    chunks = []
    async for chunk in processor._generate_and_stream(
        request_id="frontend-1",
        request={"model": "dummy-model"},
        dynamo_preproc={"model": "dummy-model"},
        tokens=[1, 2],
        vllm_preproc=vllm_preproc,
        post=post,
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert chunks[0]["id"] == "frontend-1"
    assert chunks[0]["model"] == "dummy-model"
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"] == [
        {
            "index": 0,
            "delta": {"role": "assistant", "content": "hi"},
            "finish_reason": "stop",
            "logprobs": None,
        }
    ]
    assert chunks[0]["usage"] == {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "total_tokens": 3,
    }
    assert chunks[0]["nvext"] == {"routed_experts": "encoded-routed-experts"}


@pytest.mark.asyncio
async def test_generate_and_stream_omits_nvext_when_routed_experts_absent():
    processor = VllmProcessor.__new__(VllmProcessor)
    processor.router = SimpleNamespace(
        generate=AsyncMock(
            return_value=_single_chunk_stream(
                {
                    "token_ids": [12],
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 1,
                        "total_tokens": 3,
                    },
                }
            )
        )
    )
    processor.is_kv_router = False
    processor.output_processor = SimpleNamespace(
        add_request=Mock(),
        process_outputs=Mock(
            return_value=SimpleNamespace(
                reqs_to_abort=[],
                request_outputs=[
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(
                                index=0,
                                token_ids=[12],
                                text="ok",
                                finish_reason="stop",
                                logprobs=None,
                            )
                        ]
                    )
                ],
            )
        ),
        request_states={"req-2": object()},
        abort_requests=Mock(),
    )
    post = SimpleNamespace(
        process_output=Mock(
            return_value={
                "index": 0,
                "delta": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
                "logprobs": None,
            }
        )
    )
    vllm_preproc = SimpleNamespace(request_id="req-2")

    chunks = []
    async for chunk in processor._generate_and_stream(
        request_id="frontend-2",
        request={"model": "dummy-model"},
        dynamo_preproc={"model": "dummy-model"},
        tokens=[1, 2],
        vllm_preproc=vllm_preproc,
        post=post,
    ):
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "nvext" not in chunks[0]
