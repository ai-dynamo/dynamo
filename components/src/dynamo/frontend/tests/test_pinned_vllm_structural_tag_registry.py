# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the real vLLM structural-tag behavior used by Dynamo's auto tool path."""

import pytest

try:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionToolsParam,
    )
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag
except ImportError:
    pytest.skip("requires the vLLM structural-tag registry", allow_module_level=True)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_pinned_vllm_registry_auto_floor():
    tool = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            },
            "strict": False,
        },
    }
    non_strict = ChatCompletionToolsParam.model_validate(tool)
    assert (
        get_model_structural_tag(
            model="kimi_k3",
            tools=[non_strict],
            tool_choice="auto",
            reasoning=False,
        )
        is None
    )

    tool["function"]["strict"] = True
    strict = ChatCompletionToolsParam.model_validate(tool)
    assert (
        get_model_structural_tag(
            model="kimi_k3",
            tools=[strict],
            tool_choice="auto",
            reasoning=False,
        )
        is not None
    )
