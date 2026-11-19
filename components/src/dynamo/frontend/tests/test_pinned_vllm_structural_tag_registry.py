# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pin the real vLLM structural-tag behavior used by Dynamo's tool path."""

import json
from types import SimpleNamespace

import pytest

try:
    import xgrammar as xgr
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )
    from vllm.tool_parsers.structural_tag_registry import get_model_structural_tag

except ImportError:
    pytest.skip("requires the vLLM structural-tag registry", allow_module_level=True)

import dynamo.frontend.prepost as prepost_module
from dynamo.frontend.prepost import build_tool_call_guided_decoding

pytestmark = [
    pytest.mark.core,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.timeout(60),
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


@pytest.fixture(scope="module")
def tool_grammar_compiler():
    """Compile tool grammars without a model or a tokenizer download."""
    return xgr.GrammarCompiler(
        xgr.TokenizerInfo(["x"], vocab_type=xgr.VocabType.RAW, stop_token_ids=[0]),
        max_threads=1,
    )


@pytest.mark.parametrize(
    "model_format, call, double_call, missing_argument",
    [
        (
            "qwen_3",
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>\n'
            '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Rome"}}\n</tool_call>',
            '<tool_call>\n{"name": "get_weather", "arguments": {}}\n</tool_call>',
        ),
        (
            "qwen_3_coder",
            "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=get_weather>\n<parameter=city>Paris</parameter>\n</function>\n</tool_call>\n"
            "<tool_call>\n<function=get_weather>\n<parameter=city>Rome</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=get_weather>\n</function>\n</tool_call>",
        ),
        (
            "kimi",
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|><|tool_calls_section_end|>',
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
            '<|tool_call_argument_begin|>{"city": "Paris"}<|tool_call_end|>'
            "<|tool_call_begin|>functions.get_weather:1"
            '<|tool_call_argument_begin|>{"city": "Rome"}<|tool_call_end|><|tool_calls_section_end|>',
            "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0"
            "<|tool_call_argument_begin|>{}<|tool_call_end|><|tool_calls_section_end|>",
        ),
    ],
    ids=["qwen_json", "qwen_xml", "kimi"],
)
@pytest.mark.parametrize("tool_choice", ["auto", "required", "named"])
@pytest.mark.parametrize("parallel", [True, False])
@pytest.mark.parametrize(
    "parser_path", ["registry", "typed_hook", "dict_hook", "typed_hook_without_limiter"]
)
def test_dynamo_structural_tag_limits_call_count(
    tool_grammar_compiler,
    monkeypatch,
    parser_path,
    model_format,
    call,
    double_call,
    missing_argument,
    tool_choice,
    parallel,
):
    """The transmitted grammar enforces call counts and required arguments."""
    request = ChatCompletionRequest(
        model="unused",
        messages=[{"role": "user", "content": "Weather in Paris?"}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "strict": True,
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        tool_choice=(
            {"type": "function", "function": {"name": "get_weather"}}
            if tool_choice == "named"
            else tool_choice
        ),
        parallel_tool_calls=parallel,
    )
    if parser_path == "typed_hook_without_limiter":
        monkeypatch.setattr(prepost_module, "limit_to_single_tool_call", None)
    parser = SimpleNamespace(structural_tag_model=model_format)
    if parser_path != "registry":

        def get_structural_tag(hook_request):
            tag = get_model_structural_tag(
                model=model_format,
                tools=hook_request.tools,
                tool_choice=hook_request.tool_choice,
                reasoning=False,
            )
            return tag.model_dump() if parser_path == "dict_hook" else tag

        parser = SimpleNamespace(get_structural_tag=get_structural_tag)
        monkeypatch.setattr(prepost_module, "get_model_structural_tag", None)
    guided = build_tool_call_guided_decoding(
        request,
        parser,
        structural_tag_mode="on",
        structural_tag_scope="always",
    )
    assert guided is not None
    compiled = tool_grammar_compiler.compile_structural_tag(
        json.dumps(guided["structural_tag"])
    )
    single = xgr.GrammarMatcher(compiled)
    assert single.accept_string(call)
    assert single.accept_token(0)
    assert not xgr.GrammarMatcher(compiled).accept_string(missing_argument)
    double = xgr.GrammarMatcher(compiled)
    expect_parallel = parallel and tool_choice != "named"
    assert double.accept_string(double_call) is expect_parallel
    if expect_parallel:
        assert double.accept_token(0)
