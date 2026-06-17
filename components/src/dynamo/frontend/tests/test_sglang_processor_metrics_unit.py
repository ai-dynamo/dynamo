#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Metrics-focused SGLang processor tests with lightweight SGLang stubs."""

import asyncio
import importlib
import json
import sys
import types

import pytest
from _routed_engine_fakes import FakeRoutedEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _install_module(name: str, **attrs):
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module
    return module


def _install_sglang_stubs():
    class _Function:
        pass

    class _Tool:
        pass

    class _ToolChoice:
        pass

    class _ToolChoiceFuncName:
        pass

    class _ToolCallItem:
        pass

    class _FunctionCallParser:
        pass

    class _JsonArrayParser:
        pass

    class _ReasoningParser:
        pass

    _install_module("sglang")
    _install_module("sglang.srt")
    _install_module("sglang.srt.entrypoints")
    _install_module("sglang.srt.entrypoints.openai")
    _install_module(
        "sglang.srt.entrypoints.openai.protocol",
        Function=_Function,
        Tool=_Tool,
        ToolChoice=_ToolChoice,
        ToolChoiceFuncName=_ToolChoiceFuncName,
    )
    _install_module("sglang.srt.function_call")
    _install_module("sglang.srt.function_call.core_types", ToolCallItem=_ToolCallItem)
    _install_module(
        "sglang.srt.function_call.function_call_parser",
        FunctionCallParser=_FunctionCallParser,
    )
    _install_module(
        "sglang.srt.function_call.json_array_parser",
        JsonArrayParser=_JsonArrayParser,
    )
    _install_module(
        "sglang.srt.function_call.utils",
        get_json_schema_constraint=lambda *args, **kwargs: None,
    )
    _install_module("sglang.srt.parser")
    _install_module(
        "sglang.srt.parser.jinja_template_utils",
        detect_jinja_template_content_format=lambda *args, **kwargs: "string",
        process_content_for_template_format=lambda content, *_args, **_kwargs: content,
    )
    _install_module(
        "sglang.srt.parser.reasoning_parser",
        ReasoningParser=_ReasoningParser,
    )
    _install_module("sglang.srt.utils")
    _install_module(
        "sglang.srt.utils.hf_transformers_utils",
        get_tokenizer=lambda *args, **kwargs: None,
    )


class _PostProcessor:
    def process_output(self, mapped_response):
        return {
            "index": 0,
            "delta": {"content": "x"},
            "finish_reason": mapped_response["finish_reason"],
        }


def _load_processor_module():
    _install_sglang_stubs()
    _install_module("dynamo._internal", ModelDeploymentCard=object)
    _install_module("dynamo.frontend.frontend_args", FrontendConfig=object)
    _install_module(
        "dynamo.llm",
        ModelCardInstanceId=object,
        PythonAsyncEngine=object,
        RoutedEngine=object,
    )
    _install_module(
        "dynamo.llm.exceptions",
        InvalidArgument=type("InvalidArgument", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
    )
    return importlib.import_module("dynamo.frontend.sglang_processor")


def test_stream_emits_llm_metrics_annotation():
    module = _load_processor_module()
    completion_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 3,
        "total_tokens": 13,
        "prompt_tokens_details": {"cached_tokens": 4},
    }
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101, 102, 103],
                    "finish_reason": "stop",
                    "completion_usage": completion_usage,
                }
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_id=None,
    )

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-metrics",
                {"model": "test-model"},
                {},
                list(range(10)),
                _PostProcessor(),
            )
        ]

    items = asyncio.run(collect())
    metric_items = [item for item in items if item.get("event") == "llm_metrics"]

    assert len(metric_items) == 1
    envelope = metric_items[0]
    assert envelope["_dynamo_annotated"] is True
    assert envelope["data"]["usage"] == completion_usage
    metrics = json.loads(envelope["comment"][0])
    assert metrics == {
        "input_tokens": 10,
        "output_tokens": 3,
        "chunk_tokens": 3,
        "cached_tokens": 4,
    }
