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

_MISSING = object()


@pytest.fixture
def module_stubs():
    original_modules = {}

    def remember(name: str):
        if name not in original_modules:
            original_modules[name] = sys.modules.get(name, _MISSING)

    def install_module(name: str, **attrs):
        remember(name)
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
        return module

    def remove_module(name: str):
        remember(name)
        sys.modules.pop(name, None)

    yield install_module, remove_module

    for name, module in original_modules.items():
        if module is _MISSING:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _install_module(install_module, name: str, **attrs):
    return install_module(name, **attrs)


def _install_sglang_stubs(install_module):
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

    _install_module(install_module, "sglang")
    _install_module(install_module, "sglang.srt")
    _install_module(install_module, "sglang.srt.entrypoints")
    _install_module(install_module, "sglang.srt.entrypoints.openai")
    _install_module(
        install_module,
        "sglang.srt.entrypoints.openai.protocol",
        Function=_Function,
        Tool=_Tool,
        ToolChoice=_ToolChoice,
        ToolChoiceFuncName=_ToolChoiceFuncName,
    )
    _install_module(install_module, "sglang.srt.function_call")
    _install_module(
        install_module,
        "sglang.srt.function_call.core_types",
        ToolCallItem=_ToolCallItem,
    )
    _install_module(
        install_module,
        "sglang.srt.function_call.function_call_parser",
        FunctionCallParser=_FunctionCallParser,
    )
    _install_module(
        install_module,
        "sglang.srt.function_call.json_array_parser",
        JsonArrayParser=_JsonArrayParser,
    )
    _install_module(
        install_module,
        "sglang.srt.function_call.utils",
        get_json_schema_constraint=lambda *args, **kwargs: None,
    )
    _install_module(install_module, "sglang.srt.parser")
    _install_module(
        install_module,
        "sglang.srt.parser.jinja_template_utils",
        detect_jinja_template_content_format=lambda *args, **kwargs: "string",
        process_content_for_template_format=lambda content, *_args, **_kwargs: content,
    )
    _install_module(
        install_module,
        "sglang.srt.parser.reasoning_parser",
        ReasoningParser=_ReasoningParser,
    )
    _install_module(install_module, "sglang.srt.utils")
    _install_module(
        install_module,
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


def _load_processor_module(module_stubs):
    install_module, remove_module = module_stubs
    _install_sglang_stubs(install_module)
    _install_module(install_module, "dynamo._internal", ModelDeploymentCard=object)
    _install_module(
        install_module,
        "dynamo.frontend.frontend_args",
        FrontendConfig=object,
    )
    _install_module(
        install_module,
        "dynamo.llm",
        ModelCardInstanceId=object,
        PythonAsyncEngine=object,
        RoutedEngine=object,
    )
    _install_module(
        install_module,
        "dynamo.llm.exceptions",
        InvalidArgument=type("InvalidArgument", (Exception,), {}),
        Unknown=type("Unknown", (Exception,), {}),
    )
    remove_module("dynamo.frontend.sglang_processor")
    return importlib.import_module("dynamo.frontend.sglang_processor")


def test_stream_emits_llm_metrics_annotation(module_stubs):
    module = _load_processor_module(module_stubs)
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
        eos_token_ids=None,
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
    # Zero counts are omitted (text-only request), mirroring the Rust skip-zero behavior.
    assert metrics == {
        "input_tokens": 10,
        "output_tokens": 3,
        "chunk_tokens": 3,
        "cached_tokens": 4,
    }


def test_stream_emits_multimodal_counts(module_stubs):
    # Mixed-media request: assert the SGLang extraction branch reports nonzero
    # counts (2 images + 1 video), mirroring the vLLM processor test so the
    # extract_mm_urls wiring cannot silently regress to zero.
    module = _load_processor_module(module_stubs)
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101, 102, 103],
                    "finish_reason": "stop",
                    "completion_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 3,
                        "total_tokens": 13,
                    },
                }
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )

    request = {
        "model": "test-model",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "compare"},
                    {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                    {"type": "image_url", "image_url": {"url": "http://x/b.png"}},
                    {"type": "video_url", "video_url": {"url": "http://x/c.mp4"}},
                ],
            }
        ],
    }

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-mm-counts",
                request,
                {},
                list(range(10)),
                _PostProcessor(),
            )
        ]

    items = asyncio.run(collect())
    metric_items = [item for item in items if item.get("event") == "llm_metrics"]
    assert len(metric_items) == 1
    metrics = json.loads(metric_items[0]["comment"][0])
    assert metrics["image_count"] == 2
    assert metrics["video_count"] == 1
    # audio has zero parts, so the key is omitted from the emitted metrics.
    assert metrics.get("audio_count") is None
