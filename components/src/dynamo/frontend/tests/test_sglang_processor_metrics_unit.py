#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""SGLang processor tests with lightweight backend stubs."""

import asyncio
import builtins
import importlib
import sys
import types

import pytest
from _routed_engine_fakes import FakeRoutedEngine

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.core,
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


def _install_sglang_stubs(install_module, tokenizer_module):
    """Install lightweight backend modules with the selected tokenizer layout."""

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

    class _GptOssDetector:
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
        "sglang.srt.parser.conversation",
        chat_template_exists=lambda *_args, **_kwargs: False,
    )
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
        GptOssDetector=_GptOssDetector,
    )
    _install_module(install_module, "sglang.srt.utils")
    _install_module(
        install_module,
        tokenizer_module,
        get_tokenizer=lambda *args, **kwargs: None,
    )


class _PostProcessor:
    locally_finished = False
    has_pending_stop_text = False
    local_stop_reason = None

    def process_output(self, mapped_response):
        return {
            "index": 0,
            "delta": {"content": "x"},
            "finish_reason": mapped_response["finish_reason"],
        }


def _load_processor_module(
    module_stubs, tokenizer_module="sglang.srt.utils.hf_transformers"
):
    """Import a fresh processor against the selected SGLang module layout."""
    install_module, remove_module = module_stubs
    for name in (
        "sglang.srt.hf_transformers",
        "sglang.srt.utils.hf_transformers",
    ):
        remove_module(name)
    _install_sglang_stubs(install_module, tokenizer_module)
    _install_module(install_module, "dynamo._internal", ModelDeploymentCard=object)
    # Importing common.utils initializes native bindings absent from this fixture.
    _install_module(
        install_module,
        "dynamo.common.utils.input_params",
        resolve_thinking_token_budget=lambda *_args, **_kwargs: None,
    )
    _install_module(install_module, "dynamo.common.multimodal")
    _install_module(
        install_module,
        "dynamo.common.multimodal.cache_uuid",
        reject_unsupported_multimodal_uuids=lambda *_args, **_kwargs: None,
    )
    _install_module(
        install_module,
        "dynamo.frontend.sglang_prepost",
        ReasoningParser=object,
        SglangStreamingPostProcessor=object,
        ToolCallParserType=object,
        _client_wants_separate_reasoning=lambda *_args, **_kwargs: False,
        _effective_reasoning_parser_name=lambda *_args, **_kwargs: None,
        _get_history_tool_calls_count=lambda *_args, **_kwargs: 0,
        _guided_output_requires_reasoning=lambda *_args, **_kwargs: False,
        convert_tools=lambda *_args, **_kwargs: None,
        create_parsers=lambda *_args, **_kwargs: (None, None),
        detect_force_reasoning_from_template=lambda *_args, **_kwargs: False,
        preprocess_chat_request=lambda *_args, **_kwargs: None,
        resolve_skip_special_tokens=lambda *_args, **_kwargs: True,
    )
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
        HttpError=type("HttpError", (Exception,), {}),
        InvalidArgument=type("InvalidArgument", (Exception,), {}),
        RoutedEngineError=type("RoutedEngineError", (ValueError,), {}),
        Unknown=type("Unknown", (Exception,), {}),
        WorkerUnavailable=type("WorkerUnavailable", (Exception,), {}),
    )
    remove_module("dynamo.frontend.sglang_processor")
    return importlib.import_module("dynamo.frontend.sglang_processor")


def test_step_carries_tokens_and_the_engine_usage_report(module_stubs):
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
                _PostProcessor(),
            )
        ]

    # Dynamo builds usage and llm_metrics from the step's tokens and the
    # engine's report; the step itself carries neither.
    items = asyncio.run(collect())
    assert len(items) == 1
    step = items[0]
    assert step["_dynamo_parsed"] is True
    assert step["token_ids"] == [101, 102, 103]
    assert step["completion_usage"] == completion_usage
    assert step["finish_reason"] == "stop"
    assert "usage" not in step
    assert "llm_metrics" not in step


@pytest.mark.parametrize("stream_interval", [1, 2])
def test_engine_usage_reports_survive_flushes(module_stubs, stream_interval):
    module = _load_processor_module(module_stubs)
    processor = module.SglangProcessor(
        tokenizer=None,
        routed_engine=FakeRoutedEngine(
            items=[
                {
                    "token_ids": [101],
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 1,
                        "total_tokens": 4,
                        "prompt_tokens_details": {"cached_tokens": 0},
                    },
                },
                {
                    "token_ids": [102],
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 2,
                        "total_tokens": 5,
                        "prompt_tokens_details": {"cached_tokens": 2},
                    },
                },
                {
                    "token_ids": [103],
                    "completion_usage": {
                        "prompt_tokens": 3,
                        "completion_tokens": 3,
                        "total_tokens": 6,
                    },
                },
                {"token_ids": [], "finish_reason": "stop"},
            ]
        ),
        tool_call_parser_name=None,
        reasoning_parser_name=None,
        eos_token_ids=None,
    )
    processor.stream_interval = stream_interval

    async def collect():
        return [
            item
            async for item in processor._generate_and_stream(
                "req-sparse", {"model": "test-model"}, {}, _PostProcessor()
            )
        ]

    # A report that arrives while tokens are batched goes out with their step.
    items = asyncio.run(collect())
    batches = (
        [[101], [102], [103], []] if stream_interval == 1 else [[101], [102, 103], []]
    )
    assert [item["token_ids"] for item in items] == batches
    usages = [item.get("completion_usage") or {} for item in items]
    cached = [usage.get("prompt_tokens_details") for usage in usages]
    assert cached[:2] == [{"cached_tokens": 0}, {"cached_tokens": 2}]
    assert not any(cached[2:])
    if stream_interval == 2:
        # Two reports in one batch: the later totals, with the cache count only
        # the earlier one had.
        assert usages[1]["completion_tokens"] == 3


@pytest.mark.parametrize(
    "tokenizer_module",
    [
        "sglang.srt.hf_transformers",
        "sglang.srt.utils.hf_transformers",
    ],
)
def test_tokenizer_import_layouts(module_stubs, tokenizer_module):
    """Resolve the tokenizer from current sources and the pinned releases."""
    module = _load_processor_module(module_stubs, tokenizer_module)
    assert module.get_tokenizer is sys.modules[tokenizer_module].get_tokenizer


@pytest.mark.parametrize(
    "tokenizer_module",
    ["sglang.srt.hf_transformers", "sglang.srt.utils.hf_transformers"],
)
def test_tokenizer_import_preserves_dependency_error(
    module_stubs, monkeypatch, tokenizer_module
):
    """Preserve missing dependencies instead of treating them as a layout move."""
    original_import = builtins.__import__
    error = ModuleNotFoundError(
        "No module named 'tokenizer_dependency'", name="tokenizer_dependency"
    )

    def import_with_missing_dependency(name, *args, **kwargs):
        """Fail inside the selected tokenizer module while importing others normally."""
        if name == tokenizer_module:
            raise error
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_with_missing_dependency)
    with pytest.raises(ModuleNotFoundError) as caught:
        _load_processor_module(module_stubs)
    assert caught.value is error
