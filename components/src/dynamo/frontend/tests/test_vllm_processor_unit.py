#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import importlib.util
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from _routed_engine_fakes import FakeRoutedEngine as _FakeRoutedEngine
from _thinking_parity import THINKING_PARITY_CASES
from _tool_guidance_parity import (
    TOOL_GUIDANCE_PARITY_CASES,
    assistant_response_format,
    classify_guidance_source,
    parity_tool,
    tool_choice_value,
)
from transformers import AutoTokenizer
from vllm.sampling_params import SamplingParams, StructuredOutputsParams

from dynamo.frontend import prepost as prepost_module
from dynamo.frontend.prepost import (
    StreamingPostProcessor,
    _prepare_request,
    build_tool_call_guided_decoding,
)
from dynamo.llm.exceptions import InvalidArgument

# NOTE: dynamo.frontend.vllm_processor is imported lazily inside the tests that
# need it (and via the vllm_processor_module fixture). Importing it at module
# top level would run its `from vllm.tasks import ...` /
# `from vllm.v1.engine.parallel_sampling import ...` imports during pytest
# collection, which breaks the pytest-marker-report pre-commit hook (its vllm
# stub list does not cover those submodules).

HAS_QWEN3_TOOL_PARSER = (
    importlib.util.find_spec("vllm.tool_parsers.qwen3_engine_tool_parser") is not None
    or importlib.util.find_spec("vllm.tool_parsers.qwen3coder_tool_parser") is not None
)


def _resolve_qwen3_tool_parser_class():
    try:
        from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

        return Qwen3EngineToolParser
    except ImportError:
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

        return Qwen3CoderToolParser


# Needs vllm packages, but never touches a GPU.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    # This file builds a real tokenizer. The marker declares it in the session
    # predownload manifest (tests/conftest.py), which is what keeps it fetchable
    # on a lane that predownloads and flips HF_HUB_OFFLINE; the CPU lane has no
    # predownload consumer today, so there it is fetched live.
    pytest.mark.model("Qwen/Qwen3-0.6B"),
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
    pytest.mark.skipif(
        not HAS_QWEN3_TOOL_PARSER,
        reason="requires vllm qwen3 tool parser",
    ),
]

MODEL = "Qwen/Qwen3-0.6B"
_DEFAULT_MM_DATA = object()

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


class TestDynamoJsonToolCallFallback:
    """Dynamo's forced-choice JSON fallback must emit OpenAI tool calls."""

    def _post_processor(
        self,
        tokenizer,
        *,
        tool_choice,
        stream_response,
        parallel_tool_calls=None,
        tool_parameters=None,
    ):
        request = json.loads(json.dumps(TOOL_REQUEST))
        request["tool_choice"] = tool_choice
        if parallel_tool_calls is not None:
            request["parallel_tool_calls"] = parallel_tool_calls
        if tool_parameters is not None:
            request["tools"][0]["function"]["parameters"] = tool_parameters
        request, _, _, _, _ = _prepare_request(
            request,
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        return StreamingPostProcessor(
            tokenizer=tokenizer,
            request_for_sampling=request,
            sampling_params=SamplingParams(),
            prompt_token_ids=[],
            tool_parser=None,
            reasoning_parser_class=None,
            chat_template_kwargs={},
            stream_response=stream_response,
            uses_dynamo_json_tool_call_fallback=True,
        )

    def test_streaming_required_choice_converts_json_to_tool_calls(self, tokenizer):
        post = self._post_processor(
            tokenizer, tool_choice="required", stream_response=True
        )

        assert (
            post.process_output(
                SimpleNamespace(
                    index=0,
                    text='[{"name":"get_weather","parameters":',
                    token_ids=[],
                    finish_reason=None,
                    logprobs=None,
                )
            )
            is None
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text='{"city":"Paris"}}]',
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice["finish_reason"] == "tool_calls"
        assert "content" not in choice["delta"]
        tool_call = choice["delta"]["tool_calls"][0]
        assert tool_call["function"] == {
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }

    def test_non_streaming_named_choice_converts_json_to_tool_calls(self, tokenizer):
        post = self._post_processor(
            tokenizer,
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
            stream_response=False,
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text='{"city":"Paris"}',
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice["finish_reason"] == "tool_calls"
        assert "content" not in choice["delta"]
        assert choice["delta"]["tool_calls"][0]["function"] == {
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }

    def test_invalid_fallback_output_is_returned_as_content(self, tokenizer):
        post = self._post_processor(
            tokenizer, tool_choice="required", stream_response=True
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text='[{"name":"get_weather","parameters":',
                token_ids=[],
                finish_reason="length",
                logprobs=None,
            )
        )

        assert choice["finish_reason"] == "length"
        assert choice["delta"] == {
            "role": "assistant",
            "content": '[{"name":"get_weather","parameters":',
        }

    def test_multiple_fallback_tool_calls_respect_parallel_setting(self, tokenizer):
        post = self._post_processor(
            tokenizer,
            tool_choice="required",
            stream_response=False,
            parallel_tool_calls=False,
        )
        text = (
            '[{"name":"get_weather","parameters":{"city":"Paris"}},'
            '{"name":"get_weather","parameters":{"city":"London"}}]'
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text=text,
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice["finish_reason"] == "stop"
        assert choice["delta"] == {"role": "assistant", "content": text}

    def test_non_finite_fallback_arguments_are_returned_as_content(self, tokenizer):
        post = self._post_processor(
            tokenizer, tool_choice="required", stream_response=False
        )
        text = '[{"name":"get_weather","parameters":{"temperature":NaN}}]'

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text=text,
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice is not None
        assert choice["finish_reason"] == "stop"
        assert choice["delta"] == {"role": "assistant", "content": text}

    def test_invalid_fallback_does_not_leave_partial_tool_state(self, tokenizer):
        post = self._post_processor(
            tokenizer, tool_choice="required", stream_response=False
        )
        text = (
            '[{"name":"get_weather","parameters":{"city":"Paris"}},'
            '{"name":"unknown","parameters":{}}]'
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text=text,
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice is not None
        assert choice["finish_reason"] == "stop"
        assert choice["delta"] == {"role": "assistant", "content": text}
        assert post.in_progress_tool_calls == {}

    def test_named_choice_accepts_array_arguments(self, tokenizer):
        post = self._post_processor(
            tokenizer,
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            },
            stream_response=False,
            tool_parameters={"type": "array", "items": {"type": "string"}},
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text='["Paris","Seoul"]',
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice is not None
        assert choice["delta"]["tool_calls"][0]["function"]["arguments"] == (
            '["Paris","Seoul"]'
        )

    def test_required_choice_accepts_null_arguments(self, tokenizer):
        post = self._post_processor(
            tokenizer,
            tool_choice="required",
            stream_response=False,
            tool_parameters={"type": "null"},
        )

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text='[{"name":"get_weather","parameters":null}]',
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice is not None
        assert choice["delta"]["tool_calls"][0]["function"]["arguments"] == "null"

    def test_missing_parameters_are_returned_as_content(self, tokenizer):
        post = self._post_processor(
            tokenizer, tool_choice="required", stream_response=False
        )
        text = '[{"name":"get_weather"}]'

        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text=text,
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice is not None
        assert choice["delta"] == {"role": "assistant", "content": text}

    def test_streaming_choices_keep_independent_json_buffers(self, tokenizer):
        post = self._post_processor(
            tokenizer, tool_choice="required", stream_response=True
        )

        for index in (0, 1):
            assert (
                post.process_output(
                    SimpleNamespace(
                        index=index,
                        text='[{"name":"get_weather","parameters":{"city":"',
                        token_ids=[],
                        finish_reason=None,
                        logprobs=None,
                    )
                )
                is None
            )

        choices = [
            post.process_output(
                SimpleNamespace(
                    index=index,
                    text=f'{city}"}}}}]',
                    token_ids=[],
                    finish_reason="stop",
                    logprobs=None,
                )
            )
            for index, city in ((1, "Seoul"), (0, "Paris"))
        ]

        for choice, (index, city) in zip(choices, ((1, "Seoul"), (0, "Paris"))):
            assert choice is not None
            assert choice["index"] == index
            assert choice["finish_reason"] == "tool_calls"
            assert choice["delta"]["tool_calls"][0]["function"]["arguments"] == (
                f'{{"city":"{city}"}}'
            )


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(MODEL)


# ---------------------------------------------------------------------------
# _prepare_request: tool_choice=none tool-stripping
# ---------------------------------------------------------------------------


class TestPrepareRequestToolStripping:  # FRONTEND.1 + FRONTEND.3 — tool stripping when tool_choice=none on chat-template input
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


class TestChatTemplateArgsPassthrough:
    """Per-request chat template kwargs must survive into the rendered template.

    pythonize serializes the request field under its Rust name
    ``chat_template_args`` (serde ``alias`` is deserialize-only), so the vLLM
    processor must read that key, not only vLLM's native ``chat_template_kwargs``.
    """

    def test_chat_template_args_reaches_template(self, tokenizer):
        """Kwargs keyed as chat_template_args (the pythonize'd key) reach the template."""
        _, _, _, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_args": {"enable_thinking": False},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        assert (
            chat_params.chat_template_kwargs.get("enable_thinking") is False
        ), "chat_template_args must be forwarded to the chat template"

    def test_chat_template_kwargs_native_key_still_works(self, tokenizer):
        """The vLLM-native chat_template_kwargs key keeps working."""
        _, _, _, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        assert (
            chat_params.chat_template_kwargs.get("enable_thinking") is False
        ), "native chat_template_kwargs must be forwarded to the chat template"

    def test_nested_reasoning_effort_is_not_clobbered(self, tokenizer):
        """A reasoning_effort nested in template kwargs survives the top-level default."""
        _, _, _, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_args": {"reasoning_effort": "high"},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        assert (
            chat_params.chat_template_kwargs.get("reasoning_effort") == "high"
        ), "nested reasoning_effort must not be overwritten by an absent top-level field"

    def test_top_level_reasoning_effort_wins_over_nested(self, tokenizer):
        """An explicit top-level reasoning_effort overrides a nested one."""
        _, _, _, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "low",
                "chat_template_args": {"reasoning_effort": "high"},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        assert chat_params.chat_template_kwargs.get("reasoning_effort") == "low"

    def test_reserved_render_key_in_template_args_does_not_crash(self, tokenizer):
        """A renderer-reserved key nested in template kwargs must not raise TypeError."""
        _, _, _, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_args": {"documents": [{"text": "doc"}]},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        # Renderer-managed value wins over the client's nested key (no crash either).
        assert chat_params.chat_template_kwargs["documents"] is None


class TestServerDefaultChatTemplateKwargs:
    """The server-wide --default-chat-template-kwargs must reach the template."""

    def _enable_thinking(self, tokenizer, request_args):
        """Return the template's enable_thinking under a `{enable_thinking: False}` default."""
        request = {"model": MODEL, "messages": [{"role": "user", "content": "Hello"}]}
        if request_args is not None:
            request["chat_template_args"] = request_args
        _, _, _, _, chat_params = _prepare_request(
            request,
            tokenizer=tokenizer,
            tool_parser_class=None,
            default_chat_template_kwargs={"enable_thinking": False},
        )
        return chat_params.chat_template_kwargs.get("enable_thinking")

    @pytest.mark.parametrize(
        "request_args, expected",
        [
            (None, False),  # omitted request kwargs: the server default applies
            ({"enable_thinking": True}, True),  # a request kwarg overrides the default
            (
                {"enable_thinking": None},
                False,
            ),  # unset (None) request value keeps the default
        ],
    )
    def test_server_default_precedence(self, tokenizer, request_args, expected):
        assert self._enable_thinking(tokenizer, request_args) is expected

    def test_server_default_is_not_mutated(self, tokenizer):
        """Processing must not mutate the shared server-default dict."""
        default = {"enable_thinking": False}
        _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "reasoning_effort": "high",
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            default_chat_template_kwargs=default,
        )
        assert default == {"enable_thinking": False}


class TestMultimodalFeatureMetadata:
    def _feature(
        self, modality, mm_hash, offset, length, data=_DEFAULT_MM_DATA, is_embed=None
    ):
        return SimpleNamespace(
            modality=modality,
            mm_hash=mm_hash,
            data=data,
            mm_position=SimpleNamespace(
                offset=offset,
                length=length,
                is_embed=is_embed,
            ),
        )

    def test_groups_hashes_and_placeholders_by_modality(self):
        from dynamo.frontend.vllm_processor import _group_mm_feature_metadata

        features = [
            self._feature("image", "image_hash", 4, 8),
            self._feature("audio", "audio_hash", 20, 6),
        ]

        (
            flat_hashes,
            flat_placeholders,
            hashes_by_modality,
            placeholders_by_modality,
        ) = _group_mm_feature_metadata(features)

        assert flat_hashes == []
        assert flat_placeholders == []
        assert hashes_by_modality == {
            "image": ["image_hash"],
            "audio": ["audio_hash"],
        }
        assert placeholders_by_modality == {
            "image": [(4, 8)],
            "audio": [(20, 6)],
        }

    def test_image_only_metadata_keeps_legacy_flat_fields(self):
        from dynamo.frontend.vllm_processor import _group_mm_feature_metadata

        features = [
            self._feature("image", "image_hash_0", 4, 8),
            self._feature("image", "image_hash_1", 20, 6),
        ]

        (
            flat_hashes,
            flat_placeholders,
            hashes_by_modality,
            placeholders_by_modality,
        ) = _group_mm_feature_metadata(features)

        assert flat_hashes == ["image_hash_0", "image_hash_1"]
        assert flat_placeholders == [(4, 8), (20, 6)]
        assert hashes_by_modality == {"image": ["image_hash_0", "image_hash_1"]}
        assert placeholders_by_modality == {"image": [(4, 8), (20, 6)]}

    def test_placeholder_metadata_preserves_is_embed_mask(self):
        from dynamo.frontend.vllm_processor import _group_mm_feature_metadata

        mask = [False, True, True, False]
        features = [
            self._feature("image", "image_hash", 4, 4, is_embed=mask),
        ]

        _, flat_placeholders, _, placeholders_by_modality = _group_mm_feature_metadata(
            features
        )

        expected = {"offset": 4, "length": 4, "is_embed": mask}
        assert flat_placeholders == [expected]
        assert placeholders_by_modality == {"image": [expected]}

    def test_missing_hash_skips_only_that_feature(self):
        from dynamo.frontend.vllm_processor import _group_mm_feature_metadata

        features = [
            self._feature("image", "image_hash", 4, 8),
            self._feature("image", None, 20, 8),
        ]

        assert _group_mm_feature_metadata(features) == (
            ["image_hash"],
            [(4, 8)],
            {"image": ["image_hash"]},
            {"image": [(4, 8)]},
        )

    def test_single_transfer_modality_rejects_mixed_features(self):
        from dynamo.frontend.vllm_processor import _single_transfer_modality

        assert (
            _single_transfer_modality(
                [
                    self._feature("image", "image_hash", 4, 8),
                    self._feature("audio", "audio_hash", 20, 6),
                ]
            )
            is None
        )
        assert (
            _single_transfer_modality(
                [
                    self._feature("image", "image_hash_0", 4, 8),
                    self._feature("image", "image_hash_1", 20, 8),
                ]
            )
            == "image"
        )


@pytest.mark.asyncio
async def test_prepare_mm_routing_skips_single_modality_transfer_for_mixed_features(
    vllm_processor_module,
    monkeypatch,
):
    def fail_sender():
        raise AssertionError("mixed-modality requests must not construct a sender")

    monkeypatch.setattr(vllm_processor_module, "MmKwargsShmSender", fail_sender)

    processor = vllm_processor_module.VllmProcessor.__new__(
        vllm_processor_module.VllmProcessor
    )
    processor.block_size = 16
    processor.nixl_mm_enabled = True
    processor.use_shm_transfer = True
    processor._sender = None

    def feature(modality, mm_hash, offset, length):
        return SimpleNamespace(
            modality=modality,
            mm_hash=mm_hash,
            data=object(),
            mm_position=SimpleNamespace(offset=offset, length=length),
        )

    vllm_preproc = SimpleNamespace(
        prompt_token_ids=list(range(32)),
        mm_features=[
            feature("image", "a" * 64, 0, 16),
            feature("audio", "b" * 64, 16, 8),
        ],
    )
    dynamo_preproc = {}

    mm_routing_info, cleanup_items, transferred = await processor._prepare_mm_routing(
        vllm_preproc,
        dynamo_preproc,
    )

    assert mm_routing_info is not None
    assert cleanup_items == []
    assert transferred is False
    assert dynamo_preproc["extra_args"]["mm_hashes"] == []
    assert dynamo_preproc["extra_args"]["mm_hashes_by_modality"] == {
        "image": ["a" * 64],
        "audio": ["b" * 64],
    }


@pytest.mark.asyncio
async def test_prepare_mm_routing_does_not_forward_hashes_without_exact_routing(
    vllm_processor_module,
    monkeypatch,
):
    monkeypatch.setattr(
        vllm_processor_module,
        "build_mm_routing_info_from_features",
        lambda *_, **__: None,
    )

    processor = vllm_processor_module.VllmProcessor.__new__(
        vllm_processor_module.VllmProcessor
    )
    processor.block_size = 16
    processor.nixl_mm_enabled = False
    processor.use_shm_transfer = True
    processor._sender = None

    vllm_preproc = SimpleNamespace(
        prompt_token_ids=list(range(16)),
        mm_features=[
            SimpleNamespace(
                modality="image",
                mm_hash="a" * 64,
                data=object(),
                mm_position=SimpleNamespace(offset=0, length=16),
            )
        ],
    )
    dynamo_preproc = {}

    mm_routing_info, cleanup_items, transferred = await processor._prepare_mm_routing(
        vllm_preproc,
        dynamo_preproc,
    )

    assert mm_routing_info is None
    assert cleanup_items == []
    assert transferred is False
    assert "mm_hashes" not in dynamo_preproc["extra_args"]
    assert "mm_hashes_by_modality" not in dynamo_preproc["extra_args"]


@pytest.mark.asyncio
async def test_prepare_mm_routing_processor_kwargs_keep_transfer_but_skip_exact_routing(
    vllm_processor_module,
    monkeypatch,
):
    def fail_routing(*args, **kwargs):
        raise AssertionError("processor kwargs must disable exact MM routing")

    monkeypatch.setattr(
        vllm_processor_module,
        "build_mm_routing_info_from_features",
        fail_routing,
    )

    processor = vllm_processor_module.VllmProcessor.__new__(
        vllm_processor_module.VllmProcessor
    )
    processor.block_size = 16
    processor.nixl_mm_enabled = True
    processor.use_shm_transfer = True
    processor._sender = SimpleNamespace(
        prepare=AsyncMock(
            return_value=({"mm_kwargs_shm": {"modality": "image"}}, ["cleanup"])
        )
    )

    vllm_preproc = SimpleNamespace(
        prompt_token_ids=list(range(16)),
        mm_features=[
            SimpleNamespace(
                modality="image",
                mm_hash="a" * 64,
                data=object(),
                mm_position=SimpleNamespace(offset=0, length=16),
            )
        ],
    )
    dynamo_preproc = {}

    mm_routing_info, cleanup_items, transferred = await processor._prepare_mm_routing(
        vllm_preproc,
        dynamo_preproc,
        mm_processor_kwargs={"max_pixels": 4096},
    )

    assert mm_routing_info is None
    assert cleanup_items == ["cleanup"]
    assert transferred is True
    assert "mm_hashes" not in dynamo_preproc["extra_args"]
    assert "mm_hashes_by_modality" not in dynamo_preproc["extra_args"]
    assert dynamo_preproc["extra_args"]["mm_kwargs_shm"] == {"modality": "image"}
    assert dynamo_preproc["extra_args"]["mm_placeholders"] == [(0, 16)]
    assert dynamo_preproc["extra_args"]["expanded_token_ids"] == list(range(16))


@pytest.mark.asyncio
@pytest.mark.multimodal
async def test_prepare_mm_routing_opaque_uuid_skips_routing_and_transfer(
    vllm_processor_module,
    monkeypatch,
):
    def fail_routing(*args, **kwargs):
        raise AssertionError("opaque user UUIDs must not be parsed as routing hashes")

    monkeypatch.setattr(
        vllm_processor_module,
        "build_mm_routing_info_from_features",
        fail_routing,
    )

    def fail_sender():
        raise AssertionError("opaque user UUIDs must be processed by the worker")

    monkeypatch.setattr(vllm_processor_module, "MmKwargsShmSender", fail_sender)

    processor = vllm_processor_module.VllmProcessor.__new__(
        vllm_processor_module.VllmProcessor
    )
    processor.block_size = 16
    processor.nixl_mm_enabled = True
    processor.use_shm_transfer = True
    processor._sender = None

    feature_hash = "derived-from-uuid-and-processor-kwargs"
    vllm_preproc = SimpleNamespace(
        prompt_token_ids=list(range(16)),
        mm_features=[
            SimpleNamespace(
                modality="image",
                mm_hash=feature_hash,
                data=object(),
                mm_position=SimpleNamespace(offset=0, length=16),
            )
        ],
    )
    dynamo_preproc = {"multi_modal_uuids": {"image_url": ["opaque-user-key"]}}

    mm_routing_info, cleanup_items, transferred = await processor._prepare_mm_routing(
        vllm_preproc,
        dynamo_preproc,
    )

    assert mm_routing_info is None
    assert cleanup_items == []
    assert transferred is False
    assert "extra_args" not in dynamo_preproc


class TestReasoningParserMetadata:
    def test_no_reasoning_parser_returns_none(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        metadata = _build_reasoning_parser_metadata(
            None,
            object(),
            {},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
        )

        assert metadata.engine_reasoning_ended is None
        assert metadata.response_reasoning_ended is None
        assert metadata.parser_kwargs is None

    def test_include_reasoning_false_only_marks_engine_reasoning_ended(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class FakeReasoningParser:
            def __init__(self, tokenizer, *, chat_template_kwargs):
                self.tokenizer = tokenizer
                self.chat_template_kwargs = chat_template_kwargs

            def is_reasoning_end(self, prompt_token_ids):
                return prompt_token_ids == [9, 9]

        metadata = _build_reasoning_parser_metadata(
            FakeReasoningParser,
            object(),
            {"reasoning_effort": "low"},
            SimpleNamespace(include_reasoning=False),
            [1, 2, 3],
        )

        assert metadata.engine_reasoning_ended is True
        assert metadata.response_reasoning_ended is False
        assert metadata.parser_kwargs == {
            "chat_template_kwargs": {"reasoning_effort": "low"}
        }

    def test_parser_receives_chat_template_kwargs(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class FakeReasoningParser:
            def __init__(self, tokenizer, *, chat_template_kwargs):
                self.tokenizer = tokenizer
                self.chat_template_kwargs = chat_template_kwargs

            def is_reasoning_end(self, prompt_token_ids):
                return prompt_token_ids == [9, 9]

        tokenizer = object()
        metadata = _build_reasoning_parser_metadata(
            FakeReasoningParser,
            tokenizer,
            {"reasoning_effort": "high"},
            SimpleNamespace(include_reasoning=True),
            [9, 9],
        )

        assert metadata.engine_reasoning_ended is True
        assert metadata.response_reasoning_ended is True
        assert metadata.parser_kwargs == {
            "chat_template_kwargs": {"reasoning_effort": "high"}
        }

    def test_kv_router_copies_reasoning_metadata_to_extra_args(self):
        from dynamo.frontend.vllm_processor import _inject_routing_metadata

        kv_kwargs = {"extra_args": {"mm_hashes": [123]}}
        _inject_routing_metadata(
            {
                "reasoning_ended": False,
                "reasoning_parser_kwargs": {
                    "chat_template_kwargs": {"reasoning_effort": "high"}
                },
            },
            kv_kwargs,
        )

        assert kv_kwargs["extra_args"] == {
            "mm_hashes": [123],
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
        }


@pytest.mark.asyncio
@pytest.mark.multimodal
async def test_build_engine_inputs_preserves_multimodal_uuids(
    vllm_processor_module,
):
    class Renderer:
        async def process_for_engine_async(self, prompt, arrival_time):
            assert prompt == {
                "prompt": "rendered prompt",
                "prompt_token_ids": [1, 2, 3],
                "multi_modal_data": {"image": [image]},
                "multi_modal_uuids": {"image": ["opaque-user-key"]},
                "cache_salt": "salt",
                "mm_processor_kwargs": {"do_sample_frames": False},
            }
            assert isinstance(arrival_time, float)
            return {"type": "multimodal", "mm_hashes": ["opaque-user-key"]}

    image = object()
    engine_inputs = await vllm_processor_module._build_engine_inputs(
        Renderer(),
        {
            "prompt": "rendered prompt",
            "multi_modal_data": {"image": [image]},
            "multi_modal_uuids": {"image": ["opaque-user-key"]},
        },
        [1, 2, 3],
        cache_salt="salt",
        mm_processor_kwargs={"do_sample_frames": False},
    )

    assert engine_inputs == {
        "type": "multimodal",
        "mm_hashes": ["opaque-user-key"],
    }


@pytest.mark.asyncio
@pytest.mark.multimodal
async def test_build_engine_inputs_defers_uuid_only_processing_to_worker(
    vllm_processor_module,
):
    class Renderer:
        async def process_for_engine_async(self, prompt, arrival_time):
            assert prompt == {
                "prompt": "rendered prompt",
                "prompt_token_ids": [1, 2, 3],
                "cache_salt": "salt",
                "mm_processor_kwargs": {"do_sample_frames": False},
            }
            assert isinstance(arrival_time, float)
            return {"type": "token", "prompt_token_ids": [1, 2, 3]}

    engine_inputs = await vllm_processor_module._build_engine_inputs(
        Renderer(),
        {
            "prompt": "rendered prompt",
            "multi_modal_data": {"image": [None]},
            "multi_modal_uuids": {"image": ["worker-cached-image"]},
        },
        [1, 2, 3],
        cache_salt="salt",
        mm_processor_kwargs={"do_sample_frames": False},
        defer_multimodal_processing=True,
    )

    assert engine_inputs == {"type": "token", "prompt_token_ids": [1, 2, 3]}


@pytest.mark.multimodal
def test_normalize_vllm_image_parts_defaults_detail_without_lifting_nested_uuid(
    vllm_processor_module,
) -> None:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.png",
                        "detail": None,
                        "uuid": "92b888ad-e64a-478f-b688-5091e16544e3",
                    },
                }
            ],
        }
    ]

    vllm_processor_module._normalize_vllm_image_parts(messages)

    part = messages[0]["content"][0]
    assert "uuid" not in part
    assert part["image_url"]["detail"] == "auto"


class _FakeOutputProcessor:
    def __init__(self):
        self.request_states = {}
        self.added_requests = []
        self.aborted_requests = []

    def add_request(self, preproc, *args, **kwargs):
        self.added_requests.append((preproc, args, kwargs))
        self.request_states[preproc.request_id] = object()

    def process_outputs(self, outputs):
        return SimpleNamespace(
            reqs_to_abort=[],
            request_outputs=[SimpleNamespace(outputs=[SimpleNamespace(index=0)])],
        )

    def abort_requests(self, request_ids, internal=False):
        self.aborted_requests.append((request_ids, internal))
        for request_id in request_ids:
            self.request_states.pop(request_id, None)


class _FakePostProcessor:
    def process_output(self, output):
        return {
            "index": output.index,
            "delta": {"content": "x"},
            "finish_reason": None,
        }


@pytest.fixture
def vllm_processor_module(monkeypatch):
    import dynamo.frontend.vllm_processor as module

    class FakeEngineCoreOutput:
        __struct_fields__ = ()

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    monkeypatch.setattr(module, "EngineCoreOutput", FakeEngineCoreOutput)
    monkeypatch.setattr(module._nvtx, "start_range", lambda *args, **kwargs: object())
    monkeypatch.setattr(module._nvtx, "end_range", lambda rng: None)
    return module


@pytest.mark.asyncio
async def test_generator_preserves_zero_top_logprobs(
    vllm_processor_module,
    monkeypatch,
    caplog,
):
    class RequestForSampling(SimpleNamespace):
        model_fields = frozenset()

    monkeypatch.setattr(
        vllm_processor_module,
        "preprocess_chat_request",
        AsyncMock(
            return_value=SimpleNamespace(
                request_for_sampling=RequestForSampling(
                    max_completion_tokens=None,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=0,
                    cache_salt=None,
                    mm_processor_kwargs=None,
                ),
                tool_parser=None,
                chat_template_kwargs={},
                engine_prompt={"prompt": "Hello"},
                prompt_token_ids=[1],
                guided_decoding=None,
            )
        ),
    )

    class ProjectionObserved(Exception):
        pass

    def process_inputs(request_id, engine_inputs, sampling_params, supported_tasks):
        assert sampling_params.logprobs == 0
        raise ProjectionObserved

    input_processor = SimpleNamespace(
        generation_config_fields={},
        renderer=SimpleNamespace(process_for_engine_async=AsyncMock(return_value={})),
        process_inputs=process_inputs,
    )

    processor = vllm_processor_module.VllmProcessor(
        tokenizer=SimpleNamespace(eos_token_id=2),
        input_processor=input_processor,
        output_processor=object(),
        tool_parser_class=None,
        reasoning_parser_class=None,
        routed_engine=object(),
    )

    with pytest.raises(ProjectionObserved):
        await anext(
            processor._generator_inner(
                {
                    "model": "test",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "logprobs": True,
                    "top_logprobs": 0,
                }
            )
        )
    assert (
        "Logprobs requested but not supported in distributed inference mode"
        in caplog.messages
    )


@pytest.mark.asyncio
async def test_include_reasoning_false_keeps_response_parser_active(
    vllm_processor_module,
    monkeypatch,
):
    class RequestForSampling(SimpleNamespace):
        model_fields = frozenset()

    class FakeReasoningParser:
        engine_based_streaming = False

        def __init__(self, tokenizer, *, chat_template_kwargs):
            self.tokenizer = tokenizer
            self.chat_template_kwargs = chat_template_kwargs
            self.adjusted_prompt_token_ids = None

        def is_reasoning_end(self, prompt_token_ids):
            return False

        def adjust_initial_state_from_prompt(self, prompt_token_ids):
            self.adjusted_prompt_token_ids = prompt_token_ids

        def extract_reasoning_streaming(
            self,
            previous_text,
            current_text,
            delta_text,
            previous_token_ids,
            current_token_ids,
            delta_token_ids,
        ):
            if "</think>" in delta_text:
                return prepost_module.DeltaMessage(
                    reasoning="hidden tail",
                    content="visible answer",
                )
            return prepost_module.DeltaMessage(reasoning="hidden reasoning")

        def is_reasoning_end_streaming(self, current_token_ids, delta_token_ids):
            return 3 in delta_token_ids

    monkeypatch.setattr(
        vllm_processor_module,
        "preprocess_chat_request",
        AsyncMock(
            return_value=SimpleNamespace(
                request_for_sampling=RequestForSampling(
                    max_completion_tokens=None,
                    max_tokens=1,
                    logprobs=None,
                    top_logprobs=None,
                    cache_salt=None,
                    mm_processor_kwargs=None,
                    include_reasoning=False,
                ),
                tool_parser=None,
                chat_template_kwargs={"reasoning_effort": "low"},
                engine_prompt={"prompt": "Hello"},
                prompt_token_ids=[1],
                guided_decoding=None,
                uses_dynamo_json_tool_call_fallback=False,
            )
        ),
    )
    monkeypatch.setattr(
        vllm_processor_module.InputProcessor,
        "assign_request_id",
        lambda request: None,
    )

    def process_inputs(request_id, engine_inputs, sampling_params, supported_tasks):
        return SimpleNamespace(sampling_params=sampling_params, mm_features=None)

    input_processor = SimpleNamespace(
        generation_config_fields={},
        renderer=SimpleNamespace(process_for_engine_async=AsyncMock(return_value={})),
        process_inputs=process_inputs,
    )
    processor = vllm_processor_module.VllmProcessor(
        tokenizer=SimpleNamespace(eos_token_id=2, all_special_tokens=[]),
        input_processor=input_processor,
        output_processor=object(),
        tool_parser_class=None,
        reasoning_parser_class=FakeReasoningParser,
        routed_engine=object(),
    )
    monkeypatch.setattr(
        processor,
        "_prepare_mm_routing",
        AsyncMock(return_value=(None, [], False)),
    )

    captured = {}

    async def capture_generate_and_stream(
        request_id,
        request,
        dynamo_preproc,
        tokens,
        vllm_preproc,
        post_processors,
        *,
        mm_routing_info,
        context,
    ):
        captured["dynamo_preproc"] = dynamo_preproc
        captured["post_processor"] = post_processors[0]
        yield {"captured": True}

    monkeypatch.setattr(
        processor,
        "_generate_and_stream",
        capture_generate_and_stream,
    )

    results = [
        item
        async for item in processor._generator_inner(
            {
                "model": "test",
                "messages": [{"role": "user", "content": "Hello"}],
            }
        )
    ]

    assert results == [{"captured": True}]
    assert captured["dynamo_preproc"]["reasoning_ended"] is True
    post_processor = captured["post_processor"]
    assert post_processor.reasoning_is_done is False
    assert isinstance(post_processor.reasoning_parser, FakeReasoningParser)
    assert post_processor.reasoning_parser.adjusted_prompt_token_ids == [1]

    reasoning_choice = post_processor.process_output(
        SimpleNamespace(
            index=0,
            text="<think>hidden reasoning",
            token_ids=[2],
            finish_reason=None,
            logprobs={"tokens": ["hidden"]},
        )
    )
    assert reasoning_choice is None

    content_choice = post_processor.process_output(
        SimpleNamespace(
            index=0,
            text="</think>visible answer",
            token_ids=[3],
            finish_reason="stop",
            logprobs={"tokens": ["hidden", "visible"]},
        )
    )
    assert content_choice == {
        "index": 0,
        "delta": {"role": "assistant", "content": "visible answer"},
        "finish_reason": "stop",
        "logprobs": None,
    }


def _make_processor(module, routed_engine):
    processor = module.VllmProcessor.__new__(module.VllmProcessor)
    processor.routed_engine = routed_engine
    processor.output_processor = _FakeOutputProcessor()
    return processor


def _base_preproc():
    return {
        "model": MODEL,
        "token_ids": [1, 2, 3],
        "stop_conditions": {"max_tokens": 4},
        "sampling_options": {"temperature": 0.0},
        "output_options": {},
        "eos_token_ids": [],
        "annotations": [],
        "routing": None,
    }


async def _run_generate(processor, preproc, *, mm_routing_info=None, context=None):
    vllm_preproc = SimpleNamespace(
        sampling_params=SimpleNamespace(n=1),
        request_id="vllm-request",
        external_req_id=None,
    )
    post_processors = {0: _FakePostProcessor()}

    return [
        item
        async for item in processor._generate_and_stream(
            "request-id",
            {"model": MODEL},
            preproc,
            preproc["token_ids"],
            vllm_preproc,
            post_processors,
            mm_routing_info=mm_routing_info,
            context=context,
        )
    ]


class TestRoutedEnginePath:
    @pytest.mark.asyncio
    async def test_routed_engine_gets_extra_args_metadata(self, vllm_processor_module):
        routed_engine = _FakeRoutedEngine()
        processor = _make_processor(vllm_processor_module, routed_engine)
        preproc = _base_preproc()
        preproc["extra_args"] = {"mm_hashes": [123]}
        preproc["reasoning_ended"] = False
        preproc["reasoning_parser_kwargs"] = {
            "chat_template_kwargs": {"reasoning_effort": "high"}
        }
        preproc["mm_processor_kwargs"] = {"use_audio_in_video": True}

        await _run_generate(processor, preproc)

        assert routed_engine.requests[0]["extra_args"] == {
            "mm_hashes": [123],
            "reasoning_ended": False,
            "reasoning_parser_kwargs": {
                "chat_template_kwargs": {"reasoning_effort": "high"}
            },
            "mm_processor_kwargs": {"use_audio_in_video": True},
        }

    @pytest.mark.asyncio
    async def test_routed_stream_produces_openai_chunks(self, vllm_processor_module):
        routed_engine = _FakeRoutedEngine(
            [{"token_ids": [101], "index": 0, "finish_reason": None}]
        )
        processor = _make_processor(vllm_processor_module, routed_engine)

        chunks = await _run_generate(processor, _base_preproc())

        # One annotated envelope per iteration carries both data and the
        # llm_metrics annotation; observer strips the annotation before SSE.
        assert len(chunks) == 1
        envelope = chunks[0]

        assert envelope["_dynamo_annotated"] is True
        assert envelope["data"] == {
            "id": "request-id",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "x"},
                    "finish_reason": None,
                }
            ],
            "created": envelope["data"]["created"],
            "model": MODEL,
            "object": "chat.completion.chunk",
        }

        assert envelope["event"] == "llm_metrics"
        assert len(envelope["comment"]) == 1
        # Zero counts are omitted (text-only request), mirroring the Rust skip-zero behavior.
        assert json.loads(envelope["comment"][0]) == {
            "input_tokens": 3,
            "output_tokens": 1,
            "chunk_tokens": 1,
        }

    @pytest.mark.asyncio
    async def test_routed_stream_emits_multimodal_counts(self, vllm_processor_module):
        # The Rust postprocessor is bypassed on this path, so the processor must
        # emit per-request multimodal content-part counts itself.
        routed_engine = _FakeRoutedEngine(
            [{"token_ids": [101], "index": 0, "finish_reason": None}]
        )
        processor = _make_processor(vllm_processor_module, routed_engine)

        request = {
            "model": MODEL,
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
        preproc = _base_preproc()
        chunks = [
            item
            async for item in processor._generate_and_stream(
                "request-id",
                request,
                preproc,
                preproc["token_ids"],
                SimpleNamespace(
                    sampling_params=SimpleNamespace(n=1),
                    request_id="vllm-request",
                    external_req_id=None,
                ),
                {0: _FakePostProcessor()},
                mm_routing_info=None,
                context=None,
            )
        ]

        metrics = json.loads(chunks[0]["comment"][0])
        assert metrics["image_count"] == 2
        assert metrics["video_count"] == 1
        # audio has zero parts, so the key is omitted from the emitted metrics.
        assert metrics.get("audio_count") is None


OBJECT_TYPED_TOOL_REQUEST = {
    "model": MODEL,
    "messages": [{"role": "user", "content": "set my profile"}],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "set_profile",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "profile": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "age": {"type": "integer"},
                            },
                        }
                    },
                    "required": ["profile"],
                },
            },
        }
    ],
    "tool_choice": "auto",
}


# ---------------------------------------------------------------------------
# _prepare_request: schema-aware tool-parser end-to-end regression
# ---------------------------------------------------------------------------


class TestSchemaAwareToolParser:
    """Schema-aware parsers (e.g. qwen3_coder) need ``tools`` at construction
    to coerce object/array-typed parameter values from raw text into JSON;
    without them, the value comes through as a string-in-a-string inside the
    final ``arguments`` JSON.
    """

    def test_qwen3_coder_coerces_object_typed_arg(self, tokenizer):
        """qwen3_coder must coerce object-typed parameter values into nested
        objects, not leave them as JSON-encoded strings inside ``arguments``.
        """
        model_output = (
            "<tool_call><function=set_profile>\n"
            "<parameter=profile>\n"
            '{"name": "Alice", "age": 30}\n'
            "</parameter>\n"
            "</function></tool_call>"
        )

        request_for_sampling, parser, _, _, _ = _prepare_request(
            OBJECT_TYPED_TOOL_REQUEST,
            tokenizer=tokenizer,
            tool_parser_class=_resolve_qwen3_tool_parser_class(),
        )
        assert parser is not None, "Expected _prepare_request to construct the parser"

        result = parser.extract_tool_calls(model_output, request_for_sampling)

        assert result.tools_called, f"Expected tools_called=True; got {result!r}"
        assert len(result.tool_calls) == 1
        args = json.loads(result.tool_calls[0].function.arguments)
        assert isinstance(args["profile"], dict), (
            f"Schema-aware parser should coerce object-typed arg to dict; "
            f"got {type(args['profile']).__name__}: {args['profile']!r}"
        )
        assert args["profile"] == {"name": "Alice", "age": 30}


class _FakeStructuralTag:
    def __init__(self, value):
        self.value = value

    def model_dump(self):
        return self.value


class _FakeStructuralTagParser:
    structural_tag_model = None

    def __init__(self):
        self.requests = []

    def get_structural_tag(self, request, *, reasoning=False):
        self.requests.append(request)
        strict = [tool.function.strict for tool in request.tools]
        return _FakeStructuralTag({"format": {"strict": strict}})


class _FakeGrammarToolParser:
    def get_structural_tag(self, request, *, reasoning=False):
        return None


class _FakeAdjustRequestGrammarToolParser(_FakeGrammarToolParser):
    def __init__(self, tokenizer, tools):
        del tokenizer, tools

    def adjust_request(self, request):
        request.structured_outputs = StructuredOutputsParams(
            grammar='root ::= "<tool_call>"'
        )
        return request


class _FakePassthroughToolParser(_FakeGrammarToolParser):
    def __init__(self, tokenizer, tools):
        del tokenizer, tools

    def adjust_request(self, request):
        return request


class _FakeResponseFormatConsumingToolParser(_FakeAdjustRequestGrammarToolParser):
    def adjust_request(self, request):
        request.response_format = None
        return super().adjust_request(request)


class _FakeNativeSyntaxToolParser(_FakeGrammarToolParser):
    # Mirrors parsers like Gemma4Engine/PoolsideV1 that emit native tool syntax
    # and decline a forced JSON constraint for required/named choices.
    supports_required_and_named = False


class TestPreprocessRawRequestControls:
    """Transport-only controls must survive the preprocess_chat_request boundary.

    Regression: preprocess_chat_request passed the validated model into
    _prepare_request, making its dict-only reads of chat_template_args and root
    thinking unreachable. Direct _prepare_request tests do not cover this path.
    """

    @staticmethod
    def _renderer():
        return SimpleNamespace(
            render_messages_async=AsyncMock(
                return_value=(None, {"prompt_token_ids": [1]})
            )
        )

    @pytest.mark.asyncio
    async def test_chat_template_args_survive_preprocess(self, tokenizer):
        result = await prepost_module.preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "chat_template_args": {"enable_thinking": False},
            },
            tokenizer=tokenizer,
            renderer=self._renderer(),
            tool_parser_class=None,
        )
        assert result.chat_template_kwargs.get("enable_thinking") is False

    @pytest.mark.asyncio
    async def test_root_thinking_suppresses_deployment_default(self, tokenizer):
        # An explicit root thinking control must block the deployment default
        # from injecting its own thinking_mode.
        result = await prepost_module.preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "thinking": True,
            },
            tokenizer=tokenizer,
            renderer=self._renderer(),
            tool_parser_class=None,
            default_thinking_mode="disabled",
        )
        assert "thinking_mode" not in result.chat_template_kwargs


class _FakeStructuralTagReasoningParser:
    """A reasoning parser that rewrites the caller's constraint.

    Mirrors a real text-grammar parser converting response_format into the
    structural tag its model actually speaks.
    """

    def __init__(self, tokenizer, chat_template_kwargs=None):
        del tokenizer, chat_template_kwargs

    def adjust_request(self, request):
        request.skip_special_tokens = False
        request.structured_outputs = StructuredOutputsParams(
            structural_tag='{"format": "reasoning"}'
        )
        return request


class _FakePassthroughReasoningParser(_FakeStructuralTagReasoningParser):
    def adjust_request(self, request):
        return request


class TestReasoningParserGuidanceForwarding:
    """A reasoning parser's guidance must be recomputed and forwarded.

    Regression: parser_guided_decoding was gated on `tool_parser is not None`, and
    build_tool_call_guided_decoding returns early when there is no tool-call
    guidance to build. A reasoning parser that rewrote the caller's constraint
    therefore never surfaced: the request carried the adjusted constraint while the
    engine received the pre-adjustment copy captured before _prepare_request ran.
    """

    @staticmethod
    def _request(**overrides):
        """A fresh request each call, so no test can perturb another's input."""
        return {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
            "response_format": {"type": "json_object"},
            **overrides,
        }

    @staticmethod
    def _renderer():
        return SimpleNamespace(
            render_messages_async=AsyncMock(
                return_value=(None, {"prompt_token_ids": [1]})
            )
        )

    async def _preprocess(self, tokenizer, *, request=None, **kwargs):
        kwargs.setdefault("tool_parser_class", None)
        return await prepost_module.preprocess_chat_request(
            request if request is not None else self._request(),
            tokenizer=tokenizer,
            renderer=self._renderer(),
            **kwargs,
        )

    @pytest.mark.asyncio
    async def test_rewrite_is_forwarded_without_a_tool_parser(self, tokenizer):
        """The whole point: no tool parser anywhere in the request."""
        result = await self._preprocess(
            tokenizer,
            reasoning_parser_class=_FakeStructuralTagReasoningParser,
        )

        assert result.guided_decoding == {"structural_tag": '{"format": "reasoning"}'}
        # The pre-adjustment copy must not win.
        assert result.guided_decoding != {"json": {"type": "object"}}

    @pytest.mark.asyncio
    async def test_passthrough_parser_keeps_the_caller_constraint(self, tokenizer):
        """adjust_request that changes nothing must not disturb the caller."""
        result = await self._preprocess(
            tokenizer,
            reasoning_parser_class=_FakePassthroughReasoningParser,
        )

        assert result.guided_decoding == {"json": {"type": "object"}}

    @pytest.mark.asyncio
    async def test_no_reasoning_parser_keeps_the_caller_constraint(self, tokenizer):
        result = await self._preprocess(tokenizer, reasoning_parser_class=None)

        assert result.guided_decoding == {"json": {"type": "object"}}

    @pytest.mark.asyncio
    async def test_thinking_disabled_does_not_recompute(self, tokenizer):
        """The gate is off, so no parser runs and nothing may be recomputed.

        Recomputing here would forward a constraint from a parser that the
        postprocessor will not build.
        """
        result = await self._preprocess(
            tokenizer,
            request=self._request(chat_template_kwargs={"enable_thinking": False}),
            reasoning_parser_class=_FakeStructuralTagReasoningParser,
        )

        assert result.guided_decoding == {"json": {"type": "object"}}
        assert result.request_for_sampling.skip_special_tokens is not False

    @pytest.mark.asyncio
    async def test_rewrite_is_forwarded_with_a_tool_parser_also_active(self, tokenizer):
        """Attribution is measured, so a tool parser being present is irrelevant.

        Both parsers run against the same request inside _prepare_request. The
        snapshot taken between them shows the tool parser left the guidance alone
        and the reasoning parser rewrote it, so the rewrite is attributable and
        wins -- exactly as it does with no tool parser at all.
        """
        result = await self._preprocess(
            tokenizer,
            request=self._request(tools=TOOL_REQUEST["tools"], tool_choice="auto"),
            tool_parser_class=_FakePassthroughToolParser,
            reasoning_parser_class=_FakeStructuralTagReasoningParser,
        )

        assert result.guided_decoding == {"structural_tag": '{"format": "reasoning"}'}

    @pytest.mark.asyncio
    async def test_tool_parser_rewrite_still_loses_to_the_caller(self, tokenizer):
        """The other half: tool-path precedence is deliberately unchanged.

        Same measurement, opposite attribution -- the tool parser moved the
        guidance and the reasoning parser did not -- so the caller's explicit
        constraint still wins, as pinned by
        test_assistant_guidance_takes_precedence_over_auto_tool_guidance and the
        response-format-precedence row of TOOL_GUIDANCE_PARITY_CASES.
        """
        result = await self._preprocess(
            tokenizer,
            request=self._request(tools=TOOL_REQUEST["tools"], tool_choice="auto"),
            tool_parser_class=_FakeAdjustRequestGrammarToolParser,
            reasoning_parser_class=_FakePassthroughReasoningParser,
        )

        assert result.guided_decoding == {"json": {"type": "object"}}


class TestToolCallGuidedDecoding:
    def _request(self, tokenizer, **overrides):
        raw_request = {**TOOL_REQUEST, "tool_choice": "auto", **overrides}
        request, _, _, _, _ = _prepare_request(
            raw_request,
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        return request

    def test_mode_off_emits_no_guidance(self, tokenizer):
        parser = _FakeStructuralTagParser()
        guided = build_tool_call_guided_decoding(
            self._request(tokenizer),
            parser,
            structural_tag_mode="off",
            structural_tag_scope="always",
            structural_tag_schema="strict",
        )

        assert guided is None
        assert parser.requests == []

    # Forced choices need a JSON constraint even when structural tags are disabled.
    @pytest.mark.parametrize(
        "tool_choice",
        [
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    def test_forced_choice_uses_json_guidance_when_mode_off(
        self, tokenizer, tool_choice
    ):
        parser = _FakeStructuralTagParser()
        guided = build_tool_call_guided_decoding(
            self._request(tokenizer, tool_choice=tool_choice),
            parser,
            structural_tag_mode="off",
        )

        assert guided is not None
        assert set(guided) == {"json"}
        assert parser.requests == []

    def test_required_choice_disables_parallel_calls_in_json_guidance(self, tokenizer):
        guided = build_tool_call_guided_decoding(
            self._request(
                tokenizer,
                tool_choice="required",
                parallel_tool_calls=False,
            ),
            tool_parser=None,
        )

        assert guided is not None
        assert guided["json"]["type"] == "array"
        assert guided["json"]["maxItems"] == 1

    # Parsers that require native tool syntax must not get a forced JSON schema.
    @pytest.mark.parametrize(
        "tool_choice",
        [
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    def test_forced_choice_skips_json_for_native_syntax_parser(
        self, tokenizer, tool_choice
    ):
        guided = build_tool_call_guided_decoding(
            self._request(tokenizer, tool_choice=tool_choice),
            _FakeNativeSyntaxToolParser(),
            structural_tag_mode="off",
        )

        assert guided is None

    def test_strict_schema_uses_copy(self, tokenizer):
        request = self._request(tokenizer)
        parser = _FakeStructuralTagParser()
        guided = build_tool_call_guided_decoding(
            request,
            parser,
            structural_tag_mode="on",
            structural_tag_scope="always",
            structural_tag_schema="strict",
        )

        assert guided == {"structural_tag": {"format": {"strict": [True]}}}
        assert request.tools[0].function.strict is None
        assert parser.requests[0].messages is request.messages

    # Auto schema mode must preserve an omitted strict flag as unset.
    def test_auto_schema_preserves_unset_strict(self, tokenizer):
        parser = _FakeStructuralTagParser()
        guided = build_tool_call_guided_decoding(
            self._request(tokenizer),
            parser,
            structural_tag_mode="on",
            structural_tag_scope="always",
            structural_tag_schema="auto",
        )

        assert guided == {"structural_tag": {"format": {"strict": [None]}}}

    # Parser-created grammar must survive preprocessing as guided decoding.
    @pytest.mark.asyncio
    async def test_parser_adjust_request_generated_grammar_is_forwarded(
        self, tokenizer
    ):
        result = await prepost_module.preprocess_chat_request(
            {**TOOL_REQUEST, "tool_choice": "required"},
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(None, {"prompt_token_ids": [1]})
                )
            ),
            tool_parser_class=_FakeAdjustRequestGrammarToolParser,
            structural_tag_mode="on",
            structural_tag_scope="auto",
            structural_tag_schema="strict",
        )

        assert result.guided_decoding == {"grammar": 'root ::= "<tool_call>"'}
        assert result.uses_dynamo_json_tool_call_fallback is False

    @pytest.mark.parametrize(
        "tool_parser_class",
        [None, _FakePassthroughToolParser],
        ids=["no-parser", "parser-without-guidance"],
    )
    @pytest.mark.asyncio
    async def test_generic_json_guidance_enables_matching_postprocessor(
        self, tokenizer, tool_parser_class
    ):
        result = await prepost_module.preprocess_chat_request(
            {**TOOL_REQUEST, "tool_choice": "required"},
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(None, {"prompt_token_ids": [1]})
                )
            ),
            tool_parser_class=tool_parser_class,
            structural_tag_mode="off",
        )

        assert result.guided_decoding is not None
        assert set(result.guided_decoding) == {"json"}
        assert result.uses_dynamo_json_tool_call_fallback is True

        post = StreamingPostProcessor(
            tokenizer=tokenizer,
            request_for_sampling=result.request_for_sampling,
            sampling_params=SamplingParams(),
            prompt_token_ids=result.prompt_token_ids,
            tool_parser=result.tool_parser,
            reasoning_parser_class=None,
            chat_template_kwargs=result.chat_template_kwargs,
            stream_response=False,
            uses_dynamo_json_tool_call_fallback=(
                result.uses_dynamo_json_tool_call_fallback
            ),
        )
        choice = post.process_output(
            SimpleNamespace(
                index=0,
                text='[{"name":"get_weather","parameters":{"city":"Paris"}}]',
                token_ids=[],
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice is not None
        assert choice["finish_reason"] == "tool_calls"
        assert choice["delta"]["tool_calls"][0]["function"] == {
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }

    # Explicit assistant constraints must override automatic tool-call guidance.
    @pytest.mark.parametrize(
        "assistant_constraint, expected",
        [
            (
                {"response_format": {"type": "json_object"}},
                {"json": {"type": "object"}},
            ),
            ({"guided_regex": "yes|no"}, {"regex": "yes|no"}),
        ],
    )
    @pytest.mark.asyncio
    async def test_assistant_guidance_takes_precedence_over_auto_tool_guidance(
        self, tokenizer, assistant_constraint, expected
    ):
        result = await prepost_module.preprocess_chat_request(
            {
                **TOOL_REQUEST,
                "tool_choice": "auto",
                **assistant_constraint,
            },
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(None, {"prompt_token_ids": [1]})
                )
            ),
            tool_parser_class=_FakeResponseFormatConsumingToolParser,
            structural_tag_mode="on",
            structural_tag_scope="always",
        )

        assert result.guided_decoding == expected

    # A forced tool choice with a client-provided structured_outputs constraint
    # is a conflict: reject it rather than silently reuse or replace it. (The
    # parser-generated case is covered by the test above, which is not rejected.)
    @pytest.mark.asyncio
    async def test_forced_choice_with_client_structured_outputs_is_rejected(
        self, tokenizer
    ):
        with pytest.raises(InvalidArgument):
            await prepost_module.preprocess_chat_request(
                {
                    **TOOL_REQUEST,
                    "tool_choice": "required",
                    "structured_outputs": {"grammar": 'root ::= "not_a_tool_call"'},
                },
                tokenizer=tokenizer,
                renderer=SimpleNamespace(
                    render_messages_async=AsyncMock(
                        return_value=(None, {"prompt_token_ids": [1]})
                    )
                ),
                tool_parser_class=_FakePassthroughToolParser,
                structural_tag_mode="on",
            )

    # A forced tool choice plus an explicit constraint is a client error (400),
    # matching the Rust path, not a silently dropped constraint.
    @pytest.mark.asyncio
    async def test_forced_choice_with_explicit_constraint_is_rejected(self, tokenizer):
        with pytest.raises(InvalidArgument):
            await prepost_module.preprocess_chat_request(
                {
                    **TOOL_REQUEST,
                    "tool_choice": "required",
                    "guided_regex": "yes|no",
                },
                tokenizer=tokenizer,
                renderer=SimpleNamespace(
                    render_messages_async=AsyncMock(
                        return_value=(None, {"prompt_token_ids": [1]})
                    )
                ),
                tool_parser_class=_FakePassthroughToolParser,
                structural_tag_mode="on",
            )

    # response_format is scoped to the message returned to the user, not to tool
    # calls, so a forced tool choice drops it and keeps the tool constraint -- it
    # is NOT the 400 that guided_*/structured_outputs get. This matches
    # preprocessor/tool_choice.rs (clears gd.json) and vLLM's own
    # ToolParser.adjust_request (sets response_format = None).
    @pytest.mark.parametrize(
        "tool_choice",
        [
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
        ids=["required", "named"],
    )
    @pytest.mark.parametrize(
        "response_format",
        [
            {"type": "json_object"},
            {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": {"type": "object"}},
            },
        ],
        ids=["json_object", "json_schema"],
    )
    @pytest.mark.asyncio
    async def test_forced_choice_with_response_format_keeps_tool_guidance(
        self, tokenizer, tool_choice, response_format
    ):
        result = await prepost_module.preprocess_chat_request(
            {
                **TOOL_REQUEST,
                "tool_choice": tool_choice,
                "response_format": response_format,
            },
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(None, {"prompt_token_ids": [1]})
                )
            ),
            tool_parser_class=_FakeResponseFormatConsumingToolParser,
            structural_tag_mode="on",
        )

        # The parser's tool grammar survives; response_format is dropped rather
        # than rejected or allowed to win.
        assert result.guided_decoding == {"grammar": 'root ::= "<tool_call>"'}

    # response_format={"type": "structural_tag"} is not a content format -- vLLM
    # normalizes it into structured_outputs.structural_tag, a grammar over the
    # whole token stream. A forced choice must reject it rather than silently
    # discard it the way the json_object/json_schema variants are dropped.
    @pytest.mark.asyncio
    async def test_forced_choice_with_structural_tag_response_format_is_rejected(
        self, tokenizer
    ):
        with pytest.raises(InvalidArgument):
            await prepost_module.preprocess_chat_request(
                {
                    **TOOL_REQUEST,
                    "tool_choice": "required",
                    "response_format": {
                        "type": "structural_tag",
                        "format": {"type": "any_text"},
                    },
                },
                tokenizer=tokenizer,
                renderer=SimpleNamespace(
                    render_messages_async=AsyncMock(
                        return_value=(None, {"prompt_token_ids": [1]})
                    )
                ),
                tool_parser_class=_FakePassthroughToolParser,
                structural_tag_mode="on",
            )

    # On the DYN_VLLM_SKIP_REQUEST_VALIDATION fast path, an already-typed `tools`
    # list means the request is never re-validated, so `tool_choice` stays a raw
    # dict. vLLM branches on the typed named param: get_json_schema_from_tools
    # returns None for a dict (no constraint at all) and get_structural_tag raises
    # AttributeError on .model_dump(). Normalize once at the validation boundary.
    def _fast_path_named_request(self, tokenizer):
        typed_tools = _prepare_request(
            TOOL_REQUEST, tokenizer=tokenizer, tool_parser_class=None
        )[0].tools
        return prepost_module._validate_chat_completion_request(
            {
                **TOOL_REQUEST,
                "tools": typed_tools,
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "get_weather"},
                },
            }
        )

    def test_raw_named_tool_choice_is_normalized_at_the_boundary(self, tokenizer):
        request = self._fast_path_named_request(tokenizer)
        assert not isinstance(request.tool_choice, dict)
        assert request.tool_choice.function.name == "get_weather"

    def test_normalized_named_choice_still_builds_json_schema(self, tokenizer):
        guided = build_tool_call_guided_decoding(
            self._fast_path_named_request(tokenizer), None
        )
        assert guided is not None, "named forced choice must still be constrained"
        assert "json" in guided

    def test_normalized_named_choice_does_not_break_structural_tag(self, tokenizer):
        # Regression: a raw dict reached vLLM's structural-tag registry and raised
        # AttributeError on .model_dump(), surfacing as a 500 rather than a
        # constraint. Uses the REAL hermes parser -- a fake never reaches the
        # registry, so it cannot reproduce this.
        from vllm.tool_parsers import ToolParserManager

        hermes_cls = ToolParserManager.get_tool_parser("hermes")
        request = self._fast_path_named_request(tokenizer)
        parser = hermes_cls(tokenizer, request.tools)

        guided = build_tool_call_guided_decoding(
            request, parser, structural_tag_mode="on"
        )

        assert guided is not None
        assert "structural_tag" in guided

    # A malformed tool_choice object is not a named tool choice, so it must not be
    # treated as forced and must not trigger the forced-choice conflict check.
    @pytest.mark.parametrize(
        "tool_choice",
        [{}, {"type": "function"}, {"type": "function", "function": {}}],
        ids=["empty", "no_function", "no_name"],
    )
    def test_malformed_tool_choice_is_not_forced(self, tool_choice):
        assert prepost_module._is_named_tool_choice(tool_choice) is False
        assert prepost_module._is_forced_tool_choice(tool_choice) is False

    # Legacy guided_* must resolve to a single constraint, not a merged dict.
    @pytest.mark.asyncio
    async def test_legacy_guided_fields_yield_single_constraint(self, tokenizer):
        result = await prepost_module.preprocess_chat_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                "guided_json": {"type": "object"},
                "guided_regex": "\\d+",
            },
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(None, {"prompt_token_ids": [1]})
                )
            ),
            tool_parser_class=None,
        )
        assert result.guided_decoding == {"json": {"type": "object"}}

    # Keep vLLM's guidance decisions aligned with the shared backend matrix.
    @pytest.mark.parametrize(
        "case",
        TOOL_GUIDANCE_PARITY_CASES,
        ids=lambda case: case.name,
    )
    @pytest.mark.asyncio
    async def test_shared_tool_guidance_policy(self, tokenizer, case):
        request = {
            "model": MODEL,
            "messages": [{"role": "user", "content": "Hello"}],
        }
        if case.has_tools:
            request["tools"] = [parity_tool()]
            request["tool_choice"] = tool_choice_value(case.tool_choice)
        if case.has_assistant_constraint:
            request["response_format"] = assistant_response_format()

        result = await prepost_module.preprocess_chat_request(
            request,
            tokenizer=tokenizer,
            renderer=SimpleNamespace(
                render_messages_async=AsyncMock(
                    return_value=(None, {"prompt_token_ids": [1]})
                )
            ),
            tool_parser_class=_FakeResponseFormatConsumingToolParser,
            structural_tag_mode="on",
            structural_tag_scope="always",
        )

        assert (
            classify_guidance_source(
                result.guided_decoding,
                has_assistant_constraint=case.has_assistant_constraint,
            )
            == case.expected
        )

    def test_tool_choice_none_does_not_request_vllm_guidance(self, tokenizer):
        parser = _FakeStructuralTagParser()
        guided = build_tool_call_guided_decoding(
            self._request(tokenizer, tool_choice="none"),
            parser,
            structural_tag_mode="on",
            structural_tag_scope="always",
            structural_tag_schema="strict",
        )

        assert guided is None
        assert parser.requests == []

    @pytest.mark.parametrize(
        "schema_mode, expects_unconstrained_schema",
        [("auto", True), ("strict", False)],
    )
    def test_vllm_parser_receives_schema_mode(
        self, tokenizer, schema_mode, expects_unconstrained_schema
    ):
        tools = [
            TOOL_REQUEST["tools"][0],
            {
                "type": "function",
                "function": {
                    "name": "get_temperature",
                    "description": "Get the temperature for a location",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                    "strict": False,
                },
            },
        ]
        tools[0] = {
            **tools[0],
            "function": {**tools[0]["function"], "strict": True},
        }
        request = self._request(tokenizer, tools=tools)
        parser = _FakeStructuralTagParser()

        guided = build_tool_call_guided_decoding(
            request,
            parser,
            structural_tag_mode="on",
            structural_tag_scope="always",
            structural_tag_schema=schema_mode,
        )

        assert guided is not None
        strict_values = guided["structural_tag"]["format"]["strict"]
        assert strict_values == [True, not expects_unconstrained_schema]

    def test_auto_scope_requires_parallel_tool_calls_to_be_explicit(self, tokenizer):
        explicit_request = self._request(tokenizer, parallel_tool_calls=False)
        request = type(explicit_request).model_construct(
            _fields_set=explicit_request.model_fields_set - {"parallel_tool_calls"},
            **explicit_request.__dict__,
        )

        assert request.parallel_tool_calls is False
        assert "parallel_tool_calls" not in request.model_fields_set
        assert not prepost_module._should_build_tool_call_guidance(
            request,
            structural_tag_mode="on",
            structural_tag_scope="auto",
        )

    @pytest.mark.parametrize(
        "mode, tool_choice, scope, strict, parallel, expected",
        [
            ("off", "auto", "always", False, None, False),
            ("off", "none", "always", False, None, False),
            (
                "on",
                {"type": "function", "function": {"name": "get_weather"}},
                "auto",
                False,
                None,
                True,
            ),
            ("on", "none", "auto", False, None, False),
            ("on", "required", "auto", False, None, True),
            ("on", "auto", "always", False, None, True),
            ("on", "auto", "auto", True, None, True),
            ("on", "auto", "auto", False, False, True),
            ("on", "auto", "auto", False, None, False),
        ],
    )
    def test_runtime_policy_matrix(
        self,
        tokenizer,
        mode,
        tool_choice,
        scope,
        strict,
        parallel,
        expected,
    ):
        request = self._request(
            tokenizer,
            tool_choice=tool_choice,
            parallel_tool_calls=parallel,
            tools=[
                {
                    **TOOL_REQUEST["tools"][0],
                    "function": {
                        **TOOL_REQUEST["tools"][0]["function"],
                        "strict": strict,
                    },
                }
            ],
        )

        assert (
            prepost_module._should_build_tool_call_guidance(
                request,
                structural_tag_mode=mode,
                structural_tag_scope=scope,
            )
            is expected
        )


# ---------------------------------------------------------------------------
# _prepare_request: chat_template_kwargs forwarding
# ---------------------------------------------------------------------------


@pytest.mark.core
class TestChatTemplateKwargsForwarding:
    """chat_template_kwargs from the request are forwarded to ChatParams.

    Uses Qwen3 which supports enable_thinking: False to suppress <think> blocks.
    """

    @staticmethod
    def _messages():
        return [{"role": "user", "content": "Hello"}]

    def _prepare(self, request, tokenizer):
        """Return (chat_params, messages) from _prepare_request."""
        _, _, _, messages, chat_params = _prepare_request(
            request,
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        return chat_params, messages

    def _render(self, tokenizer, chat_params) -> str:
        """Render prompt text using the chat_params template kwargs."""
        kwargs = {**chat_params.chat_template_kwargs, "tokenize": False}
        return tokenizer.apply_chat_template(self._messages(), **kwargs)

    def test_qwen3_enable_thinking_true_no_closed_think_block(self, tokenizer):
        """enable_thinking=True leaves reasoning open (model generates <think> itself)."""
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer,
        )
        prompt = self._render(tokenizer, chat_params)
        assert "</think>" not in prompt

    def test_qwen3_thinking_flag_changes_tokens(self, tokenizer):
        """enable_thinking=True vs False produces different rendered prompts."""
        think_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer,
        )
        no_think_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": False},
            },
            tokenizer,
        )
        assert self._render(tokenizer, think_params) != self._render(
            tokenizer, no_think_params
        )

    def test_reasoning_effort_forwarded_to_template_kwargs(self, tokenizer):
        """reasoning_effort is always present in chat_params.chat_template_kwargs."""
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": "low",
            },
            tokenizer,
        )
        assert chat_params.chat_template_kwargs.get("reasoning_effort") == "low"

    def test_reasoning_effort_takes_precedence_over_deployment_default(self, tokenizer):
        _, _, chat_template_kwargs, _, _ = _prepare_request(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": "high",
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            default_thinking_mode="disabled",
        )

        assert chat_template_kwargs["reasoning_effort"] == "high"
        assert "thinking" not in chat_template_kwargs
        assert "enable_thinking" not in chat_template_kwargs
        assert "thinking_mode" not in chat_template_kwargs

    def test_default_thinking_mode_disabled_reaches_template_kwargs(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": self._messages(),
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            default_thinking_mode="disabled",
        )

        for kwargs in (chat_template_kwargs, chat_params.chat_template_kwargs):
            assert kwargs["thinking"] is False
            assert kwargs["enable_thinking"] is False
            assert kwargs["thinking_mode"] == "disabled"

    def test_default_thinking_mode_does_not_override_request_kwargs(self, tokenizer):
        _, _, chat_template_kwargs, _, chat_params = _prepare_request(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"enable_thinking": True},
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            default_thinking_mode="disabled",
        )

        for kwargs in (chat_template_kwargs, chat_params.chat_template_kwargs):
            assert kwargs["enable_thinking"] is True
            assert "thinking" not in kwargs
            assert "thinking_mode" not in kwargs

    def test_null_root_thinking_does_not_suppress_deployment_default(self, tokenizer):
        _, _, chat_template_kwargs, _, _ = _prepare_request(
            {
                "model": MODEL,
                "messages": self._messages(),
                "thinking": None,
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            default_thinking_mode="disabled",
        )

        assert chat_template_kwargs["enable_thinking"] is False


@pytest.mark.parametrize(
    ("runtime_config", "expected"),
    [
        ({"context_length": 1048576}, 1048576),
        ({}, None),
        ({"context_length": None}, None),
        ({"context_length": 0}, None),
        ({"context_length": -1}, None),
        ({"context_length": "1048576"}, None),
        ({"context_length": True}, None),
        (None, None),
    ],
)
def test_runtime_config_context_length(vllm_processor_module, runtime_config, expected):
    mdc = SimpleNamespace(runtime_config=lambda: runtime_config)

    assert vllm_processor_module._runtime_config_context_length(mdc) == expected


def test_runtime_config_structural_tag_options(vllm_processor_module):
    mdc = SimpleNamespace(
        runtime_config=lambda: {
            "structural_tag_mode": "on",
            "structural_tag_scope": "always",
            "structural_tag_schema": "strict",
        }
    )

    assert vllm_processor_module._runtime_config_structural_tag_options(mdc) == (
        "on",
        "always",
        "strict",
    )


# Regression: MistralTokenizer (--tokenizer-mode mistral) has no chat_template
# attribute; a direct read used to crash vLLM preprocessing.


def _make_mistral_tokenizer():
    # Bare instance: these tests only touch attribute access, not tokenization.
    from vllm.tokenizers.mistral import MistralTokenizer

    return MistralTokenizer.__new__(MistralTokenizer)


def test_mistral_tokenizer_has_no_chat_template_attribute():
    """Direct read raises; getattr is safe; the attribute is assignable."""
    tok = _make_mistral_tokenizer()

    with pytest.raises(AttributeError):
        _ = tok.chat_template

    assert getattr(tok, "chat_template", None) is None
    tok.chat_template = "x"
    assert tok.chat_template == "x"


def test_ensure_chat_template_mistral_no_crash(vllm_processor_module, tmp_path):
    """_ensure_chat_template leaves a MistralTokenizer untouched (no attribute)."""
    tok = _make_mistral_tokenizer()

    vllm_processor_module._ensure_chat_template(tok, str(tmp_path), None)

    assert not hasattr(tok, "chat_template")


def test_ensure_chat_template_mistral_ignores_on_disk_template(
    vllm_processor_module, tmp_path
):
    """A chat_template.jinja beside a Mistral model is not attached to it."""
    (tmp_path / "chat_template.jinja").write_text("{{ messages }}")
    tok = _make_mistral_tokenizer()

    vllm_processor_module._ensure_chat_template(tok, str(tmp_path), None)

    assert not hasattr(tok, "chat_template")


def test_ensure_chat_template_preserves_existing_hf_template(
    vllm_processor_module, tokenizer, tmp_path
):
    """An HF tokenizer's existing chat_template is left untouched."""
    existing = tokenizer.chat_template
    assert existing is not None

    vllm_processor_module._ensure_chat_template(tokenizer, str(tmp_path), None)

    assert tokenizer.chat_template == existing


class TestThinkingControlParity:  # FRONTEND.10
    # Keep vLLM's thinking kwargs aligned with the shared backend matrix.
    @pytest.mark.parametrize("case", THINKING_PARITY_CASES, ids=lambda case: case.name)
    def test_shared_thinking_policy(self, tokenizer, case):
        _, _, kwargs, _, _ = _prepare_request(
            {
                "model": MODEL,
                "messages": [{"role": "user", "content": "Hello"}],
                **case.request,
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        assert kwargs == case.expected
