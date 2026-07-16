#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import importlib.util
import json
from types import SimpleNamespace

import pytest
from _routed_engine_fakes import FakeRoutedEngine as _FakeRoutedEngine
from transformers import AutoTokenizer

from dynamo.frontend.prepost import _prepare_request

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


# Needs vllm packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
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


class TestReasoningParserMetadata:
    def test_no_reasoning_parser_returns_none(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        assert _build_reasoning_parser_metadata(
            None,
            object(),
            {},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
        ) == (None, None)

    def test_include_reasoning_false_marks_reasoning_ended(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class ParserShouldNotBeBuilt:
            def __init__(self, *args, **kwargs):
                raise AssertionError("parser should not be constructed")

        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            ParserShouldNotBeBuilt,
            object(),
            {"reasoning_effort": "low"},
            SimpleNamespace(include_reasoning=False),
            [1, 2, 3],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"reasoning_effort": "low"}}

    def test_parser_receives_chat_template_kwargs(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class FakeReasoningParser:
            def __init__(self, tokenizer, *, chat_template_kwargs):
                self.tokenizer = tokenizer
                self.chat_template_kwargs = chat_template_kwargs

            def is_reasoning_end(self, prompt_token_ids):
                return prompt_token_ids == [9, 9]

        tokenizer = object()
        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            FakeReasoningParser,
            tokenizer,
            {"reasoning_effort": "high"},
            SimpleNamespace(include_reasoning=True),
            [9, 9],
        )

        assert reasoning_ended is True
        assert parser_kwargs == {"chat_template_kwargs": {"reasoning_effort": "high"}}

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

    def _kwargs(self, request, tokenizer):
        """Return the chat_template_kwargs dict built by _prepare_request."""
        _, _, chat_template_kwargs, _, _ = _prepare_request(
            request,
            tokenizer=tokenizer,
            tool_parser_class=None,
        )
        return chat_template_kwargs

    def test_reasoning_effort_none_disables_enable_thinking(self, tokenizer):
        """reasoning_effort='none' derives enable_thinking=False."""
        kwargs = self._kwargs(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": "none",
            },
            tokenizer,
        )
        assert kwargs["enable_thinking"] is False

    def test_client_enable_thinking_wins_over_reasoning_effort(self, tokenizer):
        """An explicit chat_template_args enable_thinking is recovered from the
        raw dict and wins over the reasoning_effort-derived default."""
        kwargs = self._kwargs(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_args": {"enable_thinking": False},
                "reasoning_effort": "high",
            },
            tokenizer,
        )
        assert kwargs["enable_thinking"] is False

    def test_reasoning_effort_absent_omits_key(self, tokenizer):
        """No reasoning_effort leaves the key out (no None injection, no crash)."""
        kwargs = self._kwargs(
            {
                "model": MODEL,
                "messages": self._messages(),
            },
            tokenizer,
        )
        assert "reasoning_effort" not in kwargs

    def test_adaptive_mode_skips_reasoning_effort_toggle(self, tokenizer):
        """thinking_mode='adaptive' defers to the model: a grade forwards its
        value but derives no enable_thinking toggle."""
        kwargs = self._kwargs(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_args": {"thinking_mode": "adaptive"},
                "reasoning_effort": "high",
            },
            tokenizer,
        )
        assert "enable_thinking" not in kwargs
        assert kwargs["reasoning_effort"] == "high"


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
