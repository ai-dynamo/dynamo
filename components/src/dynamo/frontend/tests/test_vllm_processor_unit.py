#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for vLLM processor components.

Tests for the tool-stripping behaviour of _prepare_request when
tool_choice='none' and the exclude_tools_when_tool_choice_none flag.
"""

import json
from types import SimpleNamespace

import pytest
from _routed_engine_fakes import FakeRoutedEngine as _FakeRoutedEngine
from transformers import AutoTokenizer
from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.reasoning import ReasoningParserManager
from vllm.reasoning.gptoss_reasoning_parser import GptOssReasoningParser
from vllm.sampling_params import StructuredOutputsParams
from vllm.tool_parsers.gptoss_tool_parser import GptOssToolParser
from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser
from vllm.tool_parsers.qwen3_engine_tool_parser import Qwen3EngineToolParser

from dynamo.frontend.prepost import StreamingPostProcessor, _prepare_request

# Needs vllm packages (gpu_1 container), but does not allocate GPU VRAM.
pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.xpu_1,
    pytest.mark.pre_merge,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.timeout(180),  # 0-GiB unit tests, floor 180s
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


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    return AutoTokenizer.from_pretrained("openai/gpt-oss-20b", local_files_only=True)


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

    def test_backend_capability_forwards_kwargs_without_response_parser(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        assert _build_reasoning_parser_metadata(
            None,
            object(),
            {"thinking": True},
            SimpleNamespace(include_reasoning=True),
            [1, 2, 3],
            reasoning_aware_guided_decoding=True,
        ) == (None, {"chat_template_kwargs": {"thinking": True}})

    def test_include_reasoning_false_preserves_backend_reasoning_state(self):
        from dynamo.frontend.vllm_processor import _build_reasoning_parser_metadata

        class FakeReasoningParser:
            def __init__(self, tokenizer, *, chat_template_kwargs):
                pass

            def is_reasoning_end(self, prompt_token_ids):
                return False

        reasoning_ended, parser_kwargs = _build_reasoning_parser_metadata(
            FakeReasoningParser,
            object(),
            {"reasoning_effort": "low"},
            SimpleNamespace(include_reasoning=False),
            [1, 2, 3],
        )

        # include_reasoning controls response visibility, not whether the model
        # reasons. The backend must still delay a guided grammar until the
        # parser observes the real reasoning boundary.
        assert reasoning_ended is False
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

    @pytest.mark.parametrize("policy_name", ["basic", "nemotron_deci"])
    def test_non_force_alias_uses_explicit_marker_parser(self, tokenizer, policy_name):
        from dynamo.frontend.vllm_processor import _reasoning_parser_class

        parser_class = _reasoning_parser_class(policy_name, "qwen3")
        parser = parser_class(tokenizer, chat_template_kwargs={})

        assert parser.is_reasoning_end([]) is True
        assert parser.extract_reasoning(
            '[{"name":"get_weather","arguments":{}}]',
            SimpleNamespace(),
        ) == (None, '[{"name":"get_weather","arguments":{}}]')
        assert parser.extract_reasoning(
            "<think>reason</think>answer", SimpleNamespace()
        ) == ("reason", "answer")

        post = StreamingPostProcessor(
            tokenizer=tokenizer,
            request_for_sampling=SimpleNamespace(
                tool_choice="auto", include_reasoning=True
            ),
            sampling_params=SimpleNamespace(),
            prompt_token_ids=[parser.start_token_id],
            tool_parser=None,
            reasoning_parser_class=parser_class,
            chat_template_kwargs={},
        )
        completion = "reason</think>answer"
        choice = post.process_output(
            SimpleNamespace(
                text=completion,
                token_ids=tokenizer.encode(completion, add_special_tokens=False),
                index=0,
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice["delta"]["reasoning_content"] == "reason"
        assert choice["delta"]["content"] == "answer"

    @pytest.mark.parametrize("policy_name", ["basic", "nemotron_deci"])
    def test_non_force_alias_detects_generated_marker_from_plain_prompt(
        self, tokenizer, policy_name
    ):
        from dynamo.frontend.vllm_processor import _reasoning_parser_class

        parser_class = _reasoning_parser_class(policy_name, "qwen3")
        post = StreamingPostProcessor(
            tokenizer=tokenizer,
            request_for_sampling=SimpleNamespace(
                tool_choice="auto",
                include_reasoning=True,
                tools=None,
                messages=[],
            ),
            sampling_params=SimpleNamespace(),
            prompt_token_ids=[],
            tool_parser=None,
            reasoning_parser_class=parser_class,
            chat_template_kwargs={},
        )
        completion = "<think>reason</think>answer"
        choice = post.process_output(
            SimpleNamespace(
                text=completion,
                token_ids=tokenizer.encode(completion, add_special_tokens=False),
                index=0,
                finish_reason="stop",
                logprobs=None,
            )
        )

        assert choice["delta"]["reasoning_content"] == "reason"
        assert choice["delta"]["content"] == "answer"

    @pytest.mark.parametrize(
        ("policy_name", "native_name"),
        [
            ("minimax_append_think", "minimax_m2_append_think"),
            ("minimax_m2_append_think", "minimax_m2_append_think"),
        ],
    )
    def test_minimax_append_uses_implicit_start_parser(
        self, tokenizer, policy_name, native_name
    ):
        from dynamo.frontend.vllm_processor import _reasoning_parser_class

        parser_class = _reasoning_parser_class(policy_name, native_name)
        parser = parser_class(tokenizer, chat_template_kwargs={})

        assert parser.extract_reasoning(
            "implicit reason</think>answer", SimpleNamespace()
        ) == ("implicit reason", "answer")
        assert "<think>" not in "".join(
            part or ""
            for part in parser.extract_reasoning(
                "implicit reason</think>answer", SimpleNamespace()
            )
        )


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
        assert json.loads(envelope["comment"][0]) == {
            "input_tokens": 3,
            "output_tokens": 1,
            "chunk_tokens": 1,
        }


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
            tool_parser_class=Qwen3EngineToolParser,
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

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected"),
        [("none", False), ("low", True), ("high", True)],
    )
    def test_reasoning_effort_derives_enable_thinking(
        self, tokenizer, reasoning_effort, expected
    ):
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": reasoning_effort,
            },
            tokenizer,
        )

        assert chat_params.chat_template_kwargs["enable_thinking"] is expected

    @pytest.mark.parametrize(
        ("explicit_kwargs", "reasoning_effort"),
        [
            ({"enable_thinking": True}, "none"),
            ({"thinking": False}, "high"),
            ({"thinking_mode": "disabled"}, "high"),
        ],
    )
    def test_any_explicit_thinking_alias_overrides_reasoning_effort(
        self, tokenizer, explicit_kwargs, reasoning_effort
    ):
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "reasoning_effort": reasoning_effort,
                "chat_template_kwargs": explicit_kwargs,
            },
            tokenizer,
        )

        for key, value in explicit_kwargs.items():
            assert chat_params.chat_template_kwargs[key] == value
        absent_aliases = {
            "thinking",
            "enable_thinking",
            "thinking_mode",
        } - explicit_kwargs.keys()
        assert absent_aliases.isdisjoint(chat_params.chat_template_kwargs)

    def test_nested_reasoning_effort_survives_when_root_is_absent(self, tokenizer):
        chat_params, _ = self._prepare(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": {"reasoning_effort": "high"},
            },
            tokenizer,
        )

        assert chat_params.chat_template_kwargs["reasoning_effort"] == "high"

    @pytest.mark.parametrize(
        ("kwargs", "enabled"),
        [({}, True), ({"thinking": False}, False), ({"enable_thinking": True}, True)],
    )
    def test_deepseek_v4_normalizes_effective_thinking(
        self, tokenizer, kwargs, enabled
    ):
        _, _, normalized, _, _ = _prepare_request(
            {
                "model": MODEL,
                "messages": self._messages(),
                "chat_template_kwargs": kwargs,
            },
            tokenizer=tokenizer,
            tool_parser_class=None,
            reasoning_policy_name="deepseek_v4",
        )

        assert normalized["thinking"] is enabled
        assert normalized["enable_thinking"] is enabled


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


class TestGuidedReasoningResponsePolicy:
    class FakeReasoningParser:
        pass

    @staticmethod
    def _request(tool_choice, *, structural_tag=None, grammar_from_parser=False):
        request = SimpleNamespace(
            tool_choice=tool_choice,
            structured_outputs=SimpleNamespace(structural_tag=structural_tag),
        )
        request._grammar_from_tool_parser = grammar_from_parser
        return request

    @pytest.mark.parametrize(
        "tool_choice",
        [
            "required",
            {"type": "function", "function": {"name": "get_weather"}},
        ],
    )
    def test_force_parser_disabled_for_bare_guided_json(self, tool_choice):
        from dynamo.frontend.vllm_processor import _response_reasoning_parser_class

        assert (
            _response_reasoning_parser_class(
                self._request(tool_choice),
                self.FakeReasoningParser,
                reasoning_ended=False,
                reasoning_aware_guided_decoding=False,
            )
            is None
        )

    @pytest.mark.parametrize(
        ("request_obj", "reasoning_ended", "capability"),
        [
            (SimpleNamespace(tool_choice="auto"), False, False),
            (SimpleNamespace(tool_choice="required"), True, False),
            (SimpleNamespace(tool_choice="required"), False, True),
        ],
    )
    def test_parser_preserved_when_guidance_cannot_conflict(
        self, request_obj, reasoning_ended, capability
    ):
        from dynamo.frontend.vllm_processor import _response_reasoning_parser_class

        assert (
            _response_reasoning_parser_class(
                request_obj,
                self.FakeReasoningParser,
                reasoning_ended=reasoning_ended,
                reasoning_aware_guided_decoding=capability,
            )
            is self.FakeReasoningParser
        )

    @pytest.mark.parametrize("grammar_from_parser", [False, True])
    def test_parser_preserved_for_structural_or_custom_grammar(
        self, grammar_from_parser
    ):
        from dynamo.frontend.vllm_processor import _response_reasoning_parser_class

        request = self._request(
            "required",
            structural_tag=None if grammar_from_parser else {"type": "object"},
            grammar_from_parser=grammar_from_parser,
        )
        assert (
            _response_reasoning_parser_class(
                request,
                self.FakeReasoningParser,
                reasoning_ended=False,
                reasoning_aware_guided_decoding=False,
            )
            is self.FakeReasoningParser
        )

    def test_kimi_k25_tool_continuation_disables_frontend_and_backend_reasoning(
        self,
    ):
        from dynamo.frontend.vllm_processor import _request_reasoning_policy

        reasoning_ended, parser_class = _request_reasoning_policy(
            "kimi_k25",
            {
                "messages": [
                    {"role": "user", "content": "Run it"},
                    {"role": "tool", "content": "done"},
                ]
            },
            SimpleNamespace(tool_choice="auto"),
            self.FakeReasoningParser,
            reasoning_ended=False,
            reasoning_aware_guided_decoding=True,
        )

        assert reasoning_ended is True
        assert parser_class is None

    @pytest.mark.parametrize(
        ("reasoning_effort", "expected_ended", "expected_parser"),
        [
            (None, True, None),
            ("none", True, None),
            ("high", False, FakeReasoningParser),
        ],
    )
    def test_mistral_reasoning_effort_policy(
        self, reasoning_effort, expected_ended, expected_parser
    ):
        from dynamo.frontend.vllm_processor import _request_reasoning_policy

        request = {"messages": [{"role": "user", "content": "hello"}]}
        if reasoning_effort is not None:
            request["reasoning_effort"] = reasoning_effort
        reasoning_ended, parser_class = _request_reasoning_policy(
            "mistral",
            request,
            SimpleNamespace(tool_choice="auto"),
            self.FakeReasoningParser,
            reasoning_ended=False,
            reasoning_aware_guided_decoding=True,
        )

        assert reasoning_ended is expected_ended
        assert parser_class is expected_parser

    @pytest.mark.parametrize("parser_name", ["deepseek_r1", "minimax_m3"])
    def test_disabled_policy_marks_backend_reasoning_ended(self, parser_name):
        from dynamo.frontend.vllm_processor import _request_reasoning_policy

        reasoning_ended, parser_class = _request_reasoning_policy(
            parser_name,
            {"messages": [{"role": "user", "content": "hello"}]},
            SimpleNamespace(tool_choice="required"),
            self.FakeReasoningParser,
            reasoning_ended=False,
            reasoning_aware_guided_decoding=True,
            normalized_template_kwargs={
                "enable_thinking": False,
                "thinking_mode": "disabled",
            },
        )

        assert reasoning_ended is True
        assert parser_class is None


class TestVllmReasoningCapabilityOverride:
    @staticmethod
    def _mdc(reasoning_parser="nemotron_v3"):
        return SimpleNamespace(
            runtime_config=lambda: {
                "reasoning_parser": reasoning_parser,
                "runtime_data": {"reasoning_aware_guided_decoding": True},
            }
        )

    def test_compatible_frontend_alias_is_accepted(self):
        from dynamo.frontend.vllm_processor import (
            _validated_reasoning_aware_guided_decoding,
        )

        assert _validated_reasoning_aware_guided_decoding(self._mdc(), "nemotron_nano")

    def test_incompatible_frontend_override_is_rejected(self):
        from dynamo.frontend.vllm_processor import (
            _validated_reasoning_aware_guided_decoding,
        )

        with pytest.raises(ValueError, match="incompatible"):
            _validated_reasoning_aware_guided_decoding(self._mdc(), "qwen3")

    def test_missing_worker_parser_fails_closed(self):
        from dynamo.frontend.vllm_processor import (
            _validated_reasoning_aware_guided_decoding,
        )

        assert not _validated_reasoning_aware_guided_decoding(
            self._mdc(reasoning_parser=None), "nemotron_v3"
        )

    def test_factory_resolution_preserves_worker_kimi_policy_with_native_override(
        self,
    ):
        from dynamo.frontend.vllm_processor import _resolve_reasoning_parser_names

        policy_name, effective_name, native_name = _resolve_reasoning_parser_names(
            "kimi_k25", "kimi_k2"
        )

        assert policy_name == "kimi_k25"
        assert effective_name == "kimi_k2"
        assert native_name == "kimi_k2"


class _BoundaryToolParser:
    def extract_tool_calls_streaming(self, **kwargs):
        delta_text = kwargs["delta_text"]
        if delta_text == "TOOL":
            return DeltaMessage(
                tool_calls=[
                    DeltaToolCall(
                        index=0,
                        type="function",
                        id="call-1",
                        function=DeltaFunctionCall(
                            name="get_weather", arguments='{"city":"SF"}'
                        ),
                    )
                ]
            )
        if delta_text:
            return DeltaMessage(content=delta_text)
        return None

    def extract_tool_calls(self, text, request):
        del text, request
        return SimpleNamespace(
            tools_called=True,
            content="\n\n",
            tool_calls=[
                SimpleNamespace(
                    id="call-1",
                    function=SimpleNamespace(
                        name="get_weather", arguments='{"city":"SF"}'
                    ),
                )
            ],
        )


def _boundary_postprocessor(*, stream_response=True, tool_choice="auto"):
    return StreamingPostProcessor(
        tokenizer=SimpleNamespace(all_special_tokens=[]),
        request_for_sampling=SimpleNamespace(
            tool_choice=tool_choice, include_reasoning=True
        ),
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=_BoundaryToolParser(),
        reasoning_parser_class=None,
        chat_template_kwargs={},
        stream_response=stream_response,
    )


def _output(text, *, finish_reason=None):
    return SimpleNamespace(
        text=text,
        token_ids=[1] if text else [],
        index=0,
        finish_reason=finish_reason,
        logprobs=None,
    )


class TestVllmToolBoundaryContent:
    def test_streaming_tool_only_discards_separator_after_tool_delta(self):
        post = _boundary_postprocessor()

        assert post.process_output(_output("TOOL")) is None
        tool_choice = post.process_output(_output("\n"))
        finish_choice = post.process_output(_output("\n", finish_reason="stop"))

        assert tool_choice["delta"].get("tool_calls")
        assert "content" not in tool_choice["delta"]
        assert finish_choice["finish_reason"] == "tool_calls"
        assert "content" not in finish_choice["delta"]

    def test_streaming_direct_answer_preserves_leading_whitespace(self):
        post = _boundary_postprocessor()

        assert post.process_output(_output("\n")) is None
        choice = post.process_output(_output("Blue", finish_reason="stop"))

        assert choice["delta"]["content"] == "\nBlue"
        assert choice["finish_reason"] == "stop"

    def test_streaming_narration_before_tool_preserves_trailing_space(self):
        post = _boundary_postprocessor()

        narration = post.process_output(_output("I will call"))
        separator = post.process_output(_output(" "))
        tool_choice = post.process_output(_output("TOOL", finish_reason="stop"))

        assert narration["delta"]["content"] == "I will call"
        assert separator["delta"]["content"] == " "
        assert tool_choice["delta"].get("tool_calls")
        assert "content" not in tool_choice["delta"]

    def test_non_streaming_tool_only_discards_parser_separator(self):
        post = _boundary_postprocessor(stream_response=False)

        choice = post.process_output(_output("ignored", finish_reason="stop"))

        assert choice["delta"].get("tool_calls")
        assert "content" not in choice["delta"]
        assert choice["finish_reason"] == "tool_calls"

    def test_length_finish_is_preserved_after_tool_delta(self):
        post = _boundary_postprocessor()

        choice = post.process_output(_output("TOOL", finish_reason="length"))

        assert choice["delta"].get("tool_calls")
        assert choice["finish_reason"] == "length"

    def test_named_tool_stop_finish_is_remapped(self):
        post = _boundary_postprocessor(
            tool_choice={
                "type": "function",
                "function": {"name": "get_weather"},
            }
        )

        choice = post.process_output(_output("TOOL", finish_reason="stop"))

        assert choice["delta"].get("tool_calls")
        assert choice["finish_reason"] == "tool_calls"


def _guided_json_postprocessor(
    tokenizer,
    tool_parser_class,
    *,
    tool_choice="required",
    reasoning_parser_class=None,
    prompt_token_ids=(),
    stream_response=True,
    tool_call_id_type="random",
):
    request = {**TOOL_REQUEST, "tool_choice": tool_choice}
    request_for_sampling, tool_parser, kwargs, _, _ = _prepare_request(
        request,
        tokenizer=tokenizer,
        tool_parser_class=tool_parser_class,
        reasoning_parser_class=reasoning_parser_class,
    )
    # Exercise the standard bare-JSON response shape independently of strict
    # structural-tag mode. Backends and parser versions may select either.
    request_for_sampling.structured_outputs = StructuredOutputsParams(
        json={"type": "array"}
    )
    return StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=list(prompt_token_ids),
        tool_parser=tool_parser,
        reasoning_parser_class=reasoning_parser_class,
        chat_template_kwargs=kwargs,
        stream_response=stream_response,
        tool_call_id_type=tool_call_id_type,
    )


@pytest.mark.parametrize(
    "tool_parser_class", [Hermes2ProToolParser, Qwen3EngineToolParser]
)
@pytest.mark.parametrize("stream_response", [False, True])
def test_required_standard_guided_json_uses_generic_parser(
    tokenizer, tool_parser_class, stream_response
):
    post = _guided_json_postprocessor(
        tokenizer,
        tool_parser_class,
        stream_response=stream_response,
    )
    text = '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    choice = post.process_output(
        SimpleNamespace(
            text=text,
            token_ids=tokenizer.encode(text, add_special_tokens=False),
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert choice["finish_reason"] == "tool_calls"
    assert "content" not in choice["delta"]
    tool_call = choice["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "SF"}


def test_required_standard_guided_json_large_delta_has_no_array_suffix(tokenizer):
    post = _guided_json_postprocessor(
        tokenizer,
        Hermes2ProToolParser,
        stream_response=True,
    )
    text = '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    choice = post.process_output(
        SimpleNamespace(
            text=text,
            token_ids=tokenizer.encode(text, add_special_tokens=False),
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    arguments = choice["delta"]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(arguments) == {"city": "SF"}
    assert not arguments.endswith("]")


def test_required_standard_guided_json_streams_split_name_and_arguments(tokenizer):
    post = _guided_json_postprocessor(
        tokenizer,
        Hermes2ProToolParser,
        stream_response=True,
    )
    chunks = (
        ('[{"name":"get_', None),
        ('weather","parameters":{"city":"', None),
        ('SF"}}]', "stop"),
    )

    choices = [
        post.process_output(
            SimpleNamespace(
                text=text,
                token_ids=tokenizer.encode(text, add_special_tokens=False),
                index=0,
                finish_reason=finish_reason,
                logprobs=None,
            )
        )
        for text, finish_reason in chunks
    ]
    calls = [choice["delta"]["tool_calls"][0] for choice in choices]

    assert calls[0]["id"].startswith("chatcmpl-tool-")
    assert calls[0]["type"] == "function"
    assert "id" not in calls[1]
    assert "type" not in calls[1]
    assert "".join(call["function"].get("name", "") for call in calls) == (
        "get_weather"
    )
    assert "".join(call["function"].get("arguments", "") for call in calls) == (
        '{"city":"SF"}'
    )
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_required_standard_guided_json_waits_for_name_derived_id(tokenizer):
    post = _guided_json_postprocessor(
        tokenizer,
        Hermes2ProToolParser,
        stream_response=True,
        tool_call_id_type="kimi_k2",
    )
    partial_name = '[{"name":"get_'
    complete_name = 'weather","parameters":{'

    assert (
        post.process_output(
            SimpleNamespace(
                text=partial_name,
                token_ids=tokenizer.encode(partial_name, add_special_tokens=False),
                index=0,
                finish_reason=None,
                logprobs=None,
            )
        )
        is None
    )
    choice = post.process_output(
        SimpleNamespace(
            text=complete_name,
            token_ids=tokenizer.encode(complete_name, add_special_tokens=False),
            index=0,
            finish_reason="length",
            logprobs=None,
        )
    )

    tool_call = choice["delta"]["tool_calls"][0]
    assert tool_call["id"] == "functions.get_weather:0"
    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == "{"
    assert choice["finish_reason"] == "length"


def test_named_standard_guided_json_streams_nested_arguments(tokenizer):
    post = _guided_json_postprocessor(
        tokenizer,
        Hermes2ProToolParser,
        tool_choice={
            "type": "function",
            "function": {"name": "get_weather"},
        },
        stream_response=True,
    )
    chunks = (
        ('{"nested":{"values":[1,', None),
        ('2]},"city":"S', None),
        ('F"}', "stop"),
    )

    choices = [
        post.process_output(
            SimpleNamespace(
                text=text,
                token_ids=tokenizer.encode(text, add_special_tokens=False),
                index=0,
                finish_reason=finish_reason,
                logprobs=None,
            )
        )
        for text, finish_reason in chunks
    ]
    calls = [choice["delta"]["tool_calls"][0] for choice in choices]

    assert calls[0]["function"]["name"] == "get_weather"
    assert all("name" not in call["function"] for call in calls[1:])
    arguments = "".join(call["function"].get("arguments", "") for call in calls)
    assert json.loads(arguments) == {
        "nested": {"values": [1, 2]},
        "city": "SF",
    }
    assert choices[-1]["finish_reason"] == "tool_calls"


def test_required_standard_guided_json_nested_array_and_multiple_calls(tokenizer):
    post = _guided_json_postprocessor(
        tokenizer,
        Hermes2ProToolParser,
        stream_response=True,
    )
    first = '[{"name":"get_weather","parameters":{"xs":[1,2]'
    second = '}},{"name":"get_weather","parameters":{"city":"SF"}}]'

    first_choice = post.process_output(
        SimpleNamespace(
            text=first,
            token_ids=tokenizer.encode(first, add_special_tokens=False),
            index=0,
            finish_reason=None,
            logprobs=None,
        )
    )
    tool_choice = post.process_output(
        SimpleNamespace(
            text=second,
            token_ids=tokenizer.encode(second, add_special_tokens=False),
            index=0,
            finish_reason=None,
            logprobs=None,
        )
    )
    finish_choice = post.process_output(
        SimpleNamespace(
            text="",
            token_ids=[],
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    first_call = first_choice["delta"]["tool_calls"][0]
    calls = tool_choice["delta"]["tool_calls"]
    assert first_call["index"] == calls[0]["index"] == 0
    assert json.loads(
        first_call["function"]["arguments"] + calls[0]["function"]["arguments"]
    ) == {"xs": [1, 2]}
    assert calls[1]["index"] == 1
    assert json.loads(calls[1]["function"]["arguments"]) == {"city": "SF"}
    assert finish_choice["finish_reason"] == "tool_calls"
    assert "tool_calls" not in finish_choice["delta"]


def test_required_standard_guided_json_preserves_partial_call_on_length(tokenizer):
    post = _guided_json_postprocessor(
        tokenizer,
        Hermes2ProToolParser,
        stream_response=True,
    )
    first = '[{"name":"get_wea'
    second = 'ther","parameters":{"city":"S'

    first_choice = post.process_output(
        SimpleNamespace(
            text=first,
            token_ids=tokenizer.encode(first, add_special_tokens=False),
            index=0,
            finish_reason=None,
            logprobs=None,
        )
    )
    length_choice = post.process_output(
        SimpleNamespace(
            text=second,
            token_ids=tokenizer.encode(second, add_special_tokens=False),
            index=0,
            finish_reason="length",
            logprobs=None,
        )
    )

    calls = [
        first_choice["delta"]["tool_calls"][0],
        length_choice["delta"]["tool_calls"][0],
    ]
    assert "".join(call["function"].get("name", "") for call in calls) == (
        "get_weather"
    )
    assert "".join(call["function"].get("arguments", "") for call in calls) == (
        '{"city":"S'
    )
    assert length_choice["finish_reason"] == "length"


def test_nemotron_v3_streams_reasoning_before_required_guided_json(tokenizer):
    reasoning_parser_class = ReasoningParserManager.get_reasoning_parser("nemotron_v3")
    post = _guided_json_postprocessor(
        tokenizer,
        Qwen3EngineToolParser,
        reasoning_parser_class=reasoning_parser_class,
        stream_response=True,
    )
    reasoning_text = "I should check the requested city.</think>"
    reasoning_choice = post.process_output(
        SimpleNamespace(
            text=reasoning_text,
            token_ids=tokenizer.encode(reasoning_text, add_special_tokens=False),
            index=0,
            finish_reason=None,
            logprobs=None,
        )
    )
    json_text = '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    tool_choice = post.process_output(
        SimpleNamespace(
            text=json_text,
            token_ids=tokenizer.encode(json_text, add_special_tokens=False),
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert reasoning_choice["delta"]["reasoning_content"] == (
        "I should check the requested city."
    )
    assert "content" not in reasoning_choice["delta"]
    assert tool_choice["delta"]["tool_calls"][0]["function"]["name"] == ("get_weather")
    assert "content" not in tool_choice["delta"]
    assert tool_choice["finish_reason"] == "tool_calls"


def test_minimax_append_streams_reasoning_before_required_guided_json(tokenizer):
    from dynamo.frontend.vllm_processor import _reasoning_parser_class

    reasoning_parser_class = _reasoning_parser_class(
        "minimax_append_think", "minimax_m2_append_think"
    )
    post = _guided_json_postprocessor(
        tokenizer,
        Qwen3EngineToolParser,
        reasoning_parser_class=reasoning_parser_class,
        stream_response=True,
    )
    reasoning_text = "I should inspect the request.</think>"
    reasoning_choice = post.process_output(
        SimpleNamespace(
            text=reasoning_text,
            token_ids=tokenizer.encode(reasoning_text, add_special_tokens=False),
            index=0,
            finish_reason=None,
            logprobs=None,
        )
    )
    json_text = '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    tool_choice = post.process_output(
        SimpleNamespace(
            text=json_text,
            token_ids=tokenizer.encode(json_text, add_special_tokens=False),
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert reasoning_choice["delta"] == {
        "role": "assistant",
        "reasoning_content": "I should inspect the request.",
    }
    assert tool_choice["delta"]["tool_calls"][0]["function"]["name"] == ("get_weather")
    assert "content" not in tool_choice["delta"]
    assert tool_choice["finish_reason"] == "tool_calls"


def test_minimax_append_auto_direct_answer_splits_reasoning_and_content(tokenizer):
    from dynamo.frontend.vllm_processor import _reasoning_parser_class

    reasoning_parser_class = _reasoning_parser_class(
        "minimax_append_think", "minimax_m2_append_think"
    )
    request_for_sampling, tool_parser, kwargs, _, _ = _prepare_request(
        {**TOOL_REQUEST, "tool_choice": "auto"},
        tokenizer=tokenizer,
        tool_parser_class=Qwen3EngineToolParser,
        reasoning_parser_class=reasoning_parser_class,
    )
    post = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=SimpleNamespace(),
        prompt_token_ids=[],
        tool_parser=tool_parser,
        reasoning_parser_class=reasoning_parser_class,
        chat_template_kwargs=kwargs,
        stream_response=True,
    )
    choice = post.process_output(
        SimpleNamespace(
            text="Think first.</think>Visible answer",
            token_ids=tokenizer.encode(
                "Think first.</think>Visible answer", add_special_tokens=False
            ),
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert choice["delta"]["reasoning_content"] == "Think first."
    assert choice["delta"]["content"] == "Visible answer"
    assert "tool_calls" not in choice["delta"]


def test_required_guided_json_honors_include_reasoning_false(tokenizer):
    reasoning_parser_class = ReasoningParserManager.get_reasoning_parser("nemotron_v3")
    post = _guided_json_postprocessor(
        tokenizer,
        Qwen3EngineToolParser,
        reasoning_parser_class=reasoning_parser_class,
        stream_response=True,
    )
    post.request_for_sampling.include_reasoning = False
    reasoning_text = "Hidden reasoning.</think>"
    assert (
        post.process_output(
            SimpleNamespace(
                text=reasoning_text,
                token_ids=tokenizer.encode(reasoning_text, add_special_tokens=False),
                index=0,
                finish_reason=None,
                logprobs=None,
            )
        )
        is None
    )
    json_text = '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    choice = post.process_output(
        SimpleNamespace(
            text=json_text,
            token_ids=tokenizer.encode(json_text, add_special_tokens=False),
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert "reasoning_content" not in choice["delta"]
    assert choice["delta"]["tool_calls"]


@pytest.mark.parametrize("stream_response", [False, True])
def test_harmony_reasoning_then_required_guided_json(
    gpt_oss_tokenizer, stream_response
):
    post = _guided_json_postprocessor(
        gpt_oss_tokenizer,
        GptOssToolParser,
        reasoning_parser_class=GptOssReasoningParser,
        stream_response=stream_response,
    )
    raw_text = (
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    )
    token_ids = gpt_oss_tokenizer.encode(raw_text, add_special_tokens=False) + [
        gpt_oss_tokenizer.eos_token_id
    ]
    # vLLM retains a terminal stop ID while its output text has special
    # Harmony markers and the stop token stripped.
    text = gpt_oss_tokenizer.decode(token_ids, skip_special_tokens=True)
    choice = post.process_output(
        SimpleNamespace(
            text=text,
            token_ids=token_ids,
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert choice["delta"]["reasoning_content"] == "Need weather."
    assert "content" not in choice["delta"]
    tool_call = choice["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert json.loads(tool_call["function"]["arguments"]) == {"city": "SF"}
    assert choice["finish_reason"] == "tool_calls"


def test_harmony_rejects_reasoning_effort_none(gpt_oss_tokenizer):
    with pytest.raises(ValueError, match="not supported for Harmony"):
        _prepare_request(
            {
                **TOOL_REQUEST,
                "reasoning_effort": "none",
                "tool_choice": "required",
            },
            tokenizer=gpt_oss_tokenizer,
            tool_parser_class=GptOssToolParser,
            reasoning_parser_class=GptOssReasoningParser,
            reasoning_policy_name="gpt_oss",
        )


def test_harmony_streams_reasoning_before_guided_json(gpt_oss_tokenizer):
    post = _guided_json_postprocessor(
        gpt_oss_tokenizer,
        GptOssToolParser,
        reasoning_parser_class=GptOssReasoningParser,
        stream_response=True,
    )
    reasoning_raw = (
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
    )
    reasoning_ids = gpt_oss_tokenizer.encode(reasoning_raw, add_special_tokens=False)
    reasoning_choice = post.process_output(
        SimpleNamespace(
            text=gpt_oss_tokenizer.decode(reasoning_ids, skip_special_tokens=True),
            token_ids=reasoning_ids,
            index=0,
            finish_reason=None,
            logprobs=None,
        )
    )
    json_raw = '[{"name":"get_weather","parameters":{"city":"SF"}}]'
    json_ids = gpt_oss_tokenizer.encode(json_raw, add_special_tokens=False) + [
        gpt_oss_tokenizer.eos_token_id
    ]
    tool_choice = post.process_output(
        SimpleNamespace(
            text=gpt_oss_tokenizer.decode(json_ids, skip_special_tokens=True),
            token_ids=json_ids,
            index=0,
            finish_reason="stop",
            logprobs=None,
        )
    )

    assert reasoning_choice["delta"]["reasoning_content"] == "Need weather."
    assert "tool_calls" not in reasoning_choice["delta"]
    assert tool_choice["delta"]["tool_calls"][0]["function"]["name"] == ("get_weather")
    assert "reasoning_content" not in tool_choice["delta"]


@pytest.mark.parametrize("stream_response", [False, True])
def test_harmony_truncated_reasoning_is_not_dropped_or_leaked(
    gpt_oss_tokenizer, stream_response
):
    post = _guided_json_postprocessor(
        gpt_oss_tokenizer,
        GptOssToolParser,
        reasoning_parser_class=GptOssReasoningParser,
        stream_response=stream_response,
    )
    raw = "<|channel|>analysis<|message|>Need more time"
    token_ids = gpt_oss_tokenizer.encode(raw, add_special_tokens=False)
    choice = post.process_output(
        SimpleNamespace(
            text=gpt_oss_tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=token_ids,
            index=0,
            finish_reason="length",
            logprobs=None,
        )
    )

    assert choice["delta"]["reasoning_content"] == "Need more time"
    assert "content" not in choice["delta"]
    assert "tool_calls" not in choice["delta"]
    assert choice["finish_reason"] == "length"


def test_harmony_truncated_guided_json_does_not_leak_content(gpt_oss_tokenizer):
    post = _guided_json_postprocessor(
        gpt_oss_tokenizer,
        GptOssToolParser,
        reasoning_parser_class=GptOssReasoningParser,
        stream_response=True,
    )
    raw = (
        "<|channel|>analysis<|message|>Need weather.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        '[{"name":"get_weather","parameters":{"city":"S'
    )
    token_ids = gpt_oss_tokenizer.encode(raw, add_special_tokens=False)
    choice = post.process_output(
        SimpleNamespace(
            text=gpt_oss_tokenizer.decode(token_ids, skip_special_tokens=True),
            token_ids=token_ids,
            index=0,
            finish_reason="length",
            logprobs=None,
        )
    )

    assert choice["delta"]["reasoning_content"] == "Need weather."
    assert "content" not in choice["delta"]
    tool_call = choice["delta"]["tool_calls"][0]
    assert tool_call["function"]["name"] == "get_weather"
    assert tool_call["function"]["arguments"] == '{"city":"S'
    assert choice["finish_reason"] == "length"
