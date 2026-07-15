# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import re as re_mod
from dataclasses import dataclass
from typing import Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip(
        "Skipping to avoid errors during collection with '-m gpu_0'. "
        "CUDA/GPU not available, but tensorrt_llm import and the test require GPU.",
        allow_module_level=True,
    )
from tensorrt_llm.executor.request import DEFAULT_REQUEST_PRIORITY
from tensorrt_llm.llmapi import DisaggregatedParams

from dynamo.llm.exceptions import EngineShutdown
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import TrtllmHealthCheckPayload
from dynamo.trtllm.multimodal_processor import MultimodalRequestProcessor
from dynamo.trtllm.request_handlers.handler_base import (
    BYPASS_REMOTE_PREFILL_ANNOTATION,
    HandlerBase,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
]

# Intentionally omit profiled_vram_gib so this import-heavy module runs in the
# sequential GPU stage instead of starting one TRT-LLM subprocess per test node.


@dataclass
class MockSamplingParams:
    """Mock sampling params object for testing."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    seed: int | None = None
    n: int | None = None
    best_of: int = 1
    ignore_eos: bool = False
    guided_decoding: object | None = None
    max_tokens: int | None = None
    min_tokens: int | None = None
    stop_token_ids: list[int] | None = None

    def __post_init__(self):
        """Called after dataclass initialization (including via replace())."""
        if self.n is not None and self.best_of < self.n:
            raise ValueError(
                f"best_of ({self.best_of}) cannot be less than n ({self.n})"
            )


class TestOverrideSamplingParams:
    """Tests for _override_sampling_params method.

    The key bug fix being tested: using `if value is None` instead of `if not value`
    ensures that falsy values like 0, False, and "" are correctly applied.
    """

    def test_falsy_values_are_applied(self):
        """Test that falsy values (0, False) are correctly set.

        This is the main regression test for the bug fix. Previously, using
        `if not value` would skip setting values like 0 or False.
        """
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "temperature": 0,  # Falsy but valid - should be set
                "top_k": 0,  # Falsy but valid - should be set
                "ignore_eos": False,  # Falsy but valid - should be set
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.temperature == 0
        assert result.top_k == 0
        assert result.ignore_eos is False

    def test_none_values_are_skipped(self):
        """Test that None values do not override existing params."""
        sampling_params = MockSamplingParams()
        original_temperature = sampling_params.temperature
        original_top_p = sampling_params.top_p

        request = {
            "sampling_options": {
                "temperature": None,
                "top_p": None,
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.temperature == original_temperature
        assert result.top_p == original_top_p

    def test_disabled_top_k_sentinel_is_converted(self):
        sampling_params = MockSamplingParams()
        request = {"sampling_options": {"top_k": -1}}

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.top_k == 0

    def test_truthy_values_are_applied(self):
        """Test that normal truthy values are correctly set."""
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 40,
                "seed": 42,
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.temperature == 0.7
        assert result.top_p == 0.9
        assert result.top_k == 40
        assert result.seed == 42

    def test_n_is_passed_through(self):
        """Test that n is passed through to sampling params."""
        sampling_params = MockSamplingParams()
        request = {"sampling_options": {"n": 2}}

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.n == 2
        assert result.best_of == 2

    def test_unknown_attributes_raise_error(self):
        """Test that unknown attributes raise a TypeError.

        dataclasses.replace() does not accept unknown field names.
        """
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "nonexistent_param": 123,
            }
        }

        with pytest.raises(TypeError):
            HandlerBase._override_sampling_params(sampling_params, request)

    def test_mixed_values(self):
        """Test a mix of None, falsy, and truthy values."""
        sampling_params = MockSamplingParams()
        original_top_p = sampling_params.top_p

        request = {
            "sampling_options": {
                "temperature": 0,  # Falsy - should be set
                "top_p": None,  # None - should be skipped
                "top_k": 100,  # Truthy - should be set
                "seed": 0,  # Falsy - should be set
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.temperature == 0
        assert result.top_p == original_top_p  # Unchanged
        assert result.top_k == 100
        assert result.seed == 0

    def test_unsupported_fields_raise(self):
        sampling_params = MockSamplingParams()
        request = {"sampling_options": {"non_existent_param": 123}}

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            _ = HandlerBase._override_sampling_params(sampling_params, request)

    def test_post_init_called_when_overriding(self):
        # This allows us to check that potential validation logic in `__post_init__` is run when
        # overriding the sampling params with what comes from the requests.
        sampling_params = MockSamplingParams()
        request = {"sampling_options": {"temperature": 0.5}}

        with mock.patch.object(MockSamplingParams, "__post_init__") as mock_post_init:
            HandlerBase._override_sampling_params(sampling_params, request)

        mock_post_init.assert_called_once()


class TestGuidedDecodingFromToolChoice:
    """Tests that guided_decoding dicts from Rust are converted to GuidedDecodingParams.

    The Rust frontend serializes guided_decoding as a plain dict over TCP.
    _override_sampling_params must convert it to a GuidedDecodingParams
    object before passing to TRT-LLM, which expects attribute access
    (e.g. .json_object, .json) on the guided_decoding field.
    """

    # Matches what the Rust frontend serializes when
    # tool_choice="required" with a single tool definition.
    GUIDED_DECODING_DICT = {
        "json": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "anyOf": [
                    {
                        "properties": {
                            "name": {"type": "string", "enum": ["get_weather"]},
                            "parameters": {
                                "type": "object",
                                "properties": {"location": {"type": "string"}},
                                "required": ["location"],
                            },
                        },
                        "required": ["name", "parameters"],
                    }
                ],
            },
        }
    }

    def test_guided_decoding_dict_is_converted(self):
        """guided_decoding dict from Rust must be converted to GuidedDecodingParams.

        The Rust frontend serializes GuidedDecodingOptions as a JSON dict.
        _override_sampling_params must convert it to TRT-LLM's
        GuidedDecodingParams so that downstream attribute access like
        .json_object works without AttributeError.
        """
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "temperature": 0.7,
                "guided_decoding": self.GUIDED_DECODING_DICT,
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert not isinstance(
            result.guided_decoding, dict
        ), "guided_decoding should be converted from dict to GuidedDecodingParams"
        # Downstream code (TRT-LLM sampling_params.py) accesses these attributes:
        assert result.guided_decoding.json_object is False
        assert result.guided_decoding.json == self.GUIDED_DECODING_DICT["json"]

    def test_choice_converted_to_regex(self):
        """guided_decoding with 'choice' must be converted to a regex pattern.

        TRT-LLM's GuidedDecodingParams doesn't have a 'choice' field.
        The handler should convert choice=["yes", "no", "maybe"] to
        regex="(yes|no|maybe)" so that GuidedDecodingParams can enforce it.
        """
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "guided_decoding": {
                    "choice": ["yes", "no", "maybe"],
                },
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert not isinstance(result.guided_decoding, dict)
        assert result.guided_decoding.regex == "(yes|no|maybe)"
        assert result.guided_decoding.json is None

    def test_choice_with_special_chars_escaped(self):
        """Choice values with regex special characters must be escaped."""
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "guided_decoding": {
                    "choice": ["yes (confirmed)", "no [rejected]"],
                },
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert not isinstance(result.guided_decoding, dict)
        expected = (
            "("
            + "|".join(re_mod.escape(c) for c in ["yes (confirmed)", "no [rejected]"])
            + ")"
        )
        assert result.guided_decoding.regex == expected

    def test_choice_not_used_when_regex_present(self):
        """If both choice and regex are specified, regex takes priority."""
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "guided_decoding": {
                    "choice": ["yes", "no"],
                    "regex": "[0-9]+",
                },
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.guided_decoding.regex == "[0-9]+"

    def test_empty_choice_ignored(self):
        """Empty choice list should not produce a regex."""
        sampling_params = MockSamplingParams()
        request: dict[str, Any] = {
            "sampling_options": {
                "guided_decoding": {
                    "choice": [],
                },
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.guided_decoding.regex is None

    def test_choice_with_none_items_filtered(self):
        """Choice list with None items should filter them out."""
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "guided_decoding": {
                    "choice": [None, "yes", None, "no"],
                },
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert not isinstance(result.guided_decoding, dict)
        assert result.guided_decoding.regex == "(yes|no)"

    def test_choice_all_none_items_no_regex(self):
        """Choice list with all None items should not produce a regex."""
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "guided_decoding": {
                    "choice": [None, None],
                },
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        assert result.guided_decoding.regex is None


class _ConcreteHandler(HandlerBase):
    """Concrete subclass of HandlerBase for testing (satisfies abstract method)."""

    async def generate(self, *args, **kwargs):
        raise NotImplementedError


class TestDeferredAbortGuard:
    """Tests for _DeferredAbort in disaggregated decode cancellation.

    In disaggregated serving, decode abort must be deferred until the first
    generation result is received (indicating KV cache transfer is complete).
    _DeferredAbort wraps GenerationResult.abort() to spawn a background task
    that waits for the first token before calling real abort.
    """

    def _make_handler(self) -> HandlerBase:
        config = MagicMock()
        config.shutdown_event = None
        config.conversation_affinity = False
        return _ConcreteHandler(config)

    @pytest.mark.asyncio
    async def test_deferred_abort_before_first_token(self):
        """abort() before first token should NOT call real abort immediately."""
        from dynamo.trtllm.request_handlers.handler_base import _DeferredAbort

        generation_result = MagicMock()
        # Make generation_result iterable (background task will try to read it)
        generation_result.__aiter__ = MagicMock(return_value=generation_result)
        never_resolve = asyncio.get_event_loop().create_future()
        generation_result.__anext__ = MagicMock(return_value=never_resolve)

        guard = _DeferredAbort(generation_result)
        guard.abort()

        # Real abort should NOT have been called — deferred to background task
        generation_result.abort.assert_not_called()

    @pytest.mark.asyncio
    async def test_deferred_abort_after_first_token(self):
        """abort() after signal_first_token should call real abort immediately."""
        from dynamo.trtllm.request_handlers.handler_base import _DeferredAbort

        generation_result = MagicMock()
        guard = _DeferredAbort(generation_result)

        guard.signal_first_token()
        guard.abort()

        generation_result.abort.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_deferred_task_completes(self):
        """Background task should call abort after first result from generation_result."""
        from dynamo.trtllm.request_handlers.handler_base import _DeferredAbort

        generation_result = MagicMock()
        result_queue = asyncio.Queue()

        async def mock_anext(self_mock):
            val = await result_queue.get()
            if val is StopAsyncIteration:
                raise StopAsyncIteration
            return val

        generation_result.__aiter__ = MagicMock(return_value=generation_result)
        generation_result.__anext__ = lambda self: mock_anext(self)

        guard = _DeferredAbort(generation_result)
        guard.abort()  # Spawns background task

        generation_result.abort.assert_not_called()

        # Simulate first result arriving (KV transfer complete)
        await result_queue.put("first_token")
        await asyncio.sleep(0.05)  # Let background task run

        generation_result.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_guard_in_non_disagg_mode(self):
        """Without _DeferredAbort wrapper, abort fires immediately on cancel."""
        handler = self._make_handler()
        generation_result = MagicMock()
        context = MagicMock()
        killed_future = asyncio.get_event_loop().create_future()
        killed_future.set_result(None)
        context.async_killed_or_stopped.return_value = killed_future
        context.id.return_value = "test-no-guard"

        # Pass real generation_result (no wrapper) — non-disagg path
        await handler._handle_cancellation(generation_result, context)
        generation_result.abort.assert_called_once()

    @pytest.mark.asyncio
    @pytest.mark.timeout(5)
    async def test_shutdown_calls_abort_directly(self):
        """Shutdown calls abort on whatever is passed (wrapper or real), immediately."""
        handler = self._make_handler()
        handler.shutdown_event = asyncio.Event()

        # Pass a _DeferredAbort wrapper — shutdown should still call .abort()
        from dynamo.trtllm.request_handlers.handler_base import _DeferredAbort

        generation_result = MagicMock()
        guard = _DeferredAbort(generation_result)

        context = MagicMock()
        killed_future = asyncio.get_event_loop().create_future()
        context.async_killed_or_stopped.return_value = killed_future
        context.id.return_value = "test-shutdown"

        task = asyncio.create_task(handler._handle_cancellation(guard, context))
        await asyncio.sleep(0.05)

        # Trigger shutdown
        handler.shutdown_event.set()

        with pytest.raises(EngineShutdown):
            await task
        # Shutdown calls guard.abort() → since no first token, spawns background task
        # The important thing is EngineShutdown is raised and abort path is entered


class TestMultimodalGuard:
    """Tests for multimodal guard when --modality multimodal is not configured."""

    IMAGE_MESSAGE = {
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "http://example.com/a.jpg"}},
            {"type": "text", "text": "describe image"},
        ],
    }

    def _make_handler(self, multimodal_processor=None) -> HandlerBase:
        config = MagicMock()
        config.multimodal_processor = multimodal_processor
        config.shutdown_event = None
        config.conversation_affinity = False
        return _ConcreteHandler(config)

    async def _prepare(self, handler, request, epd_metadata=None):
        return await handler._prepare_input_for_generation(
            request=request,
            embeddings=None,
            ep_disaggregated_params=None,
            epd_metadata=epd_metadata or {},
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "request_factory",
        [
            lambda msg: {"token_ids": [1, 2, 3], "extra_args": {"messages": [msg]}},
            lambda msg: {"token_ids": [1, 2, 3], "messages": [msg]},
        ],
        ids=["extra_args_messages", "top_level_messages"],
    )
    async def test_raises_for_image_url(self, request_factory):
        handler = self._make_handler(multimodal_processor=None)
        request = request_factory(self.IMAGE_MESSAGE)

        with pytest.raises(RuntimeError, match="--modality multimodal"):
            await self._prepare(handler, request)

    @pytest.mark.asyncio
    async def test_text_only_request_falls_back_to_token_ids(self):
        handler = self._make_handler(multimodal_processor=None)
        result = await self._prepare(handler, {"token_ids": [10, 20, 30]})
        assert result == [10, 20, 30]

    @pytest.mark.asyncio
    async def test_decode_with_prefill_metadata_bypasses_guard(self):
        handler = self._make_handler(multimodal_processor=None)
        handler.disaggregation_mode = DisaggregationMode.DECODE

        request = {"token_ids": [1, 2, 3], "messages": [self.IMAGE_MESSAGE]}
        epd_metadata = {
            "_prefill_prompt": "describe image",
            "_prefill_prompt_token_ids": [1, 2, 3],
        }

        result = await self._prepare(handler, request, epd_metadata)
        assert result["prompt"] == "describe image"
        assert result["prompt_token_ids"] == [1, 2, 3]
        assert result["multi_modal_data"] is None


class TestPrefillPromptMetadata:
    """Tests for prompt metadata in the legacy prefill handoff."""

    def _make_handler(self) -> HandlerBase:
        config = MagicMock()
        config.multimodal_processor = MagicMock()
        config.shutdown_event = None
        config.conversation_affinity = False
        return _ConcreteHandler(config)

    def _pack_metadata(self, request, processed_input, prompt, prompt_token_ids):
        params = DisaggregatedParams(request_type="context_only")
        output = MagicMock(disaggregated_params=params)
        result = MagicMock(prompt=prompt, prompt_token_ids=prompt_token_ids)

        return self._make_handler()._encode_and_pack_disaggregated_params(
            output,
            params,
            request,
            result,
            processed_input,
        )

    def test_text_only_prefill_omits_prompt_metadata(self):
        result = self._pack_metadata(
            request={"token_ids": [1, 2, 3]},
            processed_input=[1, 2, 3],
            prompt="text prompt",
            prompt_token_ids=[1, 2, 3],
        )

        assert "_epd_metadata" not in result

    def test_multimodal_prefill_preserves_prompt_metadata(self):
        result = self._pack_metadata(
            request={
                "token_ids": [1, 2, 3],
                "multi_modal_data": {
                    "image_url": [{"Url": "http://example.com/image.jpg"}]
                },
            },
            processed_input={"prompt": "describe image"},
            prompt="raw prompt",
            prompt_token_ids=[1, 2, 3],
        )

        assert result["_epd_metadata"] == {
            "_prefill_prompt": "describe image",
            "_prefill_prompt_token_ids": [1, 2, 3],
        }

    def test_epd_prefill_preserves_encoder_metadata(self):
        result = self._pack_metadata(
            request={
                "token_ids": [1, 2, 3],
                "_epd_processed_prompt": "encoder prompt",
                "_epd_prompt_token_ids": [1, 2, 3],
            },
            processed_input={"prompt": "encoder prompt"},
            prompt="engine prompt",
            prompt_token_ids=[4, 5, 6],
        )

        assert result["_epd_metadata"] == {
            "_prefill_prompt": "encoder prompt",
            "_prefill_prompt_token_ids": [4, 5, 6],
            "_epd_processed_prompt": "engine prompt",
            "_epd_prompt_token_ids": [4, 5, 6],
        }


class TestDisaggRequestId:
    """Tests for disagg_request_id population in _setup_disaggregated_params_for_mode."""

    def _make_prefill_handler(self, machine_id: int = 42) -> HandlerBase:
        config = MagicMock()
        config.shutdown_event = None
        config.disagg_machine_id = machine_id
        config.conversation_affinity = False
        handler = _ConcreteHandler(config)
        handler.disaggregation_mode = DisaggregationMode.PREFILL
        return handler

    def _make_decode_handler(self) -> HandlerBase:
        config = MagicMock()
        config.shutdown_event = None
        handler = _ConcreteHandler(config)
        handler.disaggregation_mode = DisaggregationMode.DECODE
        return handler

    def test_disagg_request_id_populated_in_prefill_mode(self):
        """When mode is PREFILL and no ep_disaggregated_params, disagg_request_id is set."""
        handler = self._make_prefill_handler()
        disagg_params, _, _ = handler._setup_disaggregated_params_for_mode(
            request={}, ep_disaggregated_params=None
        )
        assert disagg_params is not None
        assert disagg_params.disagg_request_id is not None
        assert isinstance(disagg_params.disagg_request_id, int)

    def test_disagg_request_id_unique_across_calls(self):
        """Multiple calls should produce different IDs."""
        handler = self._make_prefill_handler()
        ids = set()
        for _ in range(10):
            params, _, _ = handler._setup_disaggregated_params_for_mode(
                request={}, ep_disaggregated_params=None
            )
            ids.add(params.disagg_request_id)
        assert len(ids) == 10, f"Expected 10 unique IDs, got {len(ids)}"

    def test_disagg_request_id_set_on_ep_params_with_none(self):
        """When ep_disaggregated_params has disagg_request_id=None, it gets populated."""
        handler = self._make_prefill_handler()
        ep_params = MagicMock()
        ep_params.disagg_request_id = None
        # Make bool(ep_params) truthy so the if-branch is taken
        ep_params.__bool__ = lambda self: True

        params, _, _ = handler._setup_disaggregated_params_for_mode(
            request={}, ep_disaggregated_params=ep_params
        )
        assert params.disagg_request_id is not None
        assert isinstance(params.disagg_request_id, int)

    def test_disagg_request_id_not_overwritten_when_set(self):
        """When ep_disaggregated_params already has a disagg_request_id, keep it."""
        handler = self._make_prefill_handler()
        existing_id = 12345678
        ep_params = MagicMock()
        ep_params.disagg_request_id = existing_id
        ep_params.__bool__ = lambda self: True

        params, _, _ = handler._setup_disaggregated_params_for_mode(
            request={}, ep_disaggregated_params=ep_params
        )
        assert params.disagg_request_id == existing_id

    def test_machine_id_from_config(self):
        """disagg_machine_id is taken from the handler config."""
        handler = self._make_prefill_handler(machine_id=123)
        assert handler.disagg_machine_id == 123

    def test_different_machine_ids_produce_different_id_ranges(self):
        """Handlers with different machine_ids produce non-overlapping snowflake IDs."""
        handler_a = self._make_prefill_handler(machine_id=1)
        handler_b = self._make_prefill_handler(machine_id=2)
        params_a, _, _ = handler_a._setup_disaggregated_params_for_mode(
            request={}, ep_disaggregated_params=None
        )
        params_b, _, _ = handler_b._setup_disaggregated_params_for_mode(
            request={}, ep_disaggregated_params=None
        )
        assert params_a.disagg_request_id != params_b.disagg_request_id

    def test_decode_conditional_bypass_uses_request_disagg_params(self):
        """Conditional-disagg bypass runs full context+generation on decode."""
        handler = self._make_decode_handler()
        params, _, _ = handler._setup_disaggregated_params_for_mode(
            request={
                "annotations": [BYPASS_REMOTE_PREFILL_ANNOTATION],
                "disaggregated_params": {"request_type": "context_and_generation"},
            },
            ep_disaggregated_params=None,
        )
        assert params.request_type == "context_and_generation"

    def test_decode_conditional_bypass_preserves_epd_params(self):
        """Conditional-disagg bypass keeps EPD params for multimodal handling."""
        handler = self._make_decode_handler()
        ep_disaggregated_params = MagicMock()

        (
            params,
            returned_ep_params,
            epd_metadata,
        ) = handler._setup_disaggregated_params_for_mode(
            request={
                "annotations": [BYPASS_REMOTE_PREFILL_ANNOTATION],
                "disaggregated_params": {"request_type": "context_and_generation"},
            },
            ep_disaggregated_params=ep_disaggregated_params,
        )

        assert params.request_type == "context_and_generation"
        assert returned_ep_params is ep_disaggregated_params
        assert epd_metadata == {}


class TestHealthCheckPriority:
    """Verify generate_locally forwards the correct priority to generate_async.

    Health check requests (built by TrtllmHealthCheckPayload) must reach
    the TRT-LLM engine at priority=1.0.  Regular inference requests
    (built by the Rust frontend as PreprocessedRequest, which has no
    priority field) must fall back to DEFAULT_REQUEST_PRIORITY (0.5).
    """

    def _make_handler(self) -> HandlerBase:
        config = MagicMock()
        config.shutdown_event = None
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.conversation_affinity = False
        handler = _ConcreteHandler(config)
        handler.publisher = None
        handler.multimodal_processor = None
        handler.additional_metrics = None
        handler.max_seq_len = None
        handler.default_sampling_params = MockSamplingParams()
        return handler

    def _make_mock_generation_result(self):
        """Mock GenerationResult that yields a single finished token."""
        output = MagicMock()
        output.token_ids = [42]
        output.finish_reason = "stop"
        output.stop_reason = None
        output.request_perf_metrics = None

        res = MagicMock()
        res.outputs = [output]
        res.finished = True

        generation_result = MagicMock()
        generation_result.abort = MagicMock()

        async def mock_aiter(self_mock):
            yield res

        generation_result.__aiter__ = mock_aiter
        return generation_result

    def _make_context(self):
        """Mock Context whose cancellation never fires."""
        context = MagicMock()
        never_resolve = asyncio.get_event_loop().create_future()
        context.async_killed_or_stopped.return_value = never_resolve
        context.id.return_value = "test-priority"
        return context

    @pytest.mark.asyncio
    async def test_health_check_gets_priority_1(self):
        """TrtllmHealthCheckPayload → generate_locally → generate_async priority=1.0."""
        handler = self._make_handler()
        generation_result = self._make_mock_generation_result()
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)

        request = TrtllmHealthCheckPayload(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
        ).to_dict()

        context = self._make_context()
        chunks = [c async for c in handler.generate_locally(request, context)]
        assert len(chunks) > 0

        handler.engine.llm.generate_async.assert_called_once()
        _, kwargs = handler.engine.llm.generate_async.call_args
        assert kwargs["priority"] == 1.0

    @pytest.mark.asyncio
    async def test_regular_request_gets_default_priority(self):
        """Rust PreprocessedRequest shape (no priority key) → default 0.5."""
        handler = self._make_handler()
        generation_result = self._make_mock_generation_result()
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)

        # Mirrors the Rust PreprocessedRequest struct — no priority field.
        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": 10},
            "sampling_options": {"temperature": 0.7},
        }

        context = self._make_context()
        chunks = [c async for c in handler.generate_locally(request, context)]
        assert len(chunks) > 0

        handler.engine.llm.generate_async.assert_called_once()
        _, kwargs = handler.engine.llm.generate_async.call_args
        assert kwargs["priority"] == DEFAULT_REQUEST_PRIORITY

    @pytest.mark.asyncio
    async def test_default_max_tokens_uses_processed_prompt_token_ids(self):
        """DECODE-style processed tokens size the remaining context correctly."""
        handler = self._make_handler()
        handler.max_seq_len = 100
        handler._prepare_input_for_generation = mock.AsyncMock(
            return_value={"prompt_token_ids": list(range(40))}
        )
        generation_result = self._make_mock_generation_result()
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)

        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": None},
            "sampling_options": {},
        }

        chunks = [
            chunk
            async for chunk in handler.generate_locally(request, self._make_context())
        ]
        assert chunks

        _, kwargs = handler.engine.llm.generate_async.call_args
        assert kwargs["sampling_params"].max_tokens == 60

    @pytest.mark.asyncio
    async def test_expanded_prompt_len_is_not_forwarded_to_engine(self):
        handler = self._make_handler()
        handler._prepare_input_for_generation = mock.AsyncMock(
            return_value={
                "prompt_token_ids": [1, 2, 3],
                "expanded_prompt_len": 42,
            }
        )
        generation_result = self._make_mock_generation_result()
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)

        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": 10},
            "sampling_options": {},
        }

        chunks = [
            chunk
            async for chunk in handler.generate_locally(request, self._make_context())
        ]
        assert chunks

        _, kwargs = handler.engine.llm.generate_async.call_args
        assert "expanded_prompt_len" not in kwargs["inputs"]

    @pytest.mark.asyncio
    async def test_routing_cache_salt_forwarded_to_generate_async(self):
        handler = self._make_handler()
        generation_result = self._make_mock_generation_result()
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)

        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": 10},
            "sampling_options": {"temperature": 0.7},
            "routing": {"cache_salt": "tenant-a"},
        }

        chunks = [
            c async for c in handler.generate_locally(request, self._make_context())
        ]
        assert len(chunks) > 0

        handler.engine.llm.generate_async.assert_called_once()
        _, kwargs = handler.engine.llm.generate_async.call_args
        assert kwargs["cache_salt"] == "tenant-a"

    @pytest.mark.asyncio
    async def test_prefill_skips_generation_stop_conditions(self):
        handler = self._make_handler()
        handler.disaggregation_mode = DisaggregationMode.PREFILL
        handler.disagg_machine_id = 0
        handler.default_sampling_params = MockSamplingParams(stop_token_ids=[300])
        generation_result = MagicMock()

        async def empty_aiter(_self):
            for _ in ():
                yield

        generation_result.__aiter__ = empty_aiter
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)

        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {
                "max_tokens": 10,
                "min_tokens": 8,
                "ignore_eos": True,
                "stop_token_ids_hidden": [100],
                "stop_token_ids_visible": [200],
            },
            "sampling_options": {},
        }

        chunks = [
            c async for c in handler.generate_locally(request, self._make_context())
        ]
        assert chunks == []

        handler.engine.llm.generate_async.assert_called_once()
        _, kwargs = handler.engine.llm.generate_async.call_args
        sampling_params = kwargs["sampling_params"]
        assert sampling_params.max_tokens == 1
        assert sampling_params.min_tokens is None
        assert sampling_params.ignore_eos is False
        assert sampling_params.stop_token_ids == [300]
        assert kwargs["streaming"] is False


class TestDefaultMaxTokens:
    """Unit tests for HandlerBase._default_max_tokens (omitted max_tokens sizing)."""

    def test_text_fills_remaining_context(self):
        # 100 - len([1,2,3]) = 97
        assert HandlerBase._default_max_tokens(100, [1, 2, 3], False, None) == 97

    def test_text_on_multimodal_worker_ignores_expanded(self):
        # A text request (no images) uses len(token_ids) even if an expanded
        # length is somehow present; it must not defer like an image request.
        assert HandlerBase._default_max_tokens(100, [1, 2, 3], False, 40) == 97

    def test_returns_none_without_max_seq_len(self):
        assert HandlerBase._default_max_tokens(None, [1, 2, 3], False, None) is None

    def test_image_uses_expanded_len(self):
        # 100 - 40 = 60; token_ids ignored in favor of the expanded length
        assert HandlerBase._default_max_tokens(100, [1, 2, 3], True, 40) == 60

    def test_image_without_expanded_defers(self):
        # No expanded length available -> defer to engine default (None)
        assert HandlerBase._default_max_tokens(100, [1, 2, 3], True, None) is None

    def test_image_zero_expanded_len_is_valid(self):
        assert HandlerBase._default_max_tokens(100, [], True, 0) == 100

    def test_floors_at_one(self):
        # Prompt already at/over context -> never returns <= 0
        assert HandlerBase._default_max_tokens(3, [1, 2, 3, 4, 5], False, None) == 1


class TestExpandedPromptLen:
    """Unit tests for MultimodalRequestProcessor._expanded_prompt_len."""

    def _processor(self, mm_token_ids, tokens_per_image):
        # Bypass __init__ (which would build a real input processor) and inject a mock.
        proc = MultimodalRequestProcessor.__new__(MultimodalRequestProcessor)
        ip = MagicMock()
        ip.get_mm_token_ids.return_value = (
            torch.tensor(mm_token_ids) if mm_token_ids is not None else None
        )
        ip.get_num_tokens_per_image.side_effect = lambda image: tokens_per_image
        proc.input_processor = ip
        return proc

    def test_replaces_placeholders_with_image_tokens(self):
        # token_ids has one placeholder (99); one image expands to 256 tokens.
        # 4 tokens - 1 placeholder + 256 = 259
        proc = self._processor(mm_token_ids=[99], tokens_per_image=256)
        assert proc._expanded_prompt_len([1, 2, 99, 3], images=["img"]) == 259

    def test_multiple_images_sum(self):
        # Two placeholders, two images x 10 tokens: 5 - 2 + 20 = 23
        proc = self._processor(mm_token_ids=[99], tokens_per_image=10)
        assert proc._expanded_prompt_len([1, 99, 2, 99, 3], ["a", "b"]) == 23

    def test_none_without_input_processor(self):
        proc = MultimodalRequestProcessor.__new__(MultimodalRequestProcessor)
        proc.input_processor = None
        assert proc._expanded_prompt_len([1, 2, 3], ["img"]) is None

    def test_none_without_images(self):
        proc = self._processor(mm_token_ids=[99], tokens_per_image=256)
        assert proc._expanded_prompt_len([1, 2, 3], None) is None

    def test_none_on_processor_error(self):
        proc = MultimodalRequestProcessor.__new__(MultimodalRequestProcessor)
        ip = MagicMock()
        ip.get_mm_token_ids.side_effect = RuntimeError("boom")
        proc.input_processor = ip
        assert proc._expanded_prompt_len([1, 2, 3], ["img"]) is None


class TestRequestHasImages:
    """Unit tests for HandlerBase._request_has_images (image-vs-text classification).

    Regression coverage for the bug where a text request to a multimodal worker
    was mis-classified as multimodal and its omitted max_tokens deferred to the
    engine default (32) instead of filling from len(token_ids).
    """

    def test_text_request_has_no_images(self):
        # processed_input for text on a multimodal worker: no mm keys.
        assert HandlerBase._request_has_images({"prompt_token_ids": [1, 2, 3]}) is False

    def test_multi_modal_data_is_images(self):
        assert (
            HandlerBase._request_has_images({"multi_modal_data": {"image": ["x"]}})
            is True
        )

    def test_multi_modal_embeddings_is_images(self):
        assert (
            HandlerBase._request_has_images(
                {"multi_modal_embeddings": {"image": ["x"]}}
            )
            is True
        )

    def test_empty_mm_data_is_not_images(self):
        # mm key present but falsy (text path sets these to None) -> not images.
        assert HandlerBase._request_has_images({"multi_modal_data": None}) is False

    def test_none_is_not_images(self):
        assert HandlerBase._request_has_images(None) is False

    def test_non_dict_is_not_images(self):
        assert HandlerBase._request_has_images([1, 2, 3]) is False


class _FakeConversationParams:

    """Stand-in for tensorrt_llm.llmapi.ConversationParams. Only ``conversation_id``
    is read back by the assertions."""

    def __init__(self, conversation_id):
        self.conversation_id = conversation_id


class TestConversationAffinity:
    """Verify generate_locally's conversation-affinity branching on the legacy path.

    When engine-owned conversation-affinity ADP routing is enabled, the handler
    must (a) NOT force ``attention_dp_rank`` and (b) forward the frontend's
    ``agent_context.session_id`` as ``ConversationParams`` to
    ``generate_async``. When disabled, the router's ``dp_rank`` still drives
    ``SchedulingParams`` as before.
    """

    def _make_handler(self, *, conversation_affinity: bool) -> HandlerBase:
        config = MagicMock()
        config.shutdown_event = None
        config.disaggregation_mode = DisaggregationMode.AGGREGATED
        config.conversation_affinity = False
        handler = _ConcreteHandler(config)
        handler.publisher = None
        handler.multimodal_processor = None
        handler.additional_metrics = None
        handler.max_seq_len = None
        handler.default_sampling_params = MockSamplingParams()
        # Pre-resolve the gate so the lazy path in _generate_locally_impl is a
        # no-op — the branching under test happens downstream.
        handler._conversation_affinity = conversation_affinity
        return handler

    def _make_mock_generation_result(self):
        output = MagicMock()
        output.token_ids = [42]
        output.finish_reason = "stop"
        output.stop_reason = None
        output.request_perf_metrics = None

        res = MagicMock()
        res.outputs = [output]
        res.finished = True

        generation_result = MagicMock()
        generation_result.abort = MagicMock()

        async def mock_aiter(self_mock):
            yield res

        generation_result.__aiter__ = mock_aiter
        return generation_result

    def _make_context(self):
        context = MagicMock()
        never_resolve = asyncio.get_event_loop().create_future()
        context.async_killed_or_stopped.return_value = never_resolve
        context.id.return_value = "conv-affinity-test"
        return context

    async def _drive(self, handler, request):
        generation_result = self._make_mock_generation_result()
        handler.engine.llm.generate_async = MagicMock(return_value=generation_result)
        chunks = [
            c async for c in handler.generate_locally(request, self._make_context())
        ]
        assert len(chunks) > 0
        handler.engine.llm.generate_async.assert_called_once()
        _, kwargs = handler.engine.llm.generate_async.call_args
        return kwargs

    @pytest.mark.asyncio
    async def test_affinity_off_forwards_router_dp_rank(self):
        """affinity=False + router dp_rank → SchedulingParams with that rank."""
        handler = self._make_handler(conversation_affinity=False)
        kwargs = await self._drive(
            handler,
            {
                "token_ids": [1, 2, 3],
                "stop_conditions": {"max_tokens": 10},
                "sampling_options": {"temperature": 0.7},
                "routing": {"dp_rank": 3},
            },
        )
        scheduling_params = kwargs["scheduling_params"]
        assert scheduling_params is not None
        assert scheduling_params.attention_dp_rank == 3
        assert scheduling_params.attention_dp_relax is False
        assert "conversation_params" not in kwargs

    @pytest.mark.asyncio
    async def test_affinity_off_no_rank_leaves_scheduling_params_none(self):
        """affinity=False + no router rank → scheduling_params=None, no conv kwarg."""
        handler = self._make_handler(conversation_affinity=False)
        kwargs = await self._drive(
            handler,
            {
                "token_ids": [1, 2, 3],
                "stop_conditions": {"max_tokens": 10},
                "sampling_options": {"temperature": 0.7},
            },
        )
        assert kwargs["scheduling_params"] is None
        assert "conversation_params" not in kwargs

    @pytest.mark.asyncio
    async def test_affinity_on_with_session_id_suppresses_rank(self, monkeypatch):
        """affinity=True + session id → scheduling_params=None + ConversationParams."""
        monkeypatch.setattr(
            "dynamo.trtllm.conversation_affinity.ConversationParams",
            _FakeConversationParams,
        )
        handler = self._make_handler(conversation_affinity=True)
        # Router still stamps a rank; affinity mode must ignore it — an explicit
        # rank would bypass the engine's ConversationAwareADPRouter.
        kwargs = await self._drive(
            handler,
            {
                "token_ids": [1, 2, 3],
                "stop_conditions": {"max_tokens": 10},
                "sampling_options": {"temperature": 0.7},
                "routing": {"dp_rank": 3},
                "agent_context": {"session_id": "run-42:agent-0"},
            },
        )
        assert kwargs["scheduling_params"] is None
        conv_params = kwargs["conversation_params"]
        assert conv_params is not None
        assert conv_params.conversation_id == "run-42:agent-0"

    @pytest.mark.asyncio
    async def test_affinity_on_without_session_id_passes_none_conversation_params(
        self, monkeypatch
    ):
        """affinity=True + no session id → scheduling_params=None,
        conversation_params kwarg present with value None (no-id balancing)."""
        monkeypatch.setattr(
            "dynamo.trtllm.conversation_affinity.ConversationParams",
            _FakeConversationParams,
        )
        handler = self._make_handler(conversation_affinity=True)
        kwargs = await self._drive(
            handler,
            {
                "token_ids": [1, 2, 3],
                "stop_conditions": {"max_tokens": 10},
                "sampling_options": {"temperature": 0.7},
            },
        )
        assert kwargs["scheduling_params"] is None
        assert "conversation_params" in kwargs
        assert kwargs["conversation_params"] is None

    @pytest.mark.asyncio
    async def test_lazy_resolution_raises_when_api_missing(self, monkeypatch):
        """Startup gate: engine has affinity ON but ConversationParams API is
        missing → RuntimeError on first request, before generate_async is called."""
        monkeypatch.setattr(
            "dynamo.trtllm.request_handlers.handler_base.CONVERSATION_PARAMS_AVAILABLE",
            False,
        )
        handler = self._make_handler(conversation_affinity=False)
        # Trigger the lazy path with an engine config that enables affinity.
        handler._conversation_affinity = None
        handler.engine.llm.args.attention_dp_config = {
            "kv_cache_routing_conversation_affinity": True
        }
        handler.engine.llm.generate_async = MagicMock()

        request = {
            "token_ids": [1, 2, 3],
            "stop_conditions": {"max_tokens": 10},
            "sampling_options": {"temperature": 0.7},
        }
        with pytest.raises(RuntimeError, match="ConversationParams API"):
            async for _ in handler.generate_locally(request, self._make_context()):
                pass
        handler.engine.llm.generate_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_override_suppresses_dp_rank_and_forwards_conversation_params(
        self, monkeypatch
    ):
        """DYN_ENGINE_CONV_AFFINITY override=True + engine detection disabled →
        dp_rank suppressed, conversation_params forwarded on first request."""
        monkeypatch.setattr(
            "dynamo.trtllm.request_handlers.handler_base.CONVERSATION_PARAMS_AVAILABLE",
            True,
        )
        monkeypatch.setattr(
            "dynamo.trtllm.conversation_affinity.ConversationParams",
            _FakeConversationParams,
        )
        monkeypatch.setattr(
            "dynamo.trtllm.request_handlers.handler_base.engine_conversation_affinity_enabled",
            lambda _: False,
        )
        handler = self._make_handler(conversation_affinity=False)
        # Reset to None so lazy init runs on first request and folds in the override.
        handler._conversation_affinity = None
        handler._engine_conversation_affinity_override = True
        kwargs = await self._drive(
            handler,
            {
                "token_ids": [1, 2, 3],
                "stop_conditions": {"max_tokens": 10},
                "sampling_options": {"temperature": 0.7},
                "routing": {"dp_rank": 3},
                "agent_context": {"session_id": "run-99:agent-0"},
            },
        )
        # Override must suppress the router rank just like auto-detection does.
        assert kwargs["scheduling_params"] is None
        conv_params = kwargs["conversation_params"]
        assert conv_params is not None
        assert conv_params.conversation_id == "run-99:agent-0"

    @pytest.mark.asyncio
    async def test_override_raises_when_conversation_params_api_missing(
        self, monkeypatch
    ):
        """DYN_ENGINE_CONV_AFFINITY=true on a build without ConversationParams →
        RuntimeError on first request during lazy init."""
        monkeypatch.setattr(
            "dynamo.trtllm.request_handlers.handler_base.CONVERSATION_PARAMS_AVAILABLE",
            False,
        )
        handler = self._make_handler(conversation_affinity=False)
        # Reset to None so lazy init runs and hits the guard.
        handler._conversation_affinity = None
        handler._engine_conversation_affinity_override = True
        handler.engine.llm.generate_async = MagicMock()

        with pytest.raises(RuntimeError, match="DYN_ENGINE_CONV_AFFINITY"):
            async for _ in handler.generate_locally(
                {
                    "token_ids": [1, 2, 3],
                    "stop_conditions": {"max_tokens": 10},
                    "sampling_options": {"temperature": 0.7},
                },
                self._make_context(),
            ):
                pass
        handler.engine.llm.generate_async.assert_not_called()
