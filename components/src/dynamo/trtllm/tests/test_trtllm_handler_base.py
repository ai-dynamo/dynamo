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

from dynamo.llm.exceptions import EngineShutdown
from dynamo.trtllm.constants import DisaggregationMode
from dynamo.trtllm.health_check import TrtllmHealthCheckPayload
from dynamo.trtllm.request_handlers.handler_base import HandlerBase

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_1,
]


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
    ignore_eos: bool = False
    guided_decoding: object | None = None

    def __post_init__(self):
        """Called after dataclass initialization (including via replace())."""
        pass


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


class TestDisaggRequestId:
    """Tests for disagg_request_id population in _setup_disaggregated_params_for_mode."""

    def _make_prefill_handler(self, machine_id: int = 42) -> HandlerBase:
        config = MagicMock()
        config.shutdown_event = None
        config.disagg_machine_id = machine_id
        handler = _ConcreteHandler(config)
        handler.disaggregation_mode = DisaggregationMode.PREFILL
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
