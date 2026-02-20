# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from dataclasses import dataclass
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
from dynamo.trtllm.request_handlers.handler_base import HandlerBase

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@dataclass
class MockSamplingParams:
    """Mock sampling params object for testing."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
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


class _ConcreteHandler(HandlerBase):
    """Concrete subclass of HandlerBase for testing (satisfies abstract method)."""

    async def generate(self, *args, **kwargs):
        raise NotImplementedError


class TestHandleCancellationAbortToggle:
    """Tests for the disable_request_abort toggle in _handle_cancellation."""

    def _make_handler(self, disable_request_abort: bool) -> HandlerBase:
        """Create a HandlerBase with mocked config."""
        config = MagicMock()
        config.disable_request_abort = disable_request_abort
        config.shutdown_event = None
        return _ConcreteHandler(config)

    @pytest.mark.asyncio
    async def test_abort_called_by_default(self):
        handler = self._make_handler(disable_request_abort=False)
        generation_result = MagicMock()
        context = MagicMock()
        # async_killed_or_stopped returns an awaitable that resolves immediately
        # (simulating the client cancelling the request)
        killed_future = asyncio.get_event_loop().create_future()
        killed_future.set_result(None)
        context.async_killed_or_stopped.return_value = killed_future
        context.id.return_value = "test-id-1"

        await handler._handle_cancellation(generation_result, context)

        generation_result.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_abort_not_called_when_disabled(self):
        handler = self._make_handler(disable_request_abort=True)
        generation_result = MagicMock()
        context = MagicMock()
        killed_future = asyncio.get_event_loop().create_future()
        killed_future.set_result(None)
        context.async_killed_or_stopped.return_value = killed_future
        context.id.return_value = "test-id-2"

        await handler._handle_cancellation(generation_result, context)

        generation_result.abort.assert_not_called()
