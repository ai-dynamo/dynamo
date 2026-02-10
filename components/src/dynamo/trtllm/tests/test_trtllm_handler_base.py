# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest import mock

import pytest

from dynamo.trtllm.request_handlers.handler_base import HandlerBase

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@dataclass
class MockGuidedDecodingParams:
    """Mock of TRT-LLM's GuidedDecodingParams dataclass."""

    json: object | None = None
    regex: str | None = None
    grammar: str | None = None
    json_object: bool = False
    structural_tag: str | None = None

    def _validate(self):
        pass


@dataclass
class MockSamplingParams:
    """Mock sampling params object for testing."""

    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    repetition_penalty: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    guided_decoding: MockGuidedDecodingParams | None = None

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
    """Regression tests for tool_choice=required causing 500 errors.

    When the Dynamo Rust frontend receives tool_choice="required" + tools,
    it converts them into a guided_decoding JSON schema and serializes it
    as a plain dict over TCP. The TRT-LLM handler must convert this dict
    into a proper GuidedDecodingParams object. Without conversion:

        AttributeError: 'dict' object has no attribute 'json_object'
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

    def test_guided_decoding_dict_overwrites_as_raw_dict(self):
        """_override_sampling_params sets guided_decoding to a raw dict.

        The Rust frontend serializes GuidedDecodingOptions as a JSON dict.
        dataclasses.replace() blindly sets it on SamplingParams without
        converting to GuidedDecodingParams, so downstream code that does
        `self.guided_decoding.json_object` crashes with AttributeError.
        """
        sampling_params = MockSamplingParams()
        request = {
            "sampling_options": {
                "temperature": 0.7,
                "guided_decoding": self.GUIDED_DECODING_DICT,
            }
        }

        result = HandlerBase._override_sampling_params(sampling_params, request)

        # Bug: guided_decoding is a raw dict instead of GuidedDecodingParams
        assert isinstance(result.guided_decoding, dict)
        assert not isinstance(result.guided_decoding, MockGuidedDecodingParams)

        # This is exactly what TRT-LLM does in sampling_params.py
        # _get_guided_decoding_params() and it crashes:
        with pytest.raises(AttributeError, match="json_object"):
            _ = result.guided_decoding.json_object
