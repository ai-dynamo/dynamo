# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for disagg_utils.py

Tests the disaggregated params utility functions.
"""

import base64
import sys
from dataclasses import dataclass
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest


# Mock tensorrt_llm before importing our module
@pytest.fixture(autouse=True)
def mock_tensorrt_llm():
    """Mock tensorrt_llm module for testing without TRT-LLM installed."""
    mock_trtllm = MagicMock()

    @dataclass
    class MockDisaggregatedParams:
        request_type: str = "context_only"
        opaque_state: Any = None
        multimodal_embedding_handles: Optional[list] = None
        multimodal_hashes: Optional[list] = None

    mock_trtllm.llmapi.DisaggregatedParams = MockDisaggregatedParams

    with patch.dict(sys.modules, {"tensorrt_llm": mock_trtllm, "tensorrt_llm.llmapi": mock_trtllm.llmapi}):
        # Re-import after mocking
        if "dynamo.trtllm.utils.disagg_utils" in sys.modules:
            del sys.modules["dynamo.trtllm.utils.disagg_utils"]
        if "dynamo.trtllm.constants" in sys.modules:
            del sys.modules["dynamo.trtllm.constants"]

        yield MockDisaggregatedParams


class TestDisaggregatedParamsCodec:
    """Tests for DisaggregatedParamsCodec"""

    def test_encode_converts_bytes_to_base64(self, mock_tensorrt_llm):
        """Test that encode converts bytes to base64 string."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec

        params = mock_tensorrt_llm(
            request_type="context_only",
            opaque_state=b"test_bytes",
        )

        encoded = DisaggregatedParamsCodec.encode(params)

        expected_b64 = base64.b64encode(b"test_bytes").decode("utf-8")
        assert encoded.opaque_state == expected_b64

    def test_encode_returns_none_for_none_input(self, mock_tensorrt_llm):
        """Test that encode returns None for None input."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec

        result = DisaggregatedParamsCodec.encode(None)

        assert result is None

    def test_decode_converts_base64_to_bytes(self, mock_tensorrt_llm):
        """Test that decode converts base64 string to bytes."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec

        b64_string = base64.b64encode(b"test_bytes").decode("utf-8")
        params = mock_tensorrt_llm(
            request_type="context_only",
            opaque_state=b64_string,
        )

        decoded = DisaggregatedParamsCodec.decode(params)

        assert decoded.opaque_state == b"test_bytes"

    def test_decode_returns_none_for_none_input(self, mock_tensorrt_llm):
        """Test that decode returns None for None input."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec

        result = DisaggregatedParamsCodec.decode(None)

        assert result is None

    def test_encode_decode_roundtrip(self, mock_tensorrt_llm):
        """Test that encode/decode is a roundtrip."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsCodec

        original_bytes = b"test_opaque_state_data"
        params = mock_tensorrt_llm(
            request_type="context_only",
            opaque_state=original_bytes,
        )

        encoded = DisaggregatedParamsCodec.encode(params)
        decoded = DisaggregatedParamsCodec.decode(encoded)

        assert decoded.opaque_state == original_bytes


class TestDisaggregatedParamsUtils:
    """Tests for DisaggregatedParamsUtils"""

    def test_decode_from_prefill_extracts_params(self, mock_tensorrt_llm):
        """Test decode_from_prefill extracts and decodes params."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        b64_state = base64.b64encode(b"test").decode("utf-8")
        prefill_result = {
            "disaggregated_params": {
                "request_type": "context_only",
                "opaque_state": b64_state,
            }
        }

        params, metadata = DisaggregatedParamsUtils.decode_from_prefill(prefill_result)

        assert params.request_type == "generation_only"
        assert params.opaque_state == b"test"
        assert metadata == {}

    def test_decode_from_prefill_extracts_metadata(self, mock_tensorrt_llm):
        """Test decode_from_prefill extracts EPD metadata."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        prefill_result = {
            "disaggregated_params": {
                "request_type": "context_only",
                "opaque_state": None,
                "_epd_metadata": {
                    "_prefill_prompt": "test prompt",
                    "_prefill_prompt_token_ids": [1, 2, 3],
                },
            }
        }

        params, metadata = DisaggregatedParamsUtils.decode_from_prefill(prefill_result)

        assert metadata["_prefill_prompt"] == "test prompt"
        assert metadata["_prefill_prompt_token_ids"] == [1, 2, 3]

    def test_decode_from_prefill_removes_worker_id(self, mock_tensorrt_llm):
        """Test decode_from_prefill removes worker_id from params."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        prefill_result = {
            "disaggregated_params": {
                "request_type": "context_only",
                "opaque_state": None,
                "worker_id": "worker-123",  # Should be removed
            }
        }

        params, metadata = DisaggregatedParamsUtils.decode_from_prefill(prefill_result)

        # worker_id should not cause an error (it's popped before creating params)
        assert params.request_type == "generation_only"

    def test_decode_from_prefill_clears_multimodal_handles(self, mock_tensorrt_llm):
        """Test decode_from_prefill clears multimodal_embedding_handles."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        prefill_result = {
            "disaggregated_params": {
                "request_type": "context_only",
                "opaque_state": None,
                "multimodal_embedding_handles": ["handle1", "handle2"],
            }
        }

        params, metadata = DisaggregatedParamsUtils.decode_from_prefill(prefill_result)

        assert params.multimodal_embedding_handles is None

    def test_build_prefill_metadata_from_processed_input(self, mock_tensorrt_llm):
        """Test _build_prefill_metadata uses processed_input prompt."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        res = MagicMock()
        res.prompt = "raw_prompt"
        res.prompt_token_ids = [1, 2, 3]

        processed_input = {"prompt": "processed_prompt"}
        request: dict[str, Any] = {}

        metadata = DisaggregatedParamsUtils.build_prefill_metadata(
            request, res, processed_input
        )

        assert metadata["_prefill_prompt"] == "processed_prompt"
        assert metadata["_prefill_prompt_token_ids"] == [1, 2, 3]

    def test_build_prefill_metadata_falls_back_to_res_prompt(self, mock_tensorrt_llm):
        """Test _build_prefill_metadata falls back to res.prompt."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        res = MagicMock()
        res.prompt = "raw_prompt"
        res.prompt_token_ids = [1, 2, 3]

        metadata = DisaggregatedParamsUtils.build_prefill_metadata(
            request={}, res=res, processed_input=None
        )

        assert metadata["_prefill_prompt"] == "raw_prompt"

    def test_build_prefill_metadata_includes_epd_fields(self, mock_tensorrt_llm):
        """Test _build_prefill_metadata includes EPD-specific fields."""
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        res = MagicMock()
        res.prompt = "prompt"
        res.prompt_token_ids = [1, 2, 3]

        request = {
            "_epd_processed_prompt": "epd_prompt",
            "_epd_prompt_token_ids": [4, 5, 6],
        }

        metadata = DisaggregatedParamsUtils.build_prefill_metadata(
            request, res, processed_input=None
        )

        assert metadata["_epd_processed_prompt"] == "prompt"
        assert metadata["_epd_prompt_token_ids"] == [1, 2, 3]


class TestSetupForMode:
    """Tests for DisaggregatedParamsUtils.setup_for_mode"""

    def test_prefill_mode_creates_context_only_params(self, mock_tensorrt_llm):
        """Test PREFILL mode creates params with request_type=context_only."""
        from dynamo.trtllm.constants import DisaggregationMode
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        params, ep_params, metadata = DisaggregatedParamsUtils.setup_for_mode(
            DisaggregationMode.PREFILL,
            request={},
            ep_disaggregated_params=None,
        )

        assert params.request_type == "context_only"
        assert metadata == {}

    def test_prefill_mode_uses_ep_params_when_provided(self, mock_tensorrt_llm):
        """Test PREFILL mode uses ep_disaggregated_params when provided."""
        from dynamo.trtllm.constants import DisaggregationMode
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        ep_params = mock_tensorrt_llm(request_type="original")

        params, returned_ep_params, metadata = DisaggregatedParamsUtils.setup_for_mode(
            DisaggregationMode.PREFILL,
            request={},
            ep_disaggregated_params=ep_params,
        )

        assert params.request_type == "context_only"
        assert params is ep_params  # Should be the same object

    def test_decode_mode_extracts_from_prefill_result(self, mock_tensorrt_llm):
        """Test DECODE mode extracts params from prefill_result."""
        from dynamo.trtllm.constants import DisaggregationMode
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        request = {
            "prefill_result": {
                "disaggregated_params": {
                    "request_type": "context_only",
                    "opaque_state": None,
                    "_epd_metadata": {"_prefill_prompt": "test"},
                }
            }
        }

        params, ep_params, metadata = DisaggregatedParamsUtils.setup_for_mode(
            DisaggregationMode.DECODE,
            request=request,
            ep_disaggregated_params=None,
        )

        assert params.request_type == "generation_only"
        assert metadata["_prefill_prompt"] == "test"

    def test_aggregated_mode_returns_none_params(self, mock_tensorrt_llm):
        """Test AGGREGATED mode returns None params when no prefill_result."""
        from dynamo.trtllm.constants import DisaggregationMode
        from dynamo.trtllm.utils.disagg_utils import DisaggregatedParamsUtils

        params, ep_params, metadata = DisaggregatedParamsUtils.setup_for_mode(
            DisaggregationMode.AGGREGATED,
            request={},
            ep_disaggregated_params=None,
        )

        assert params is None
        assert metadata == {}
