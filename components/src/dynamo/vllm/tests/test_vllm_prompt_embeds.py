# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt embeddings support in vLLM backend."""

import base64
import io
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
]


@pytest.fixture
def mock_handler():
    """Create a mock handler with _decode_prompt_embeds method."""

    class MockHandler:
        pass

    handler = MockHandler()
    handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(
        handler
    )
    return handler


def encode_tensor_to_base64(tensor: torch.Tensor) -> str:
    """Helper to encode a tensor to base64 using PyTorch format."""
    buffer = io.BytesIO()
    torch.save(tensor, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestPromptEmbedsDecode:
    """Tests for prompt embeddings decoding functionality."""

    @pytest.mark.parametrize(
        "shape,dtype",
        [
            ((10, 4096), torch.float32),  # 2D: sequence x hidden
            ((10, 768), torch.float32),  # 2D: smaller hidden dim
            ((2, 10, 768), torch.float32),  # 3D: batch x sequence x hidden
            ((5, 20, 1024), torch.float16),  # 3D with float16
        ],
        ids=["2d-4096", "2d-768", "3d-batch", "3d-float16"],
    )
    def test_decode_valid_embeddings_various_shapes(self, mock_handler, shape, dtype):
        """Test decoding embeddings with various shapes and dtypes."""
        embeddings = torch.randn(*shape, dtype=dtype)
        embeddings_base64 = encode_tensor_to_base64(embeddings)

        result = mock_handler._decode_prompt_embeds(embeddings_base64)

        assert isinstance(result, torch.Tensor)
        assert result.shape == shape, f"Shape should be preserved: {shape}"
        assert result.dtype == dtype, f"Dtype should be preserved: {dtype}"
        torch.testing.assert_close(result, embeddings, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "invalid_input,error_match,description",
        [
            # Invalid base64
            ("not-valid-base64!!!", r"(Invalid base64|Failed to decode)", "bad base64"),
            # Empty string
            ("", r".", "empty string"),
            # Raw bytes (not PyTorch format)
            (
                base64.b64encode(b"not a pytorch tensor").decode("utf-8"),
                r"Failed to decode.*PyTorch",
                "raw bytes",
            ),
            # Corrupted PyTorch format
            (
                base64.b64encode(b"PK\x03\x04" + b"invalid_data" * 10).decode("utf-8"),
                r"Failed to decode.*PyTorch",
                "corrupted zip",
            ),
        ],
        ids=["bad-base64", "empty", "raw-bytes", "corrupted-zip"],
    )
    def test_decode_invalid_inputs(
        self, mock_handler, invalid_input, error_match, description
    ):
        """Test that invalid inputs raise ValueError."""
        with pytest.raises(ValueError, match=error_match):
            mock_handler._decode_prompt_embeds(invalid_input)

    def test_decode_numpy_format_rejected(self, mock_handler):
        """Test that NumPy format is rejected (PyTorch format required)."""
        embeddings = np.random.randn(10, 768).astype(np.float32)
        buffer = io.BytesIO()
        np.save(buffer, embeddings)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        with pytest.raises(ValueError, match="Failed to decode.*PyTorch"):
            mock_handler._decode_prompt_embeds(embeddings_base64)

    def test_decode_non_tensor_object_rejected(self, mock_handler):
        """Test that non-tensor PyTorch objects are rejected."""
        non_tensor = {"key": "value"}
        embeddings_base64 = encode_tensor_to_base64_obj(non_tensor)

        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            mock_handler._decode_prompt_embeds(embeddings_base64)


def encode_tensor_to_base64_obj(obj) -> str:
    """Helper to encode any object to base64 using torch.save."""
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


class TestEmbeddingsDataFormats:
    """Tests for various embedding data formats and value ranges."""

    @pytest.mark.parametrize("size", [128, 384, 768, 1024, 2048, 4096])
    def test_various_embedding_sizes(self, mock_handler, size):
        """Test decoding embeddings of various sizes."""
        embeddings = torch.randn(size, dtype=torch.float32)
        embeddings_base64 = encode_tensor_to_base64(embeddings)

        result = mock_handler._decode_prompt_embeds(embeddings_base64)

        assert result.shape == (size,), f"Failed for size {size}"
        torch.testing.assert_close(result, embeddings, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize(
        "values",
        [
            [0.0, 0.0, 0.0],  # Zeros
            [1.0, 1.0, 1.0],  # Ones
            [-1.0, 0.0, 1.0],  # Mixed
            [1e-6, 1e-3, 1e3],  # Various magnitudes
            [3.14159265, 2.71828182, 1.41421356],  # Precise values
        ],
        ids=["zeros", "ones", "mixed", "magnitudes", "precise"],
    )
    def test_embedding_value_ranges_preserved(self, mock_handler, values):
        """Test that various value ranges are preserved with float32 precision."""
        embeddings = torch.tensor(values, dtype=torch.float32)
        embeddings_base64 = encode_tensor_to_base64(embeddings)

        result = mock_handler._decode_prompt_embeds(embeddings_base64)

        torch.testing.assert_close(result, embeddings, rtol=1e-6, atol=1e-6)


class TestUsageStatistics:
    """Tests for usage statistics calculation."""

    @pytest.mark.parametrize(
        "prompt_token_ids,embedding_seq_len,completion_tokens,expected_prompt,expected_total",
        [
            # Embeddings: use embedding_sequence_length
            ([], 10, 5, 10, 15),
            # Text: use len(prompt_token_ids)
            ([1, 2, 3, 4, 5, 6, 7], None, 3, 7, 10),
            # Embeddings override token_ids
            ([1, 2, 3], 20, 2, 20, 22),
            # Zero sequence length edge case
            ([], 0, 2, 0, 2),
        ],
        ids=["embeddings", "text", "embeddings-override", "zero-seq-len"],
    )
    def test_build_completion_usage(
        self,
        prompt_token_ids,
        embedding_seq_len,
        completion_tokens,
        expected_prompt,
        expected_total,
    ):
        """Test usage statistics calculation for various scenarios."""
        mock_output = Mock()
        mock_output.prompt_token_ids = prompt_token_ids
        mock_output.outputs = [Mock(token_ids=list(range(completion_tokens)))]
        mock_output.num_cached_tokens = 0

        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=embedding_seq_len
        )

        assert result["prompt_tokens"] == expected_prompt
        assert result["completion_tokens"] == completion_tokens
        assert result["total_tokens"] == expected_total

    def test_build_completion_usage_no_prompt_info(self):
        """Test usage when no prompt token info available."""
        mock_output = Mock()
        mock_output.prompt_token_ids = None
        mock_output.outputs = [Mock(token_ids=[1, 2, 3])]
        mock_output.num_cached_tokens = 0

        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=None
        )

        assert result["prompt_tokens"] is None
        assert result["completion_tokens"] == 3
        assert result["total_tokens"] is None

    def test_build_completion_usage_with_cached_tokens(self):
        """Test that cached tokens are reported in prompt_tokens_details."""
        mock_output = Mock()
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs = [Mock(token_ids=[6, 7])]
        mock_output.num_cached_tokens = 3

        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=None
        )

        assert result["prompt_tokens"] == 5
        assert result["completion_tokens"] == 2
        assert result["prompt_tokens_details"] == {"cached_tokens": 3}
