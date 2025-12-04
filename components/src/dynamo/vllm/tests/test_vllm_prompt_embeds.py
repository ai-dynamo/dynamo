# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for prompt embeddings support in vLLM backend."""

import base64

import numpy as np
import pytest
import torch

from dynamo.vllm.handlers import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
]


class TestPromptEmbedsDecode:
    """Tests for prompt embeddings decoding functionality."""

    def test_decode_valid_embeddings_pytorch_format(self):
        """Test decoding embeddings in PyTorch format (torch.save)."""
        import io

        # Create test embeddings with 2D shape
        embeddings = torch.randn(10, 4096, dtype=torch.float32)

        # Encode as PyTorch format (preserves shape and dtype!)
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Mock handler
        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        # Decode
        result = handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

        # Verify shape and dtype are preserved!
        assert isinstance(result, torch.Tensor)
        assert result.shape == (
            10,
            4096,
        ), "Shape should be preserved from PyTorch format"
        assert result.dtype == torch.float32, "Dtype should be preserved"
        torch.testing.assert_close(result, embeddings, rtol=1e-5, atol=1e-5)

    def test_decode_embeddings_2d_pytorch_format(self):
        """Test decoding 2D embeddings using PyTorch format."""
        import io

        # Create 2D embeddings
        embeddings = torch.randn(10, 768, dtype=torch.float32)

        # Encode as PyTorch format (preserves 2D shape)
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        result = handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

        assert isinstance(result, torch.Tensor)
        # With PyTorch format, shape is preserved!
        assert result.shape == (10, 768), "PyTorch format should preserve 2D shape"
        torch.testing.assert_close(result, embeddings, rtol=1e-5, atol=1e-5)

    def test_decode_embeddings_3d_pytorch_format(self):
        """Test decoding 3D embeddings using PyTorch format."""
        import io

        # Create 3D embeddings (batch, sequence, hidden)
        embeddings = torch.randn(2, 10, 768, dtype=torch.float32)

        # Encode as PyTorch format (preserves 3D shape)
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        result = handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

        assert isinstance(result, torch.Tensor)
        # With PyTorch format, 3D shape is preserved!
        assert result.shape == (2, 10, 768), "PyTorch format should preserve 3D shape"
        torch.testing.assert_close(result, embeddings, rtol=1e-5, atol=1e-5)

    def test_decode_invalid_numpy_format(self):
        """Test that NumPy format is rejected (no longer supported)."""
        import io

        # Create NumPy format (deprecated - should fail)
        embeddings = np.random.randn(10, 768).astype(np.float32)
        buffer = io.BytesIO()
        np.save(buffer, embeddings)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        # NumPy format should fail - PyTorch format required
        with pytest.raises(ValueError, match="Failed to decode.*PyTorch"):
            handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

    def test_decode_invalid_base64(self):
        """Test that invalid base64 raises ValueError."""
        invalid_base64 = "not-valid-base64!!!"

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        with pytest.raises(ValueError, match="(Invalid base64|Failed to decode)"):
            handler._decode_prompt_embeds(invalid_base64)  # type: ignore[attr-defined]

    def test_decode_empty_string(self):
        """Test that empty string raises ValueError."""

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        with pytest.raises(ValueError):
            handler._decode_prompt_embeds("")  # type: ignore[attr-defined]

    def test_decode_invalid_pytorch_data(self):
        """Test that invalid PyTorch data is rejected."""
        # Create raw bytes (not PyTorch format)
        bad_bytes = b"not a pytorch tensor"
        bad_base64 = base64.b64encode(bad_bytes).decode("utf-8")

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        # Should raise ValueError
        with pytest.raises(ValueError, match="Failed to decode.*PyTorch"):
            handler._decode_prompt_embeds(bad_base64)  # type: ignore[attr-defined]

    def test_decode_non_tensor_object(self):
        """Test that non-tensor PyTorch objects are rejected."""
        import io

        # Save a non-tensor object using torch.save
        non_tensor = {"key": "value"}  # Dictionary, not tensor
        buffer = io.BytesIO()
        torch.save(non_tensor, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        # Should raise ValueError for non-tensor
        with pytest.raises(ValueError, match="must be a torch.Tensor"):
            handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]


class TestEmbeddingsDataFormats:
    """Tests for various embedding data formats."""

    def test_various_embedding_sizes(self):
        """Test decoding embeddings of various sizes."""
        sizes = [128, 384, 768, 1024, 1536, 2048, 4096]

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        for size in sizes:
            embeddings = torch.tensor(np.random.randn(size).astype(np.float32))
            # Use PyTorch format
            import io

            buffer = io.BytesIO()
            torch.save(embeddings, buffer)
            embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            result = handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

            assert result.shape == (size,), f"Failed for size {size}"
            torch.testing.assert_close(result, embeddings, rtol=1e-5, atol=1e-5)

    def test_embedding_value_ranges(self):
        """Test that various value ranges are preserved."""
        # Test different value ranges
        test_cases = [
            np.array([0.0, 0.0, 0.0], dtype=np.float32),  # Zeros
            np.array([1.0, 1.0, 1.0], dtype=np.float32),  # Ones
            np.array([-1.0, 0.0, 1.0], dtype=np.float32),  # Mixed
            np.array([1e-6, 1e-3, 1e3], dtype=np.float32),  # Various magnitudes
        ]

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        for test_data in test_cases:
            # Use PyTorch format
            embeddings_tensor = torch.tensor(test_data)
            import io

            buffer = io.BytesIO()
            torch.save(embeddings_tensor, buffer)
            embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

            result = handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

            torch.testing.assert_close(result, embeddings_tensor, rtol=1e-5, atol=1e-5)

    def test_embedding_precision(self):
        """Test that float32 precision is maintained."""
        # Create embeddings with precise values
        embeddings_np = np.array(
            [
                3.14159265,
                2.71828182,
                1.41421356,
            ],
            dtype=np.float32,
        )

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        # Use PyTorch format
        embeddings_tensor = torch.tensor(embeddings_np)
        import io

        buffer = io.BytesIO()
        torch.save(embeddings_tensor, buffer)
        embeddings_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        result = handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]

        # Should match to float32 precision
        torch.testing.assert_close(result, embeddings_tensor, rtol=1e-6, atol=1e-6)

    def test_corrupted_pytorch_format_raises_error(self):
        """Test that corrupted PyTorch data raises an error."""
        # Create data that looks like it might be PyTorch format but is corrupted
        corrupted_data = (
            b"PK\x03\x04"  # ZIP magic bytes (PyTorch uses ZIP) but truncated/invalid
        )
        corrupted_data += b"invalid_pytorch_data" * 10

        embeddings_base64 = base64.b64encode(corrupted_data).decode("utf-8")

        class MockHandler:
            pass

        handler = MockHandler()
        handler._decode_prompt_embeds = BaseWorkerHandler._decode_prompt_embeds.__get__(handler)  # type: ignore[attr-defined]

        # Should raise ValueError about invalid PyTorch format
        with pytest.raises(ValueError, match="Failed to decode.*PyTorch"):
            handler._decode_prompt_embeds(embeddings_base64)  # type: ignore[attr-defined]


class TestUsageStatistics:
    """Tests for usage statistics calculation (v2.0.4 fix)."""

    def test_build_completion_usage_with_embeddings(self):
        """Test usage statistics when using embeddings."""
        from unittest.mock import Mock

        # Create mock RequestOutput
        mock_output = Mock()
        mock_output.prompt_token_ids = []  # Empty for embeddings
        mock_output.outputs = [Mock(token_ids=[1, 2, 3, 4, 5])]  # 5 completion tokens
        mock_output.num_cached_tokens = 0

        # Call with embedding_sequence_length
        embedding_sequence_length = 10
        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=embedding_sequence_length
        )

        # Verify usage statistics
        assert result["prompt_tokens"] == 10, "Should use embedding_sequence_length"
        assert result["completion_tokens"] == 5, "Should count completion tokens"
        assert result["total_tokens"] == 15, "Should sum prompt + completion"

    def test_build_completion_usage_with_text(self):
        """Test usage statistics when using text prompts (backward compatibility)."""
        from unittest.mock import Mock

        # Create mock RequestOutput with prompt_token_ids
        mock_output = Mock()
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5, 6, 7]  # 7 prompt tokens
        mock_output.outputs = [Mock(token_ids=[8, 9, 10])]  # 3 completion tokens
        mock_output.num_cached_tokens = 0

        # Call WITHOUT embedding_sequence_length (normal text)
        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=None
        )

        # Verify usage statistics
        assert result["prompt_tokens"] == 7, "Should use len(prompt_token_ids)"
        assert result["completion_tokens"] == 3, "Should count completion tokens"
        assert result["total_tokens"] == 10, "Should sum prompt + completion"

    def test_build_completion_usage_embeddings_override_token_ids(self):
        """Test that embedding_sequence_length takes precedence over prompt_token_ids."""
        from unittest.mock import Mock

        # Create mock RequestOutput with BOTH prompt_token_ids and embedding_sequence_length
        mock_output = Mock()
        mock_output.prompt_token_ids = [1, 2, 3]  # 3 tokens (should be ignored)
        mock_output.outputs = [Mock(token_ids=[4, 5])]  # 2 completion tokens
        mock_output.num_cached_tokens = 0

        # Call with embedding_sequence_length (should override)
        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=20
        )

        # Verify embedding_sequence_length takes precedence
        assert (
            result["prompt_tokens"] == 20
        ), "Should use embedding_sequence_length, not prompt_token_ids"
        assert result["completion_tokens"] == 2
        assert result["total_tokens"] == 22

    def test_build_completion_usage_no_prompt_tokens(self):
        """Test usage statistics when no prompt tokens available."""
        from unittest.mock import Mock

        # Create mock RequestOutput with no prompt info
        mock_output = Mock()
        mock_output.prompt_token_ids = None
        mock_output.outputs = [Mock(token_ids=[1, 2, 3])]
        mock_output.num_cached_tokens = 0

        # Call without embedding_sequence_length
        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=None
        )

        # Verify None handling
        assert result["prompt_tokens"] is None, "Should be None when no prompt info"
        assert result["completion_tokens"] == 3
        assert (
            result["total_tokens"] is None
        ), "Should be None when prompt_tokens is None"

    def test_build_completion_usage_with_cached_tokens(self):
        """Test that cached tokens are reported in prompt_tokens_details."""
        from unittest.mock import Mock

        # Create mock RequestOutput with cached tokens
        mock_output = Mock()
        mock_output.prompt_token_ids = [1, 2, 3, 4, 5]
        mock_output.outputs = [Mock(token_ids=[6, 7])]
        mock_output.num_cached_tokens = 3  # 3 tokens were cached

        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=None
        )

        # Verify cached tokens reported
        assert result["prompt_tokens"] == 5
        assert result["completion_tokens"] == 2
        assert result["prompt_tokens_details"] == {"cached_tokens": 3}

    def test_build_completion_usage_zero_sequence_length(self):
        """Test that embedding_sequence_length=0 is handled correctly."""
        from unittest.mock import Mock

        # Edge case: empty embeddings tensor (shouldn't happen in practice)
        mock_output = Mock()
        mock_output.prompt_token_ids = []
        mock_output.outputs = [Mock(token_ids=[1, 2])]
        mock_output.num_cached_tokens = 0

        result = BaseWorkerHandler._build_completion_usage(
            mock_output, embedding_sequence_length=0  # Edge case
        )

        # Should still calculate correctly
        assert result["prompt_tokens"] == 0
        assert result["completion_tokens"] == 2
        assert result["total_tokens"] == 2
