# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for prompt embeddings support in Dynamo."""

import base64
import io
import logging

import numpy as np
import pytest
import torch
import transformers
from openai import OpenAI

logger = logging.getLogger(__name__)

# Test model - small and fast for CI
TEST_MODEL = "Qwen/Qwen3-0.6B"


@pytest.fixture
def dynamo_client():
    """Create OpenAI client pointing to Dynamo frontend."""
    return OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1",
    )


def generate_test_embeddings(model_name: str, prompt: str) -> str:
    """
    Generate test embeddings for integration testing in PyTorch format.

    Args:
        model_name: Model to use for generating embeddings
        prompt: Text prompt to encode

    Returns:
        Base64-encoded embeddings in PyTorch format (torch.save)
    """
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_name)

        # Tokenize
        token_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Get embeddings
        embedding_layer = model.get_input_embeddings()
        prompt_embeds = embedding_layer(token_ids).squeeze(0)

        # Encode as PyTorch format (preserves shape and dtype - matches NIM-LLM)
        buffer = io.BytesIO()
        torch.save(prompt_embeds.cpu().detach(), buffer)
        buffer.seek(0)
        binary_data = buffer.read()

        return base64.b64encode(binary_data).decode("utf-8")
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {e}")
        raise


def generate_random_embeddings(shape=(10, 768)) -> str:
    """
    Generate random embeddings for testing in PyTorch format.

    Args:
        shape: Shape of embeddings tensor

    Returns:
        Base64-encoded PyTorch tensor
    """
    embeddings = torch.randn(*shape, dtype=torch.float32)
    buffer = io.BytesIO()
    torch.save(embeddings, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.nightly  # Run in nightly CI
@pytest.mark.gpu_1  # Requires 1 GPU
@pytest.mark.model(TEST_MODEL)
class TestPromptEmbedsIntegration:
    """Integration tests for prompt embeddings end-to-end flow."""

    def test_prompt_embeds_only(self, dynamo_client):
        """Test completion with only prompt_embeds (no prompt text)."""
        # Generate embeddings
        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Hello, world!")

        # Send request
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",  # Empty prompt when using embeddings
            max_tokens=10,
            temperature=0.7,
            extra_body={"prompt_embeds": encoded_embeds},
        )

        # Verify response
        assert response.choices, "Response should contain choices"
        assert len(response.choices) > 0, "Should have at least one choice"
        assert response.choices[0].text, "Choice should have text"
        assert response.choices[0].finish_reason, "Should have finish reason"

        # Verify usage stats
        assert response.usage, "Should have usage statistics"
        assert response.usage.completion_tokens > 0, "Should have completion tokens"

        logger.info(f"Generated text: {response.choices[0].text}")

    def test_both_prompt_and_embeds(self, dynamo_client):
        """Test that both prompt and prompt_embeds can be provided."""
        # Generate embeddings
        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Test prompt")

        # Send request with both fields
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="This text should be ignored",
            max_tokens=10,
            extra_body={"prompt_embeds": encoded_embeds},
        )

        # Should succeed - embeddings take precedence
        assert response.choices
        assert len(response.choices[0].text) > 0

    def test_prompt_embeds_with_max_tokens(self, dynamo_client):
        """Test that max_tokens is respected with embeddings."""
        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Short")

        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": encoded_embeds},
        )

        assert response.choices
        # Token count should be <= max_tokens
        assert response.usage.completion_tokens <= 5

    def test_prompt_embeds_with_temperature(self, dynamo_client):
        """Test embeddings with different temperature settings."""
        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Temperature test")

        # Temperature 0 (deterministic)
        response1 = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=10,
            temperature=0.0,
            extra_body={"prompt_embeds": encoded_embeds},
        )

        # Temperature 1.0 (more random)
        response2 = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=10,
            temperature=1.0,
            extra_body={"prompt_embeds": encoded_embeds},
        )

        assert response1.choices
        assert response2.choices
        # Both should produce output
        assert len(response1.choices[0].text) > 0
        assert len(response2.choices[0].text) > 0

    def test_prompt_embeds_streaming(self, dynamo_client):
        """Test streaming responses with embeddings."""
        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Streaming test")

        stream = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=10,
            stream=True,
            extra_body={"prompt_embeds": encoded_embeds},
        )

        chunks = []
        for chunk in stream:
            chunks.append(chunk)
            logger.debug(f"Received chunk: {chunk}")

        assert len(chunks) > 0, "Should receive at least one chunk"

        # Last chunk should have finish_reason
        if chunks[-1].choices:
            assert chunks[-1].choices[0].finish_reason is not None

    def test_invalid_base64(self, dynamo_client):
        """Test error handling for invalid base64."""
        with pytest.raises(Exception) as exc_info:
            dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=10,
                extra_body={"prompt_embeds": "not-valid-base64!!!"},
            )

        error_msg = str(exc_info.value).lower()
        assert "base64" in error_msg or "invalid" in error_msg

    def test_embeddings_too_small(self, dynamo_client):
        """Test that very small embeddings are rejected."""
        # Create tiny embeddings (< 100 bytes)
        tiny_data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        tiny_base64 = base64.b64encode(tiny_data.tobytes()).decode("utf-8")

        with pytest.raises(Exception) as exc_info:
            dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=10,
                extra_body={"prompt_embeds": tiny_base64},
            )

        error_msg = str(exc_info.value).lower()
        assert "100 bytes" in error_msg or "too small" in error_msg

    def test_embeddings_invalid_data(self, dynamo_client):
        """Test that truly invalid data (not a valid pickle/tensor) is rejected."""
        # Create completely invalid data that cannot be decoded as a tensor
        invalid_data = b"this is not a valid pickle or tensor format at all!!!"
        invalid_base64 = base64.b64encode(invalid_data).decode("utf-8")

        with pytest.raises(Exception) as exc_info:
            dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=10,
                extra_body={"prompt_embeds": invalid_base64},
            )

        # Should fail with decoding error
        error_msg = str(exc_info.value).lower()
        assert (
            "pytorch" in error_msg
            or "tensor" in error_msg
            or "invalid" in error_msg
            or "decode" in error_msg
        )

    def test_backward_compatibility(self, dynamo_client):
        """Test that normal text prompts still work (no regression)."""
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="Hello, how are you?",
            max_tokens=10,
            temperature=0.7,
        )

        assert response.choices
        assert len(response.choices[0].text) > 0
        assert response.usage.completion_tokens > 0


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.nightly  # Run in nightly CI
@pytest.mark.gpu_1  # Requires 1 GPU
@pytest.mark.model(TEST_MODEL)
class TestPromptEmbedsUsageStatistics:
    """Tests for usage statistics reporting with prompt embeddings (v2.0.4 fix)."""

    def test_usage_prompt_tokens_2d_embeddings(self, dynamo_client):
        """
        Test that prompt_tokens equals embedding sequence length for 2D tensors.

        This validates the v2.0.4 fix where prompt_tokens was incorrectly
        reported as 0 when using embeddings.
        """
        # Create 2D embeddings with known sequence length
        sequence_length = 15
        hidden_dim = 1024  # Qwen/Qwen3-0.6B hidden size
        embeddings = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)

        # Encode as PyTorch format
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.read()).decode()

        # Send request
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": embeddings_base64},
        )

        # Verify usage statistics
        assert response.usage, "Should have usage statistics"
        assert response.usage.prompt_tokens == sequence_length, (
            f"prompt_tokens should equal embedding sequence length ({sequence_length}), "
            f"got {response.usage.prompt_tokens}"
        )
        assert response.usage.completion_tokens > 0, "Should have completion tokens"
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        ), "total_tokens should equal prompt_tokens + completion_tokens"

        logger.info(
            f"Usage stats correct: prompt_tokens={response.usage.prompt_tokens}, "
            f"completion_tokens={response.usage.completion_tokens}, "
            f"total_tokens={response.usage.total_tokens}"
        )

    def test_usage_prompt_tokens_3d_embeddings(self, dynamo_client):
        """
        Test that prompt_tokens equals embedding sequence length for 3D tensors.

        Shape: (batch, sequence_length, hidden_dim)
        Should extract sequence_length from shape[1].
        """
        # Create 3D embeddings (batch=1, seq_len=12, hidden_dim=1024)
        batch_size = 1
        sequence_length = 12
        hidden_dim = 1024
        embeddings = torch.randn(
            batch_size, sequence_length, hidden_dim, dtype=torch.float32
        )

        # Encode as PyTorch format
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.read()).decode()

        # Send request
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": embeddings_base64},
        )

        # Verify usage statistics
        assert response.usage, "Should have usage statistics"
        assert response.usage.prompt_tokens == sequence_length, (
            f"prompt_tokens should equal embedding sequence length ({sequence_length}), "
            f"got {response.usage.prompt_tokens}"
        )
        assert response.usage.completion_tokens > 0, "Should have completion tokens"
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        ), "total_tokens should equal prompt_tokens + completion_tokens"

        logger.info(
            f"3D tensor usage stats: prompt_tokens={response.usage.prompt_tokens}, "
            f"completion_tokens={response.usage.completion_tokens}, "
            f"total_tokens={response.usage.total_tokens}"
        )

    def test_usage_prompt_tokens_not_zero(self, dynamo_client):
        """
        Regression test: Ensure prompt_tokens is never 0 when using embeddings.

        This was the bug in versions < v2.0.4.
        """
        # Create embeddings
        embeddings = torch.randn(20, 1024, dtype=torch.float32)

        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.read()).decode()

        # Send request
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=3,
            extra_body={"prompt_embeds": embeddings_base64},
        )

        # THE FIX: prompt_tokens must NOT be 0
        assert (
            response.usage.prompt_tokens != 0
        ), "BUG REGRESSION: prompt_tokens is 0! This was the bug in v2.0.3."
        assert (
            response.usage.prompt_tokens == 20
        ), f"Expected prompt_tokens=20, got {response.usage.prompt_tokens}"

    def test_usage_comparison_embeddings_vs_text(self, dynamo_client):
        """
        Compare usage statistics between embeddings and text prompts.

        Both should report accurate prompt_tokens.
        """
        # 1. Text prompt
        text_response = dynamo_client.completions.create(
            model=TEST_MODEL, prompt="Hello, how are you?", max_tokens=5  # ~5-6 tokens
        )

        text_prompt_tokens = text_response.usage.prompt_tokens
        assert text_prompt_tokens > 0, "Text prompt should have > 0 prompt_tokens"

        # 2. Embeddings prompt with same-ish length
        embeddings = torch.randn(text_prompt_tokens, 1024, dtype=torch.float32)
        buffer = io.BytesIO()
        torch.save(embeddings, buffer)
        buffer.seek(0)
        embeddings_base64 = base64.b64encode(buffer.read()).decode()

        embed_response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": embeddings_base64},
        )

        embed_prompt_tokens = embed_response.usage.prompt_tokens

        # Both should report prompt_tokens
        assert embed_prompt_tokens == text_prompt_tokens, (
            f"Embeddings and text with same length should report same prompt_tokens: "
            f"text={text_prompt_tokens}, embeddings={embed_prompt_tokens}"
        )

        logger.info(
            f"Comparison: text.prompt_tokens={text_prompt_tokens}, "
            f"embeddings.prompt_tokens={embed_prompt_tokens}"
        )


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.nightly  # Run in nightly CI
@pytest.mark.gpu_1  # Requires 1 GPU
@pytest.mark.model(TEST_MODEL)
class TestPromptEmbedsPerformance:
    """Performance and behavior tests for embeddings."""

    def test_embeddings_vs_text_timing(self, dynamo_client):
        """Compare timing between embeddings and text (embeddings should be faster)."""
        import time

        test_prompt = "This is a test prompt for performance comparison"

        # Test with text
        start = time.time()
        response_text = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt=test_prompt,
            max_tokens=5,
        )
        text_time = time.time() - start

        # Test with embeddings
        encoded_embeds = generate_test_embeddings(TEST_MODEL, test_prompt)
        start = time.time()
        response_embeds = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": encoded_embeds},
        )
        embeds_time = time.time() - start

        logger.info(f"Text time: {text_time:.3f}s, Embeddings time: {embeds_time:.3f}s")

        # Both should work
        assert response_text.choices
        assert response_embeds.choices

        # Embeddings should be faster (skips tokenization)
        # Note: In practice, this depends on network latency and other factors
        # So we just log the comparison rather than asserting

    def test_large_embeddings(self, dynamo_client):
        """Test with large embeddings (close to 10MB limit)."""
        # Create embeddings close to limit using PyTorch format
        # Target: ~7MB decoded (to stay well under 10MB with PyTorch overhead)
        # 7MB / 4 bytes per float32 = 1.75M floats
        large_shape = (1700, 1024)  # 1740800 float32s ≈ 6.6MB

        large_embeds = torch.randn(large_shape, dtype=torch.float32)

        # Use PyTorch format (torch.save)
        buffer = io.BytesIO()
        torch.save(large_embeds, buffer)
        buffer.seek(0)
        large_bytes = buffer.read()
        large_base64 = base64.b64encode(large_bytes).decode("utf-8")

        logger.info(
            f"Large embeddings: decoded size={len(large_bytes)} bytes "
            f"({len(large_bytes)/1024/1024:.2f}MB), base64 size={len(large_base64)} bytes"
        )

        # Should work (under 10MB limit)
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": large_base64},
        )

        assert response.choices
        assert len(large_bytes) < 10 * 1024 * 1024, "Test data should be under 10MB"


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.nightly  # Run in nightly CI
@pytest.mark.gpu_1  # Requires 1 GPU
@pytest.mark.model(TEST_MODEL)
class TestPromptEmbedsFormats:
    """Test different embedding formats."""

    def test_pytorch_format_preferred(self, dynamo_client):
        """Test that PyTorch format (torch.save) works correctly."""
        # Use generate_test_embeddings to get embeddings with correct dimensions
        pytorch_base64 = generate_test_embeddings(TEST_MODEL, "PyTorch format test")

        # Should succeed - PyTorch format is the supported format
        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": pytorch_base64},
        )

        assert response.choices
        assert len(response.choices[0].text) > 0

    def test_format_compatibility_note(self, dynamo_client):
        """
        NOTE: NumPy .npy and certain other pickle-based formats may be loadable by torch.load()
        and thus accepted by the system. However, PyTorch format (torch.save) is the officially
        supported and recommended format for maximum compatibility.

        This test documents the expected behavior without enforcing strict rejection.
        """
        # Use generate_test_embeddings to get embeddings with correct dimensions
        pytorch_base64 = generate_test_embeddings(
            TEST_MODEL, "Format compatibility test"
        )

        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": pytorch_base64},
        )

        assert response.choices
        logger.info("PyTorch format (torch.save) is the recommended format")


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.nightly  # Run in nightly CI
@pytest.mark.gpu_1  # Requires 1 GPU
@pytest.mark.model(TEST_MODEL)
class TestPromptEmbedsConcurrency:
    """Test concurrent requests with embeddings."""

    def test_concurrent_embeddings_requests(self, dynamo_client):
        """Test multiple concurrent requests with embeddings."""
        import concurrent.futures

        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Concurrent test")

        def send_request():
            return dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=5,
                extra_body={"prompt_embeds": encoded_embeds},
            )

        # Send 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert len(results) == 5
        for response in results:
            assert response.choices
            assert len(response.choices[0].text) > 0

    def test_mixed_requests(self, dynamo_client):
        """Test mixing normal and embeddings requests."""
        import concurrent.futures

        encoded_embeds = generate_test_embeddings(TEST_MODEL, "Mixed test")

        def send_text_request():
            return dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="Hello",
                max_tokens=5,
            )

        def send_embeds_request():
            return dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=5,
                extra_body={"prompt_embeds": encoded_embeds},
            )

        # Mix of text and embeddings requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(send_text_request),
                executor.submit(send_embeds_request),
                executor.submit(send_text_request),
                executor.submit(send_embeds_request),
            ]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        # All should succeed
        assert len(results) == 4
        for response in results:
            assert response.choices
