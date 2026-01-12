# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end tests for prompt embeddings support in Dynamo.

These tests validate behavior that cannot be covered by Rust unit tests:
- Streaming responses with embeddings
- Python-side tensor decoding errors
- Usage statistics from worker (the v2.0.4 bug fix)
- Large payload handling through NATS
- Concurrent request handling

Validation tests (base64, size limits, empty prompt) are covered by Rust unit tests
in lib/llm/src/protocols/openai/completions.rs
"""

import base64
import concurrent.futures
import io
import logging

import pytest
import torch
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


def create_embeddings_base64(shape: tuple[int, ...]) -> str:
    """Create random embeddings tensor and return as base64-encoded PyTorch format."""
    embeddings = torch.randn(*shape, dtype=torch.float32)
    buffer = io.BytesIO()
    torch.save(embeddings, buffer)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


@pytest.mark.integration
@pytest.mark.vllm
@pytest.mark.nightly
@pytest.mark.gpu_1
@pytest.mark.model(TEST_MODEL)
class TestPromptEmbedsE2E:
    """
    End-to-end tests for prompt embeddings.

    These tests require a running Dynamo instance with vLLM backend.
    They validate behavior that Rust unit tests cannot cover.
    """

    def test_streaming_with_embeddings(self, dynamo_client):
        """
        Test streaming responses work correctly with embeddings.

        This is E2E only - Rust tests can't verify streaming behavior.
        """
        embeddings_base64 = create_embeddings_base64((10, 1024))

        stream = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=10,
            stream=True,
            extra_body={"prompt_embeds": embeddings_base64},
        )

        chunks = list(stream)

        assert len(chunks) > 0, "Should receive at least one chunk"
        # Last chunk should have finish_reason
        if chunks[-1].choices:
            assert chunks[-1].choices[0].finish_reason is not None

    def test_invalid_tensor_data_rejected(self, dynamo_client):
        """
        Test that invalid tensor data is properly rejected by Python decoder.

        This tests the Python-side torch.load() error handling, which
        Rust validation cannot cover (Rust only checks base64 and size).
        """
        # Create data that passes Rust validation (valid base64, >100 bytes)
        # but fails Python torch.load()
        invalid_data = b"this is not a valid pytorch tensor format!" * 10
        invalid_base64 = base64.b64encode(invalid_data).decode("utf-8")

        with pytest.raises(Exception) as exc_info:
            dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=5,
                extra_body={"prompt_embeds": invalid_base64},
            )

        error_msg = str(exc_info.value).lower()
        assert any(
            keyword in error_msg
            for keyword in ["pytorch", "tensor", "invalid", "decode", "error"]
        ), f"Expected tensor decode error, got: {error_msg}"

    def test_usage_prompt_tokens_not_zero(self, dynamo_client):
        """
        CRITICAL REGRESSION TEST: Ensure prompt_tokens is correctly reported.

        This validates the v2.0.4 fix where prompt_tokens was incorrectly
        reported as 0 when using embeddings. The worker extracts sequence
        length from tensor shape and includes it in completion_usage.

        Rust tests cannot verify this - it requires E2E validation.
        """
        sequence_length = 20
        embeddings_base64 = create_embeddings_base64((sequence_length, 1024))

        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=3,
            extra_body={"prompt_embeds": embeddings_base64},
        )

        assert response.usage is not None, "Should have usage statistics"
        assert (
            response.usage.prompt_tokens != 0
        ), "BUG REGRESSION: prompt_tokens is 0! This was the bug in v2.0.3."
        assert (
            response.usage.prompt_tokens == sequence_length
        ), f"Expected prompt_tokens={sequence_length}, got {response.usage.prompt_tokens}"
        assert response.usage.total_tokens == (
            response.usage.prompt_tokens + response.usage.completion_tokens
        ), "total_tokens should equal prompt_tokens + completion_tokens"

    def test_large_embeddings_through_nats(self, dynamo_client):
        """
        Test large embeddings are handled correctly through NATS.

        This validates the NATS max_payload configuration (15MB) handles
        large embedding payloads. Rust unit tests can't test this E2E path.
        """
        # Create ~7MB embeddings (well under 10MB limit, but large enough to stress NATS)
        large_shape = (1700, 1024)  # ~6.6MB of float32 data
        large_embeds = torch.randn(large_shape, dtype=torch.float32)

        buffer = io.BytesIO()
        torch.save(large_embeds, buffer)
        buffer.seek(0)
        large_bytes = buffer.read()
        large_base64 = base64.b64encode(large_bytes).decode("utf-8")

        logger.info(
            f"Testing large embeddings: {len(large_bytes)/1024/1024:.2f}MB decoded"
        )

        response = dynamo_client.completions.create(
            model=TEST_MODEL,
            prompt="",
            max_tokens=5,
            extra_body={"prompt_embeds": large_base64},
        )

        assert response.choices, "Large embeddings should produce valid response"
        assert len(large_bytes) < 10 * 1024 * 1024, "Test data should be under 10MB"

    def test_concurrent_embeddings_requests(self, dynamo_client):
        """
        Test concurrent requests with embeddings are handled correctly.

        This validates the worker can handle multiple embedding requests
        simultaneously without race conditions or resource conflicts.
        """
        embeddings_base64 = create_embeddings_base64((10, 1024))

        def send_request():
            return dynamo_client.completions.create(
                model=TEST_MODEL,
                prompt="",
                max_tokens=5,
                extra_body={"prompt_embeds": embeddings_base64},
            )

        # Send 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(send_request) for _ in range(5)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        assert len(results) == 5, "All concurrent requests should complete"
        for response in results:
            assert response.choices, "Each response should have choices"
            assert len(response.choices[0].text) > 0, "Each response should have text"
