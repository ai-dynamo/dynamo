# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for VideoGenerationWorkerHandler."""

import asyncio
import base64
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from dynamo.sglang.request_handlers.video_generation.video_generation_handler import (
    VideoGenerationWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,  # No GPU needed for unit tests
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_component():
    """Mock Dynamo Component."""
    return MagicMock()


@pytest.fixture
def mock_generator():
    """Mock SGLang DiffGenerator."""
    generator = MagicMock()
    generator.generate = MagicMock()
    return generator


@pytest.fixture
def mock_config():
    """Mock Config object."""
    config = MagicMock()
    config.dynamo_args = MagicMock()
    config.dynamo_args.video_generation_fs_url = "file:///tmp/dynamo_videos"
    return config


@pytest.fixture
def mock_fs():
    """Mock fsspec filesystem."""
    fs = MagicMock()
    fs.pipe = MagicMock()
    return fs


@pytest.fixture
def mock_context():
    """Mock Context object."""
    context = MagicMock()
    context.id = MagicMock(return_value="test-context-id")
    context.trace_id = "test-trace-id"
    context.span_id = "test-span-id"
    context.is_cancelled = MagicMock(return_value=False)
    return context


def _make_fake_frames(n=5, width=64, height=64):
    """Create a list of fake numpy frames for testing."""
    return [np.random.randint(0, 255, (height, width, 3), dtype=np.uint8) for _ in range(n)]


@pytest.fixture
def handler(mock_component, mock_generator, mock_config, mock_fs):
    """Create VideoGenerationWorkerHandler instance."""
    return VideoGenerationWorkerHandler(
        component=mock_component,
        generator=mock_generator,
        config=mock_config,
        publisher=None,
        fs=mock_fs,
    )


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


class TestVideoGenerationWorkerHandler:
    """Test suite for VideoGenerationWorkerHandler."""

    def test_initialization(self, handler, mock_generator, mock_fs):
        """Test handler initialization."""
        assert handler.generator is mock_generator
        assert handler.fs is mock_fs
        assert handler.fs_url == "file:///tmp/dynamo_videos"
        assert isinstance(handler._generate_lock, asyncio.Lock)

    @patch("torch.cuda.empty_cache")
    def test_cleanup(self, mock_empty_cache, handler):
        """Test cleanup method."""
        handler.cleanup()
        mock_empty_cache.assert_called_once()

    def test_parse_size(self, handler):
        """Test _parse_size method."""
        assert handler._parse_size("832x480") == (832, 480)
        assert handler._parse_size("1920x1080") == (1920, 1080)

    def test_parse_size_invalid(self, handler):
        """Test _parse_size raises on bad input."""
        with pytest.raises(ValueError, match="Invalid size format"):
            handler._parse_size("invalid")

    def test_encode_base64(self, handler):
        """Test _encode_base64 method."""
        data = b"test video data"
        expected = base64.b64encode(data).decode("utf-8")
        assert handler._encode_base64(data) == expected

    # ------------------------------------------------------------------
    # Seed handling
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @patch(
        "dynamo.sglang.request_handlers.video_generation.video_generation_handler.VideoGenerationWorkerHandler._frames_to_video",
        return_value=b"fake-mp4",
    )
    async def test_seed_zero_is_preserved(self, _mock_ftv, handler, mock_context):
        """seed=0 is a valid value and must not be replaced by a random seed."""
        handler.generator.generate = Mock(return_value={"frames": _make_fake_frames()})

        request = {
            "prompt": "test",
            "model": "test-model",
            "size": "64x64",
            "response_format": "b64_json",
            "nvext": {"seed": 0, "num_inference_steps": 1, "fps": 8, "num_frames": 5},
        }

        async for _ in handler.generate(request, mock_context):
            pass

        call_kwargs = handler.generator.generate.call_args
        sampling_params = call_kwargs.kwargs.get(
            "sampling_params_kwargs", call_kwargs[1].get("sampling_params_kwargs")
        )
        assert sampling_params["seed"] == 0, "seed=0 should be preserved, not replaced"

    @pytest.mark.asyncio
    @patch(
        "dynamo.sglang.request_handlers.video_generation.video_generation_handler.VideoGenerationWorkerHandler._frames_to_video",
        return_value=b"fake-mp4",
    )
    async def test_seed_none_gets_random(self, _mock_ftv, handler, mock_context):
        """seed=None should produce a random seed (non-None int)."""
        handler.generator.generate = Mock(return_value={"frames": _make_fake_frames()})

        request = {
            "prompt": "test",
            "model": "test-model",
            "size": "64x64",
            "response_format": "b64_json",
            "nvext": {"num_inference_steps": 1, "fps": 8, "num_frames": 5},
            # seed not set -> defaults to None
        }

        async for _ in handler.generate(request, mock_context):
            pass

        call_kwargs = handler.generator.generate.call_args
        sampling_params = call_kwargs.kwargs.get(
            "sampling_params_kwargs", call_kwargs[1].get("sampling_params_kwargs")
        )
        assert isinstance(sampling_params["seed"], int)
        assert sampling_params["seed"] >= 0

    # ------------------------------------------------------------------
    # Generate end-to-end
    # ------------------------------------------------------------------

    @pytest.mark.asyncio
    @patch(
        "dynamo.sglang.request_handlers.video_generation.video_generation_handler.VideoGenerationWorkerHandler._frames_to_video",
        return_value=b"fake-mp4",
    )
    async def test_generate_success_url_format(self, _mock_ftv, handler, mock_context):
        """Test successful video generation with URL response format."""
        handler.generator.generate = Mock(return_value={"frames": _make_fake_frames()})

        request = {
            "prompt": "A curious raccoon",
            "model": "test-model",
            "size": "832x480",
            "response_format": "url",
            "nvext": {"num_inference_steps": 1, "fps": 8, "num_frames": 5},
        }

        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        assert len(results) == 1
        response = results[0]
        assert "data" in response
        assert len(response["data"]) == 1
        assert response["data"][0]["url"].startswith("file:///tmp/dynamo_videos/")

    @pytest.mark.asyncio
    @patch(
        "dynamo.sglang.request_handlers.video_generation.video_generation_handler.VideoGenerationWorkerHandler._frames_to_video",
        return_value=b"fake-mp4",
    )
    async def test_generate_success_b64_format(self, _mock_ftv, handler, mock_context):
        """Test successful video generation with base64 response format."""
        handler.generator.generate = Mock(return_value={"frames": _make_fake_frames()})

        request = {
            "prompt": "A blue ocean",
            "model": "test-model",
            "size": "832x480",
            "response_format": "b64_json",
            "nvext": {"num_inference_steps": 1, "fps": 8, "num_frames": 5},
        }

        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        assert len(results) == 1
        response = results[0]
        assert "data" in response
        assert len(response["data"]) == 1
        b64 = response["data"][0]["b64_json"]
        decoded = base64.b64decode(b64)
        assert decoded == b"fake-mp4"

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, handler, mock_context):
        """Test error response on generator failure."""
        handler.generator.generate = Mock(side_effect=RuntimeError("GPU OOM"))

        request = {
            "prompt": "test",
            "model": "test-model",
            "size": "832x480",
            "response_format": "url",
            "nvext": {"num_inference_steps": 1, "fps": 8, "num_frames": 5},
        }

        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        assert len(results) == 1
        assert "GPU OOM" in results[0]["error"]
        assert results[0]["data"] == []


# ---------------------------------------------------------------------------
# Concurrency Safety Tests
# ---------------------------------------------------------------------------


class ConcurrencyTracker:
    """Mock generator.generate() that tracks concurrent access."""

    def __init__(self, sleep_seconds: float = 0.15):
        self._lock = threading.Lock()
        self._active_count = 0
        self.max_concurrent = 0
        self._sleep_seconds = sleep_seconds

    def generate(self, **kwargs):
        with self._lock:
            self._active_count += 1
            if self._active_count > self.max_concurrent:
                self.max_concurrent = self._active_count

        # Simulate work -- creates a window where concurrent access is observable
        time.sleep(self._sleep_seconds)

        with self._lock:
            self._active_count -= 1

        return {"frames": _make_fake_frames()}


class TestVideoHandlerConcurrency:
    """Verify that concurrent requests are serialized through _generate_lock."""

    def _make_handler(self, tracker: ConcurrencyTracker):
        component = MagicMock()
        config = MagicMock()
        config.dynamo_args = MagicMock()
        config.dynamo_args.video_generation_fs_url = "file:///tmp/test"

        generator = MagicMock()
        generator.generate = tracker.generate

        handler = VideoGenerationWorkerHandler(
            component=component,
            generator=generator,
            config=config,
            publisher=None,
            fs=MagicMock(),
        )
        return handler

    def _make_request(self):
        return {
            "prompt": "concurrency test",
            "model": "test-model",
            "size": "64x64",
            "response_format": "b64_json",
            "nvext": {"num_inference_steps": 1, "fps": 8, "num_frames": 5},
        }

    async def _drain_generator(self, handler, request, context):
        async for _ in handler.generate(request, context):
            pass

    @pytest.mark.asyncio
    @patch(
        "dynamo.sglang.request_handlers.video_generation.video_generation_handler.VideoGenerationWorkerHandler._frames_to_video",
        return_value=b"fake-mp4",
    )
    async def test_concurrent_requests_are_serialized(self, _mock_ftv):
        """Fire N concurrent requests and assert max_concurrent == 1."""
        tracker = ConcurrencyTracker(sleep_seconds=0.15)
        handler = self._make_handler(tracker)
        n_requests = 3

        contexts = []
        for i in range(n_requests):
            ctx = MagicMock()
            ctx.id = MagicMock(return_value=f"req-{i}")
            ctx.trace_id = f"trace-{i}"
            ctx.span_id = f"span-{i}"
            ctx.is_cancelled = MagicMock(return_value=False)
            contexts.append(ctx)

        tasks = [
            asyncio.create_task(
                self._drain_generator(handler, self._make_request(), contexts[i])
            )
            for i in range(n_requests)
        ]
        await asyncio.gather(*tasks)

        assert tracker.max_concurrent == 1, (
            f"Expected max_concurrent=1 (serialized), got {tracker.max_concurrent}. "
            "The _generate_lock is not properly serializing generator access."
        )
