# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for video diffusion components.

Tests for Modality enum, DiffusionConfig, VideoGenerationHandler helpers,
video protocol types, and concurrency safety.

These tests do NOT require visual_gen, torch, or GPU - they test logic only.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from dynamo.common.protocols.image_protocol import (
    ImageData,
    ImageNvExt,
    NvCreateImageRequest,
    NvImagesResponse,
)
from dynamo.trtllm.configs.diffusion_config import DiffusionConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

# [gluo FIXME] many parts of the test are validated as part of test_trtllm_video_diffusion.py,
# we should have common test suite for diffusion and additional tests for different modalities.

# =============================================================================
# Part 1: Modality Enum Tests
# =============================================================================

# This part of the test has been covered in test_trtllm_video_diffusion.py

# =============================================================================
# Part 2: DiffusionConfig Tests
# =============================================================================

# This part of the test has been covered in test_trtllm_video_diffusion.py

# =============================================================================
# Part 3: VideoGenerationHandler Helper Tests
# =============================================================================


class MockDiffusionConfig:
    """Mock config for testing handler helpers without full DiffusionConfig."""

    default_width: int = 832
    default_height: int = 480
    default_num_frames: int = 81
    default_num_images_per_prompt: int = 1
    default_fps: int = 24
    default_seconds: int = 4
    max_width: int = 4096
    max_height: int = 4096


class TestImageHandlerParseSize:
    """Tests for ImageGenerationHandler._parse_size method.

    We test the method logic by creating a minimal mock handler.
    """

    def setup_method(self):
        """Set up mock handler for each test."""
        # Import here to avoid issues if handler has complex imports
        from dynamo.trtllm.request_handlers.diffusion.image_handler import (
            ImageGenerationHandler,
        )

        # Create handler with mocked dependencies
        self.handler = object.__new__(ImageGenerationHandler)
        self.handler.config = MockDiffusionConfig()

    def test_parse_size_valid(self):
        """Test valid 'WxH' string parsing."""
        width, height = self.handler._parse_size("832x480")
        assert width == 832
        assert height == 480

    def test_parse_size_different_dimensions(self):
        """Test parsing various dimension strings."""
        assert self.handler._parse_size("1920x1080") == (1920, 1080)
        assert self.handler._parse_size("640x360") == (640, 360)
        assert self.handler._parse_size("1x1") == (1, 1)

    def test_parse_size_none(self):
        """Test None returns defaults."""
        width, height = self.handler._parse_size(None)
        assert width == MockDiffusionConfig.default_width
        assert height == MockDiffusionConfig.default_height

    def test_parse_size_empty_string(self):
        """Test empty string returns defaults."""
        width, height = self.handler._parse_size("")
        assert width == MockDiffusionConfig.default_width
        assert height == MockDiffusionConfig.default_height

    def test_parse_size_invalid_format(self):
        """Test invalid format returns defaults with warning."""
        # No 'x' separator
        assert self.handler._parse_size("832480") == (832, 480)

        # Only one number
        assert self.handler._parse_size("832") == (832, 480)

        # Non-numeric
        assert self.handler._parse_size("widthxheight") == (832, 480)

        # Trailing 'x'
        assert self.handler._parse_size("832x") == (832, 480)

    def test_parse_size_exceeds_max_width(self):
        """Test that width exceeding max_width raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.handler._parse_size("5000x480")
        assert "width 5000 exceeds max_width 4096" in str(exc_info.value)
        assert "safety check to prevent out-of-memory" in str(exc_info.value)

    def test_parse_size_exceeds_max_height(self):
        """Test that height exceeding max_height raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            self.handler._parse_size("832x5000")
        assert "height 5000 exceeds max_height 4096" in str(exc_info.value)

    def test_parse_size_exceeds_both_dimensions(self):
        """Test that both dimensions exceeding raises ValueError with both errors."""
        with pytest.raises(ValueError) as exc_info:
            self.handler._parse_size("10000x10000")
        error_msg = str(exc_info.value)
        assert "width 10000 exceeds max_width 4096" in error_msg
        assert "height 10000 exceeds max_height 4096" in error_msg

    def test_parse_size_at_max_boundary(self):
        """Test that dimensions exactly at max are allowed."""
        # Should not raise - exactly at limit
        width, height = self.handler._parse_size("4096x4096")
        assert width == 4096
        assert height == 4096


# =============================================================================
# Part 4: Image Protocol Tests
# =============================================================================


class TestNvCreateImageRequest:
    """Tests for NvCreateImageRequest protocol type."""

    def test_required_fields(self):
        """Test that prompt and model are required."""
        req = NvCreateImageRequest(prompt="A cat", model="black-forest-labs/FLUX.1-dev")
        assert req.prompt == "A cat"
        assert req.model == "black-forest-labs/FLUX.1-dev"

    def test_required_fields_missing_prompt(self):
        """Test that missing prompt raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            NvCreateImageRequest(model="black-forest-labs/FLUX.1-dev")  # type: ignore

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        req = NvCreateImageRequest(prompt="A cat")

        # [gluo NOTE] in protocol the model is optional, but actually Dynamo will
        # fill with default value if not provided.
        assert req.model is None
        assert req.n is None
        assert req.quality is None
        assert req.style is None
        assert req.user is None
        assert req.moderation is None
        assert req.input_reference is None
        assert req.size is None
        assert req.response_format is None
        assert req.nvext is None

    def test_full_request_valid(self):
        """Test a fully populated request with nvext."""
        req = NvCreateImageRequest(
            prompt="A majestic lion",
            model="black-forest-labs/FLUX.1-dev",
            size="1920x1080",
            response_format="b64_json",
            nvext=ImageNvExt(
                annotations=["tag1", "tag2"],
                negative_prompt="blurry, low quality",
                num_inference_steps=30,
                guidance_scale=7.5,
                seed=42,
            ),
        )

        assert req.prompt == "A majestic lion"
        assert req.model == "black-forest-labs/FLUX.1-dev"
        assert req.size == "1920x1080"
        assert req.response_format == "b64_json"
        assert req.nvext.annotations == ["tag1", "tag2"]
        assert req.nvext.negative_prompt == "blurry, low quality"
        assert req.nvext.num_inference_steps == 30
        assert req.nvext.guidance_scale == 7.5
        assert req.nvext.seed == 42


class TestImageData:
    """Tests for ImageData protocol type."""

    def test_url_only(self):
        """Test ImageData with URL only."""
        data = ImageData(url="/tmp/image.png")
        assert data.url == "/tmp/image.png"
        assert data.b64_json is None

    def test_b64_only(self):
        """Test ImageData with base64 only."""
        data = ImageData(b64_json="SGVsbG8gV29ybGQ=")
        assert data.url is None
        assert data.b64_json == "SGVsbG8gV29ybGQ="

    def test_both_fields(self):
        """Test ImageData with both fields (unusual but valid)."""
        data = ImageData(url="/tmp/image.png", b64_json="SGVsbG8=")
        assert data.url == "/tmp/image.png"
        assert data.b64_json == "SGVsbG8="

    def test_empty_defaults(self):
        """Test ImageData with no arguments."""
        data = ImageData()
        assert data.url is None
        assert data.b64_json is None


class TestNvImagesResponse:
    """Tests for NvImagesResponse protocol type."""

    def test_default_values(self):
        """Test default values for completed response."""
        response = NvImagesResponse(
            created=1234567890,
        )
        assert response.created == 1234567890
        assert response.data == []

    def test_with_image_data(self):
        """Test response with image data."""
        image = ImageData(url="/tmp/output.png")
        response = NvImagesResponse(
            created=1234567890,
            data=[image],
        )
        assert len(response.data) == 1
        assert response.data[0].url == "/tmp/output.png"

    def test_model_dump(self):
        """Test serialization with model_dump()."""
        response = NvImagesResponse(
            id="req-123",
            created=1234567890,
            data=[ImageData(url="/tmp/image.png")],
        )

        dumped = response.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["created"] == 1234567890
        assert len(dumped["data"]) == 1
        assert dumped["data"][0]["url"] == "/tmp/image.png"


# =============================================================================
# Part 5: DiffusionEngine Unit Tests
# =============================================================================

# This part of the test has been covered in test_trtllm_video_diffusion.py

# =============================================================================
# Part 6: Concurrency Safety Tests
# =============================================================================

# [gluo NOTE] this part have been covered in test_trtllm_video_diffusion.py,
# but need sanity check with image generation as the handler is different.
# Could be merged once a base DiffusionHandler is introduced.


class ConcurrencyTracker:
    """Mock replacement for ``DiffusionEngine.generate()`` that records
    the peak number of threads executing it simultaneously.

    What it mocks:
        ``engine.generate(**kwargs)`` — the blocking GPU call inside
        ``VideoGenerationHandler``.  The handler dispatches this via
        ``asyncio.to_thread()``, so each request runs ``generate()``
        in a separate OS thread.

    What it focuses on:
        Detecting *concurrent* entry into ``generate()``.  It does NOT
        test correctness of generated frames, GPU memory, or CUDA
        streams — only whether multiple threads overlap inside the call.

    How it works:
        1. On entry: atomically increment ``_active_count`` and update
           the high-water mark ``max_concurrent``.
        2. Sleep for ``sleep_seconds`` to hold the thread inside the
           function, creating a window where other threads *would*
           overlap if nothing serializes them.
        3. On exit: atomically decrement ``_active_count``.

    After the test, inspect ``max_concurrent``:
        - 1  → accesses were serialized (lock is working).
        - >1 → concurrent access occurred (lock is missing/broken).
    """

    def __init__(self, sleep_seconds: float = 0.1):
        self._active_count = 0
        self._lock = threading.Lock()
        self.max_concurrent = 0
        self.sleep_seconds = sleep_seconds

    def generate(self, **kwargs):
        """Mock engine.generate() that tracks concurrent access."""
        with self._lock:
            self._active_count += 1
            if self._active_count > self.max_concurrent:
                self.max_concurrent = self._active_count

        # Hold the thread here to widen the overlap window.  Without
        # serialization, other threads will enter generate() during
        # this sleep and bump _active_count above 1.
        time.sleep(self.sleep_seconds)

        with self._lock:
            self._active_count -= 1

        # Return a mock MediaOutput with a video tensor
        return SimpleNamespace(
            video=None,
            image=torch.zeros((1, 64, 64, 3), dtype=torch.uint8),
            audio=None,
        )


class TestVideoHandlerConcurrency:
    """Verifies that ``ImageGenerationHandler`` serializes access to the
    underlying ``engine.generate()`` call.

    Why this matters:
        The visual_gen pipeline is a global singleton with mutable state,
        unprotected CUDA graph caches, and shared config objects.  It is
        NOT thread-safe.  ``ImageGenerationHandler`` dispatches generate()
        via ``asyncio.to_thread()``, which runs each request in a
        separate OS thread.  Without an ``asyncio.Lock`` guarding the
        call, concurrent requests would enter generate() simultaneously
        and corrupt shared pipeline state.

    How the test works:
        1. Wires a ``ConcurrencyTracker`` as the mock engine so that
           each generate() call sleeps long enough for overlapping
           threads to be observable.
        2. Fires N requests concurrently with ``asyncio.gather()``,
           each of which calls ``handler.generate()`` → ``asyncio.to_thread()``
           → ``tracker.generate()``.
        3. Asserts ``tracker.max_concurrent == 1``: only one thread was
           inside generate() at any point.

    Why it works:
        - ``asyncio.gather()`` schedules all coroutines on the same
          event loop, so they all reach ``asyncio.to_thread()``
          nearly simultaneously.
        - Without the handler's ``asyncio.Lock``, each coroutine
          immediately spawns a thread, and those threads overlap
          inside ``tracker.generate()`` during the sleep window →
          ``max_concurrent > 1``.
        - With the lock, only one coroutine enters the
          ``async with self._generate_lock`` block at a time; the
          others suspend cooperatively on the event loop.  So only
          one thread is ever inside generate() → ``max_concurrent == 1``.
    """

    def _make_handler(self):
        """Create a ImageGenerationHandler with mock engine and config."""
        from dynamo.trtllm.request_handlers.diffusion.image_handler import (
            ImageGenerationHandler,
        )

        tracker = ConcurrencyTracker(sleep_seconds=0.1)

        mock_engine = MagicMock()
        mock_engine.generate = tracker.generate

        config = DiffusionConfig(
            media_output_fs_url="file:///tmp/test_media",
            default_fps=24,
            default_seconds=4,
        )

        with patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.get_fs",
            return_value=MagicMock(),
        ):
            handler = ImageGenerationHandler(
                engine=mock_engine,
                config=config,
            )

        return handler, tracker

    def _make_request(self):
        """Create a minimal valid image generation request dict."""
        return {
            "prompt": "a test image",
            "model": "test-model",
        }

    async def _drain_generator(self, handler, request):
        """Run handler.generate() and drain the async generator."""
        async for _ in handler.generate(request, MagicMock()):
            pass

    @pytest.mark.timeout(5)
    def test_concurrent_requests_are_serialized(self):
        """Fires 3 concurrent requests and asserts only one thread enters
        engine.generate() at a time (max_concurrent == 1).

        If the asyncio.Lock in ImageGenerationHandler is removed, the 3
        asyncio.to_thread() calls run in parallel OS threads, overlapping
        inside the tracker's sleep window, and max_concurrent rises to 3.
        """

        async def run():
            handler, tracker = self._make_handler()

            requests = [self._make_request() for _ in range(3)]

            with patch(
                "dynamo.trtllm.request_handlers.diffusion.image_handler.encode_to_png_bytes",
                return_value=b"fake_image_bytes",
            ), patch(
                "dynamo.trtllm.request_handlers.diffusion.image_handler.upload_to_fs",
                return_value="http://fake/image.png",
            ):
                await asyncio.gather(
                    *(self._drain_generator(handler, req) for req in requests)
                )

            return tracker

        tracker = asyncio.run(run())

        assert tracker.max_concurrent == 1, (
            f"Expected max_concurrent=1 (serialized), got {tracker.max_concurrent}. "
            "Pipeline was accessed concurrently — this would corrupt visual_gen state."
        )


# =============================================================================
# Part 6: ImageGenerationHandler Response Format Tests
# =============================================================================


class TestImageHandlerResponseFormats:
    """Tests for ImageGenerationHandler generate() response format branching."""

    def _make_handler(self):
        """Create a handler with mocked engine and fs."""
        from dynamo.trtllm.request_handlers.diffusion.image_handler import (
            ImageGenerationHandler,
        )

        mock_output = SimpleNamespace(
            video=None,
            image=torch.zeros((1, 64, 64, 3), dtype=torch.uint8),
            audio=None,
        )
        mock_engine = MagicMock()
        mock_engine.generate = MagicMock(return_value=mock_output)

        config = DiffusionConfig(
            media_output_fs_url="file:///tmp/test_media",
            media_output_http_url="https://cdn.example.com/media",
            default_fps=24,
            default_seconds=4,
        )

        with patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.get_fs",
            return_value=MagicMock(),
        ):
            handler = ImageGenerationHandler(
                engine=mock_engine,
                config=config,
            )

        return handler

    @pytest.mark.asyncio
    async def test_url_response_format(self):
        """Test generate() with url response format calls upload_to_fs."""
        handler = self._make_handler()

        request = {
            "prompt": "a test image",
            "model": "test-model",
            "response_format": "url",
        }

        with patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.encode_to_png_bytes",
            return_value=b"fake_image_bytes",
        ), patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.upload_to_fs",
            return_value="https://cdn.example.com/media/images/test.png",
        ) as mock_upload:
            results = []
            async for result in handler.generate(request, MagicMock()):
                results.append(result)

        assert len(results) == 1
        response = results[0]
        assert response["status"] == "completed"
        assert len(response["data"]) == 1
        assert (
            response["data"][0]["url"]
            == "https://cdn.example.com/media/images/test.png"
        )
        mock_upload.assert_called_once()

    @pytest.mark.asyncio
    async def test_b64_response_format(self):
        """Test generate() with b64_json response format returns base64 encoded image."""
        handler = self._make_handler()

        request = {
            "prompt": "a test image",
            "model": "test-model",
            "response_format": "b64_json",
        }

        with patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.encode_to_png_bytes",
            return_value=b"fake_image_bytes",
        ):
            results = []
            async for result in handler.generate(request, MagicMock()):
                results.append(result)

        assert len(results) == 1
        response = results[0]
        assert response["status"] == "completed"
        assert len(response["data"]) == 1
        assert response["data"][0]["b64_json"] is not None
        assert response["data"][0].get("url") is None

        # Verify valid base64
        import base64

        decoded = base64.b64decode(response["data"][0]["b64_json"])
        assert decoded == b"fake_image_bytes"

    @pytest.mark.asyncio
    async def test_default_response_format_is_url(self):
        """Test that generate() defaults to url response format."""
        handler = self._make_handler()

        request = {
            "prompt": "a test image",
            "model": "test-model",
            # No response_format specified
        }

        with patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.encode_to_png_bytes",
            return_value=b"fake_image_bytes",
        ), patch(
            "dynamo.trtllm.request_handlers.diffusion.image_handler.upload_to_fs",
            return_value="https://cdn.example.com/media/images/test.png",
        ) as mock_upload:
            results = []
            async for result in handler.generate(request, MagicMock()):
                results.append(result)

        assert len(results) == 1
        # Default should be "url" format, so upload_to_fs should be called.
        mock_upload.assert_called_once()
        assert results[0]["data"][0]["url"] is not None

    @pytest.mark.asyncio
    async def test_error_response_on_failure(self):
        """
        Test that generate() raises exception on engine failure. This is different from video generation.
        In video generation where the error is embedded in the response, but in image generation,
        the response doesn't contain the error, so the handler doesn't suppress it and let it propagate.
        """
        handler = self._make_handler()
        handler.engine.generate = MagicMock(side_effect=RuntimeError("GPU OOM"))

        request = {
            "prompt": "a test image",
            "model": "test-model",
        }

        with pytest.raises(RuntimeError) as exc_info:
            async for _ in handler.generate(request, MagicMock()):
                pass

        assert "GPU OOM" in str(exc_info.value)
