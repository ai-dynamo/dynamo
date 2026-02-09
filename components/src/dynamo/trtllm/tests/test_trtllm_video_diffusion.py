# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for video diffusion components.

Tests for Modality enum, DiffusionConfig, DiffusionEngine auto-detection,
VideoGenerationHandler helpers, video protocol types, and concurrency safety.

These tests do NOT require visual_gen or GPU - they test logic only.
"""

import asyncio
import json
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from dynamo.trtllm.configs.diffusion_config import DiffusionConfig
from dynamo.trtllm.constants import Modality
from dynamo.trtllm.engines.diffusion_engine import DiffusionEngine
from dynamo.trtllm.protocols.video_protocol import (
    NvCreateVideoRequest,
    NvVideosResponse,
    VideoData,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


# =============================================================================
# Part 1: Modality Enum Tests
# =============================================================================


class TestModality:
    """Tests for the Modality enum and its helper methods."""

    def test_modality_values_exist(self):
        """Test that TEXT, MULTIMODAL, and VIDEO_DIFFUSION exist."""
        assert Modality.TEXT.value == "text"
        assert Modality.MULTIMODAL.value == "multimodal"
        assert Modality.VIDEO_DIFFUSION.value == "video_diffusion"

    def test_is_diffusion_true_for_video_diffusion(self):
        """Test that VIDEO_DIFFUSION returns True for is_diffusion."""
        assert Modality.is_diffusion(Modality.VIDEO_DIFFUSION) is True

    def test_is_diffusion_false_for_text(self):
        """Test that TEXT returns False for is_diffusion."""
        assert Modality.is_diffusion(Modality.TEXT) is False

    def test_is_diffusion_false_for_multimodal(self):
        """Test that MULTIMODAL returns False for is_diffusion."""
        assert Modality.is_diffusion(Modality.MULTIMODAL) is False

    def test_is_llm_true_for_text(self):
        """Test that TEXT returns True for is_llm."""
        assert Modality.is_llm(Modality.TEXT) is True

    def test_is_llm_true_for_multimodal(self):
        """Test that MULTIMODAL returns True for is_llm."""
        assert Modality.is_llm(Modality.MULTIMODAL) is True

    def test_is_llm_false_for_video_diffusion(self):
        """Test that VIDEO_DIFFUSION returns False for is_llm."""
        assert Modality.is_llm(Modality.VIDEO_DIFFUSION) is False


# =============================================================================
# Part 2: DiffusionConfig Tests
# =============================================================================


class TestDiffusionConfig:
    """Tests for DiffusionConfig dataclass."""

    def test_default_values(self):
        """Test that default values are set correctly."""
        config = DiffusionConfig()

        # Dynamo runtime defaults
        assert config.namespace == "dynamo"  # May be overridden by env var
        assert config.component == "diffusion"
        assert config.endpoint == "generate"

        # Generation defaults
        assert config.default_height == 480
        assert config.default_width == 832
        assert config.default_num_frames == 81
        assert config.default_num_inference_steps == 50
        assert config.default_guidance_scale == 5.0

        # Model defaults
        assert config.output_dir == "/tmp/dynamo_videos"

        # Optimization defaults
        assert config.enable_teacache is False
        assert config.attn_type == "default"
        assert config.linear_type == "default"

        # Parallelism defaults
        assert config.dit_dp_size == 1
        assert config.dit_tp_size == 1

    def test_custom_values(self):
        """Test that custom values override defaults."""
        config = DiffusionConfig(
            default_height=720,
            default_width=1280,
            default_num_frames=120,
            enable_teacache=True,
            dit_tp_size=2,
        )

        assert config.default_height == 720
        assert config.default_width == 1280
        assert config.default_num_frames == 120
        assert config.enable_teacache is True
        assert config.dit_tp_size == 2

    def test_str_representation(self):
        """Test that __str__ includes key fields."""
        config = DiffusionConfig(
            model_path="test/model",
            default_height=480,
        )

        str_repr = str(config)

        assert "DiffusionConfig(" in str_repr
        assert "model_path=test/model" in str_repr
        assert "default_height=480" in str_repr
        assert "dit_tp_size=" in str_repr


# =============================================================================
# Part 3: DiffusionEngine Auto-Detection Tests (no visual_gen needed)
# =============================================================================


class TestDetectPipelineInfo:
    """Tests for DiffusionEngine.detect_pipeline_info() auto-detection."""

    def _make_model_dir(self, tmp_path: Path, model_index: dict) -> str:
        """Create a temp model directory with model_index.json."""
        model_dir = tmp_path / "test_model"
        model_dir.mkdir()
        with open(model_dir / "model_index.json", "w") as f:
            json.dump(model_index, f)
        return str(model_dir)

    def test_detect_wan_pipeline_single_transformer(self, tmp_path):
        """Test WanPipeline with single transformer -> ditWanPipeline."""
        model_path = self._make_model_dir(
            tmp_path,
            {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.32.2",
                "scheduler": ["diffusers", "UniPCMultistepScheduler"],
                "text_encoder": ["transformers", "UMT5EncoderModel"],
                "tokenizer": ["transformers", "AutoTokenizer"],
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            },
        )

        info = DiffusionEngine.detect_pipeline_info(model_path)

        assert info.module_path == "visual_gen.pipelines.wan_pipeline"
        assert info.class_name == "ditWanPipeline"
        assert "video_diffusion" in info.modalities
        assert info.config_overrides["torch_compile_models"] == "transformer"

    def test_detect_wan_pipeline_dual_transformer(self, tmp_path):
        """Test WanPipeline with dual transformer -> Wan 2.2 config."""
        model_path = self._make_model_dir(
            tmp_path,
            {
                "_class_name": "WanPipeline",
                "_diffusers_version": "0.32.2",
                "transformer": ["diffusers", "WanTransformer3DModel"],
                "transformer_2": ["diffusers", "WanTransformer3DModel"],
                "vae": ["diffusers", "AutoencoderKLWan"],
            },
        )

        info = DiffusionEngine.detect_pipeline_info(model_path)

        assert info.class_name == "ditWanPipeline"
        # Dual transformer detected from model_index.json keys
        assert (
            info.config_overrides["torch_compile_models"] == "transformer,transformer_2"
        )

    def test_detect_unknown_class_raises_valueerror(self, tmp_path):
        """Test that unknown _class_name raises ValueError with helpful message."""
        model_path = self._make_model_dir(
            tmp_path,
            {
                "_class_name": "SomeUnknownPipeline",
            },
        )

        with pytest.raises(ValueError) as exc_info:
            DiffusionEngine.detect_pipeline_info(model_path)

        error_msg = str(exc_info.value)
        assert "Unsupported diffusion pipeline 'SomeUnknownPipeline'" in error_msg
        assert "Supported pipelines:" in error_msg
        assert "WanPipeline" in error_msg

    def test_detect_missing_model_index_raises_error(self, tmp_path):
        """Test that missing model_index.json for local path raises appropriate error."""
        # Create empty dir with no model_index.json and a non-HF path
        empty_dir = tmp_path / "empty_model"
        empty_dir.mkdir()

        # Local path without model_index.json will try HF Hub download
        # which should fail for a local path that doesn't exist on HF
        with pytest.raises(Exception):
            DiffusionEngine.detect_pipeline_info(str(empty_dir))

    def test_pipeline_registry_has_wan_pipeline(self):
        """Test that WanPipeline entry exists with correct structure."""
        assert "WanPipeline" in DiffusionEngine.PIPELINE_REGISTRY

        entry = DiffusionEngine.PIPELINE_REGISTRY["WanPipeline"]
        assert len(entry) == 3  # (module_path, class_name, modalities)
        module_path, class_name, modalities = entry

        assert "visual_gen" in module_path
        assert class_name == "ditWanPipeline"
        assert "video_diffusion" in modalities


class TestDiffusionEngineDevice:
    """Tests for DiffusionEngine.device property.

    These tests verify device selection aligns with enable_async_cpu_offload config.
    We create engine instances without initializing (no visual_gen import needed).
    """

    def test_device_cuda_by_default(self):
        """Test device is 'cuda' when CPU offload is disabled (default)."""
        config = DiffusionConfig(enable_async_cpu_offload=False)
        engine = object.__new__(DiffusionEngine)
        engine.config = config
        assert engine.device == "cuda"

    def test_device_cpu_when_offload_enabled(self):
        """Test device is 'cpu' when CPU offload is enabled."""
        config = DiffusionConfig(enable_async_cpu_offload=True)
        engine = object.__new__(DiffusionEngine)
        engine.config = config
        assert engine.device == "cpu"


# =============================================================================
# Part 4: VideoGenerationHandler Helper Tests
# =============================================================================


class MockDiffusionConfig:
    """Mock config for testing handler helpers without full DiffusionConfig."""

    default_width: int = 832
    default_height: int = 480
    default_num_frames: int = 81
    default_fps: int = 24
    default_seconds: int = 4
    max_width: int = 4096
    max_height: int = 4096


@dataclass
class MockVideoRequest:
    """Mock video request for testing _compute_num_frames."""

    prompt: str = "test prompt"
    model: str = "test-model"
    num_frames: Optional[int] = None
    seconds: Optional[int] = None
    fps: Optional[int] = None


class TestVideHandlerParseSize:
    """Tests for VideoGenerationHandler._parse_size method.

    We test the method logic by creating a minimal mock handler.
    """

    def setup_method(self):
        """Set up mock handler for each test."""
        # Import here to avoid issues if handler has complex imports
        from dynamo.trtllm.request_handlers.video_diffusion.video_handler import (
            VideoGenerationHandler,
        )

        # Create handler with mocked dependencies
        self.handler = object.__new__(VideoGenerationHandler)
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


class TestVideoHandlerComputeNumFrames:
    """Tests for VideoGenerationHandler._compute_num_frames method."""

    def setup_method(self):
        """Set up mock handler for each test."""
        from dynamo.trtllm.request_handlers.video_diffusion.video_handler import (
            VideoGenerationHandler,
        )

        self.handler = object.__new__(VideoGenerationHandler)
        self.handler.config = MockDiffusionConfig()

    def test_compute_num_frames_explicit(self):
        """Test that explicit num_frames takes priority."""
        req = NvCreateVideoRequest(
            prompt="test",
            model="test-model",
            num_frames=100,
            seconds=10,  # Should be ignored
            fps=30,  # Should be ignored
        )
        assert self.handler._compute_num_frames(req) == 100

    def test_compute_num_frames_from_seconds_fps(self):
        """Test computation from seconds * fps."""
        req = NvCreateVideoRequest(
            prompt="test",
            model="test-model",
            seconds=4,
            fps=24,
        )
        assert self.handler._compute_num_frames(req) == 96  # 4 * 24

    def test_compute_num_frames_only_seconds(self):
        """Test seconds with default fps (24)."""
        req = NvCreateVideoRequest(
            prompt="test",
            model="test-model",
            seconds=5,
        )
        # seconds=5, default fps=24 -> 5 * 24 = 120
        assert self.handler._compute_num_frames(req) == 120

    def test_compute_num_frames_only_fps(self):
        """Test fps with default seconds (4)."""
        req = NvCreateVideoRequest(
            prompt="test",
            model="test-model",
            fps=30,
        )
        # default seconds=4, fps=30 -> 4 * 30 = 120
        assert self.handler._compute_num_frames(req) == 120

    def test_compute_num_frames_defaults(self):
        """Test all None uses config default."""
        req = NvCreateVideoRequest(
            prompt="test",
            model="test-model",
        )
        assert (
            self.handler._compute_num_frames(req)
            == MockDiffusionConfig.default_num_frames
        )


# =============================================================================
# Part 5: Video Protocol Tests
# =============================================================================


class TestNvCreateVideoRequest:
    """Tests for NvCreateVideoRequest protocol type."""

    def test_required_fields(self):
        """Test that prompt and model are required."""
        req = NvCreateVideoRequest(prompt="A cat", model="wan_t2v")
        assert req.prompt == "A cat"
        assert req.model == "wan_t2v"

    def test_required_fields_missing_prompt(self):
        """Test that missing prompt raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            NvCreateVideoRequest(model="wan_t2v")  # type: ignore

    def test_required_fields_missing_model(self):
        """Test that missing model raises validation error."""
        with pytest.raises(Exception):  # Pydantic ValidationError
            NvCreateVideoRequest(prompt="A cat")  # type: ignore

    def test_optional_fields_default_none(self):
        """Test that optional fields default to None."""
        req = NvCreateVideoRequest(prompt="A cat", model="wan_t2v")

        assert req.size is None
        assert req.seconds is None
        assert req.fps is None
        assert req.num_frames is None
        assert req.num_inference_steps is None
        assert req.guidance_scale is None
        assert req.negative_prompt is None
        assert req.seed is None
        assert req.response_format is None

    def test_full_request_valid(self):
        """Test a fully populated request."""
        req = NvCreateVideoRequest(
            prompt="A majestic lion",
            model="wan_t2v",
            size="1920x1080",
            seconds=5,
            fps=30,
            num_frames=150,
            num_inference_steps=30,
            guidance_scale=7.5,
            negative_prompt="blurry, low quality",
            seed=42,
            response_format="b64_json",
        )

        assert req.prompt == "A majestic lion"
        assert req.model == "wan_t2v"
        assert req.size == "1920x1080"
        assert req.seconds == 5
        assert req.fps == 30
        assert req.num_frames == 150
        assert req.num_inference_steps == 30
        assert req.guidance_scale == 7.5
        assert req.negative_prompt == "blurry, low quality"
        assert req.seed == 42
        assert req.response_format == "b64_json"


class TestVideoData:
    """Tests for VideoData protocol type."""

    def test_url_only(self):
        """Test VideoData with URL only."""
        data = VideoData(url="/tmp/video.mp4")
        assert data.url == "/tmp/video.mp4"
        assert data.b64_json is None

    def test_b64_only(self):
        """Test VideoData with base64 only."""
        data = VideoData(b64_json="SGVsbG8gV29ybGQ=")
        assert data.url is None
        assert data.b64_json == "SGVsbG8gV29ybGQ="

    def test_both_fields(self):
        """Test VideoData with both fields (unusual but valid)."""
        data = VideoData(url="/tmp/video.mp4", b64_json="SGVsbG8=")
        assert data.url == "/tmp/video.mp4"
        assert data.b64_json == "SGVsbG8="

    def test_empty_defaults(self):
        """Test VideoData with no arguments."""
        data = VideoData()
        assert data.url is None
        assert data.b64_json is None


class TestNvVideosResponse:
    """Tests for NvVideosResponse protocol type."""

    def test_default_values(self):
        """Test default values for completed response."""
        response = NvVideosResponse(
            id="req-123",
            model="wan_t2v",
            created=1234567890,
        )

        assert response.id == "req-123"
        assert response.object == "video"
        assert response.model == "wan_t2v"
        assert response.status == "completed"
        assert response.progress == 100
        assert response.created == 1234567890
        assert response.data == []
        assert response.error is None

    def test_error_response(self):
        """Test error response structure."""
        response = NvVideosResponse(
            id="req-456",
            model="wan_t2v",
            created=1234567890,
            status="failed",
            progress=0,
            error="Model failed to load",
        )

        assert response.status == "failed"
        assert response.progress == 0
        assert response.error == "Model failed to load"

    def test_with_video_data(self):
        """Test response with video data."""
        video = VideoData(url="/tmp/output.mp4")
        response = NvVideosResponse(
            id="req-789",
            model="wan_t2v",
            created=1234567890,
            data=[video],
            inference_time_s=42.5,
        )

        assert len(response.data) == 1
        assert response.data[0].url == "/tmp/output.mp4"
        assert response.inference_time_s == 42.5

    def test_model_dump(self):
        """Test serialization with model_dump()."""
        response = NvVideosResponse(
            id="req-123",
            model="wan_t2v",
            created=1234567890,
            data=[VideoData(url="/tmp/video.mp4")],
        )

        dumped = response.model_dump()

        assert isinstance(dumped, dict)
        assert dumped["id"] == "req-123"
        assert dumped["object"] == "video"
        assert dumped["model"] == "wan_t2v"
        assert dumped["status"] == "completed"
        assert len(dumped["data"]) == 1
        assert dumped["data"][0]["url"] == "/tmp/video.mp4"


# =============================================================================
# Part 6: Concurrency Safety Tests
# =============================================================================


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

        # Return fake frames (shape: [num_frames, H, W, C])
        import numpy as np

        return np.zeros((4, 64, 64, 3), dtype=np.uint8)


class TestVideoHandlerConcurrency:
    """Verifies that ``VideoGenerationHandler`` serializes access to the
    underlying ``engine.generate()`` call.

    Why this matters:
        The visual_gen pipeline is a global singleton with mutable state,
        unprotected CUDA graph caches, and shared config objects.  It is
        NOT thread-safe.  ``VideoGenerationHandler`` dispatches generate()
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
        """Create a VideoGenerationHandler with mock engine and config."""
        from dynamo.trtllm.request_handlers.video_diffusion.video_handler import (
            VideoGenerationHandler,
        )

        tracker = ConcurrencyTracker(sleep_seconds=0.1)

        mock_engine = MagicMock()
        mock_engine.generate = tracker.generate

        config = DiffusionConfig(
            output_dir="/tmp/test_videos",
            default_fps=24,
            default_seconds=4,
        )

        handler = VideoGenerationHandler(
            component=MagicMock(),
            engine=mock_engine,
            config=config,
        )

        return handler, tracker

    def _make_request(self):
        """Create a minimal valid video generation request dict."""
        return {
            "prompt": "a test video",
            "model": "test-model",
        }

    async def _drain_generator(self, handler, request):
        """Run handler.generate() and drain the async generator."""
        async for _ in handler.generate(request, MagicMock()):
            pass

    def test_concurrent_requests_are_serialized(self):
        """Fires 3 concurrent requests and asserts only one thread enters
        engine.generate() at a time (max_concurrent == 1).

        If the asyncio.Lock in VideoGenerationHandler is removed, the 3
        asyncio.to_thread() calls run in parallel OS threads, overlapping
        inside the tracker's sleep window, and max_concurrent rises to 3.
        """

        async def run():
            handler, tracker = self._make_handler()

            requests = [self._make_request() for _ in range(3)]

            with patch(
                "dynamo.trtllm.request_handlers.video_diffusion.video_handler.encode_to_mp4",
                return_value="/tmp/test.mp4",
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
