# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ImageDiffusionWorkerHandler."""

import base64
import io
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from dynamo.sglang.request_handlers.image_diffusion.image_diffusion_handler import (
    ImageDiffusionWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,  # No GPU needed for unit tests
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


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
    config.dynamo_args.image_diffusion_fs_url = "file:///tmp/images"
    config.dynamo_args.image_diffusion_base_url = "file:///tmp/images"
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


@pytest.fixture
def handler(
    mock_component, mock_generator, mock_config, mock_fs
) -> ImageDiffusionWorkerHandler:
    """Create ImageDiffusionWorkerHandler instance."""
    return ImageDiffusionWorkerHandler(
        component=mock_component,
        generator=mock_generator,
        config=mock_config,
        publisher=None,
        fs=mock_fs,
    )


class TestImageDiffusionWorkerHandler:
    """Test suite for ImageDiffusionWorkerHandler."""

    def test_initialization(self, handler, mock_generator, mock_fs):
        """Test handler initialization."""
        assert handler.generator == mock_generator
        assert handler.fs == mock_fs
        assert handler.fs_url == "file:///tmp/images"
        assert handler.base_url == "file:///tmp/images"

    def test_initialization_with_url_base(
        self, mock_component, mock_generator, mock_fs
    ):
        """Test handler initialization with URL base."""
        config = MagicMock()
        config.dynamo_args = MagicMock()
        config.dynamo_args.image_diffusion_fs_url = "s3://my-bucket/images"
        config.dynamo_args.image_diffusion_base_url = "http://localhost:8008/images"

        handler = ImageDiffusionWorkerHandler(
            component=mock_component,
            generator=mock_generator,
            config=config,
            publisher=None,
            fs=mock_fs,
        )

        assert handler.base_url == "http://localhost:8008/images"
        assert handler.fs_url == "s3://my-bucket/images"

    @patch("torch.cuda.empty_cache")
    def test_cleanup(self, mock_empty_cache, handler):
        """Test cleanup method."""
        _original_generator = handler.generator
        handler.cleanup()
        # Generator should be set to None after cleanup
        # Note: We can't assert it's None because the attribute gets deleted
        mock_empty_cache.assert_called_once()

    def test_parse_size(self, handler):
        """Test _parse_size method."""
        width, height = handler._parse_size("1024x1024")
        assert width == 1024
        assert height == 1024

        width, height = handler._parse_size("512x768")
        assert width == 512
        assert height == 768

    def test_encode_base64(self, handler):
        """Test _encode_base64 method."""
        test_bytes = b"test image data"
        expected = base64.b64encode(test_bytes).decode("utf-8")
        result = handler._encode_base64(test_bytes)
        assert result == expected

    @pytest.mark.asyncio
    async def test_generate_success_url_format(self, handler, mock_context):
        """Test successful image generation with URL response format."""
        # Create a simple test image
        test_image = Image.new("RGB", (256, 256), color="red")
        img_buffer: io.BytesIO = io.BytesIO()
        test_image.save(img_buffer, format="PNG")

        # Mock generator response
        handler.generator.generate = Mock(
            return_value={"frames": [test_image.convert("RGB")]}
        )

        request = {
            "prompt": "A red square",
            "model": "test-model",
            "size": "256x256",
            "response_format": "url",
            "user": "test-user",
            "nvext": {
                "num_inference_steps": 10,
                "guidance_scale": 7.5,
                "seed": 42,
                "negative_prompt": None,
            },
        }

        # Execute generation
        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        # Verify results
        assert len(results) == 1
        response = results[0]
        assert "created" in response
        assert "data" in response
        assert len(response["data"]) == 1
        assert "url" in response["data"][0]
        assert response["data"][0]["url"].startswith("file:///tmp/images/users/")

    @pytest.mark.asyncio
    async def test_generate_success_b64_format(self, handler, mock_context):
        """Test successful image generation with base64 response format."""
        # Create a simple test image
        test_image = Image.new("RGB", (256, 256), color="blue")

        # Mock generator response
        handler.generator.generate = Mock(
            return_value={"frames": [test_image.convert("RGB")]}
        )

        request = {
            "prompt": "A blue square",
            "model": "test-model",
            "size": "256x256",
            "response_format": "b64_json",
            "user": "test-user",
            "nvext": {
                "num_inference_steps": 10,
                "guidance_scale": 7.5,
                "seed": 42,
                "negative_prompt": None,
            },
        }

        # Execute generation
        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        # Verify results
        assert len(results) == 1
        response = results[0]
        assert "created" in response
        assert "data" in response
        assert len(response["data"]) == 1
        assert "b64_json" in response["data"][0]
        # Verify it's valid base64
        b64_data = response["data"][0]["b64_json"]
        decoded = base64.b64decode(b64_data)
        assert len(decoded) > 0

    @pytest.mark.asyncio
    async def test_generate_with_default_num_inference_steps(
        self, handler, mock_context
    ):
        """Test that num_inference_steps defaults to 50."""
        test_image = Image.new("RGB", (256, 256), color="green")
        handler.generator.generate = Mock(return_value={"frames": [test_image]})

        request = {
            "prompt": "A green square",
            "model": "test-model",
            "size": "256x256",
            "response_format": "b64_json",
            "user": "test-user",
        }

        # Execute generation
        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

    @pytest.mark.asyncio
    async def test_generate_error_handling(self, handler, mock_context):
        """Test error handling in generate method."""
        # Mock generator to raise an exception
        handler.generator.generate = Mock(side_effect=RuntimeError("Generation failed"))

        request = {
            "prompt": "Test prompt",
            "model": "test-model",
            "size": "256x256",
            "response_format": "url",
            "user": "test-user",
            "nvext": {
                "num_inference_steps": 10,
                "guidance_scale": 7.5,
                "seed": 42,
                "negative_prompt": None,
            },
        }

        # Execute generation
        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        # Verify error response
        assert len(results) == 1
        response = results[0]
        assert "error" in response
        assert "Generation failed" in response["error"]
        assert response["data"] == []

    @pytest.mark.asyncio
    async def test_upload_to_fs(self, handler):
        """Test _upload_to_fs method."""
        image_bytes = b"test image data"
        user_id = "user123"
        request_id = "req456"

        url = await handler._upload_to_fs(image_bytes, user_id, request_id)

        # Verify storage path format
        assert f"users/{user_id}/generations/{request_id}/" in url
        assert url.endswith(".png")

    @pytest.mark.asyncio
    async def test_generate_images_with_numpy_array(self, handler):
        """Test _generate_images handles numpy arrays."""
        import numpy as np

        # Create a numpy array representing an image
        np_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)

        handler.generator.generate = Mock(return_value={"frames": [np_image]})

        images = await handler._generate_images(
            prompt="test",
            width=256,
            height=256,
            num_inference_steps=10,
            guidance_scale=7.5,
            seed=42,
        )

        assert len(images) == 1
        assert isinstance(images[0], bytes)

    @pytest.mark.asyncio
    async def test_generate_images_with_pil_image(self, handler):
        """Test _generate_images handles PIL Images."""
        pil_image = Image.new("RGB", (256, 256), color="red")

        handler.generator.generate = Mock(return_value={"frames": [pil_image]})

        images = await handler._generate_images(
            prompt="test",
            width=256,
            height=256,
            num_inference_steps=10,
            guidance_scale=7.5,
            seed=42,
        )

        assert len(images) == 1
        assert isinstance(images[0], bytes)

    @pytest.mark.asyncio
    async def test_generate_images_with_bytes(self, handler):
        """Test _generate_images handles bytes directly."""
        img_bytes = b"raw image bytes"

        handler.generator.generate = Mock(return_value={"frames": [img_bytes]})

        images = await handler._generate_images(
            prompt="test",
            width=256,
            height=256,
            num_inference_steps=10,
            guidance_scale=7.5,
            seed=42,
        )

        assert len(images) == 1
        assert images[0] == img_bytes

    @pytest.mark.asyncio
    async def test_generate_with_nvext(self, handler, mock_context):
        """Test that nvext parameters are passed to the generator."""
        test_image = Image.new("RGB", (256, 256), color="yellow")

        handler._generate_images = Mock(return_value=[test_image.tobytes()])
        handler._get_trace_header = Mock(
            return_value={"traceparent": "00-1234567890-1234567890-01"}
        )

        request = {
            "prompt": "A yellow square",
            "model": "test-model",
            "size": "256x256",
            "response_format": "b64_json",
            "user": "test-user",
            "nvext": {
                "num_inference_steps": 10,
                "guidance_scale": 7.5,
                "seed": 42,
                "negative_prompt": "negative",
            },
        }

        # Execute generation
        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        # Verify results
        handler._generate_images.assert_called_once_with(
            prompt="A yellow square",
            width=256,
            height=256,
            num_inference_steps=10,
            guidance_scale=7.5,
            seed=42,
            negative_prompt="negative",
        )
