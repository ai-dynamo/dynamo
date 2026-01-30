# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for ImageDiffusionWorkerHandler."""

import base64
import io
import os
from unittest.mock import MagicMock, Mock, patch

import pytest
from PIL import Image

from dynamo.sglang.request_handlers.image_diffusion.image_diffusion_handler import (
    ImageDiffusionWorkerHandler,
    ImageStoragePathResolver,
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
    config.dynamo_args.image_diffusion_url_base = None
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
def handler(mock_component, mock_generator, mock_config, mock_fs):
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
        assert handler.url_base is None

    def test_initialization_with_url_base(
        self, mock_component, mock_generator, mock_fs
    ):
        """Test handler initialization with URL base."""
        config = MagicMock()
        config.dynamo_args = MagicMock()
        config.dynamo_args.image_diffusion_fs_url = "s3://my-bucket/images"
        config.dynamo_args.image_diffusion_url_base = "http://localhost:8008/images"

        handler = ImageDiffusionWorkerHandler(
            component=mock_component,
            generator=mock_generator,
            config=config,
            publisher=None,
            fs=mock_fs,
        )

        assert handler.url_base == "http://localhost:8008/images"
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
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "seed": 42,
            "negative_prompt": None,
            "user": "test-user",
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
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "seed": 42,
            "negative_prompt": None,
            "user": "test-user",
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
            "guidance_scale": 7.5,
            "seed": 42,
            "negative_prompt": None,
            "user": "test-user",
        }

        # Execute generation
        results = []
        async for result in handler.generate(request, mock_context):
            results.append(result)

        # Verify default was applied
        assert "num_inference_steps" in request
        assert request["num_inference_steps"] == 50

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
            "num_inference_steps": 10,
            "guidance_scale": 7.5,
            "seed": 42,
            "negative_prompt": None,
            "user": "test-user",
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


class TestImageStoragePathResolver:
    """Test suite for ImageStoragePathResolver."""

    def test_file_protocol_initialization(self):
        """Test initialization with file:// protocol."""
        resolver = ImageStoragePathResolver("file:///tmp/images")
        assert resolver.fs_base_url == "file:///tmp/images"
        assert resolver.fs_root == "/tmp/images"

    def test_s3_protocol_initialization(self):
        """Test initialization with s3:// protocol."""
        resolver = ImageStoragePathResolver("s3://my-bucket/images")
        assert resolver.fs_base_url == "s3://my-bucket/images"
        assert resolver.fs_root is None

    def test_get_path_with_file_protocol(self):
        """Test get_path with file:// protocol."""
        resolver = ImageStoragePathResolver("file:///tmp/images")
        storage_path = "users/user123/image.png"
        result = resolver.get_path(storage_path)
        assert result == "/tmp/images/users/user123/image.png"

    def test_get_path_with_s3_protocol(self):
        """Test get_path with S3 protocol returns storage path as-is."""
        resolver = ImageStoragePathResolver("s3://my-bucket/images")
        storage_path = "users/user123/image.png"
        result = resolver.get_path(storage_path)
        assert result == storage_path

    def test_get_fs_url_with_s3(self):
        """Test get_fs_url generates correct S3 URL."""
        resolver = ImageStoragePathResolver("s3://my-bucket/path")
        storage_path = "users/user123/image.png"
        result = resolver.get_fs_url(storage_path)
        assert (
            result == "https://my-bucket.s3.amazonaws.com/path/users/user123/image.png"
        )

    def test_get_fs_url_with_s3_and_region(self):
        """Test get_fs_url with AWS region."""
        with patch.dict(os.environ, {"AWS_REGION": "us-west-2"}):
            resolver = ImageStoragePathResolver("s3://my-bucket/path")
            storage_path = "users/user123/image.png"
            result = resolver.get_fs_url(storage_path)
            assert (
                result
                == "https://my-bucket.s3.us-west-2.amazonaws.com/path/users/user123/image.png"
            )

    def test_get_fs_url_with_gcs(self):
        """Test get_fs_url generates correct GCS URL."""
        resolver = ImageStoragePathResolver("gs://my-bucket/path")
        storage_path = "users/user123/image.png"
        result = resolver.get_fs_url(storage_path)
        assert (
            result
            == "https://storage.googleapis.com/my-bucket/path/users/user123/image.png"
        )

    def test_get_fs_url_with_azure(self):
        """Test get_fs_url generates correct Azure URL."""
        resolver = ImageStoragePathResolver("az://container@account/path")
        storage_path = "users/user123/image.png"
        result = resolver.get_fs_url(storage_path)
        assert (
            result
            == "https://account.blob.core.windows.net/container/path/users/user123/image.png"
        )

    def test_get_fs_url_with_file_protocol(self):
        """Test get_fs_url with file:// protocol."""
        resolver = ImageStoragePathResolver("file:///tmp/images")
        storage_path = "users/user123/image.png"
        result = resolver.get_fs_url(storage_path)
        assert result == "file:///tmp/images/users/user123/image.png"

    def test_get_fs_url_unknown_protocol(self):
        """Test get_fs_url raises error for unknown protocol."""
        resolver = ImageStoragePathResolver("unknown://something")
        storage_path = "users/user123/image.png"
        with pytest.raises(ValueError, match="Unknown filesystem type"):
            resolver.get_fs_url(storage_path)

    def test_get_url_with_url_base(self):
        """Test get_url uses url_base when configured."""
        resolver = ImageStoragePathResolver(
            "file:///tmp/images", url_base="http://localhost:8008/images"
        )
        storage_path = "users/user123/image.png"
        result = resolver.get_url(storage_path)
        assert result == "http://localhost:8008/images/users/user123/image.png"

    def test_get_url_without_url_base(self):
        """Test get_url falls back to filesystem URL when url_base not set."""
        resolver = ImageStoragePathResolver("s3://my-bucket/images")
        storage_path = "users/user123/image.png"
        result = resolver.get_url(storage_path)
        # Should return S3 URL
        assert result.startswith("https://my-bucket.s3.amazonaws.com/")

    def test_get_url_with_trailing_slash_in_base(self):
        """Test get_url handles trailing slashes correctly."""
        resolver = ImageStoragePathResolver(
            "file:///tmp/images", url_base="http://localhost:8008/images/"
        )
        storage_path = "users/user123/image.png"
        result = resolver.get_url(storage_path)
        # Should not have double slash
        assert result == "http://localhost:8008/images/users/user123/image.png"
        assert "//" not in result.replace("http://", "")

    def test_get_url_with_s3_protocol_and_url_base(self):
        """Test get_url with s3 protocol and url base redirecting to other service."""
        resolver = ImageStoragePathResolver(
            "s3://my-bucket/images", url_base="http://localhost:8008/images/"
        )
        storage_path = "users/user123/image.png"
        result = resolver.get_url(storage_path)
        # Should not have double slash
        assert result == "http://localhost:8008/images/users/user123/image.png"

    def test_get_url_with_leading_slash_in_path(self):
        """Test get_url handles leading slashes in storage path."""
        resolver = ImageStoragePathResolver(
            "file:///tmp/images", url_base="http://localhost:8008/images"
        )
        storage_path = "/users/user123/image.png"
        result = resolver.get_url(storage_path)
        assert result == "http://localhost:8008/images/users/user123/image.png"
