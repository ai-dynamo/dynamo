# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for multimodal data extraction in standard vLLM workers.

Tests the _extract_multimodal_data() method added to BaseWorkerHandler
to support base64 and HTTP image URLs in PreprocessedRequest.
"""

from unittest.mock import MagicMock, patch

import pytest
from PIL import Image


@pytest.fixture
def mock_handler():
    """Create a mock handler with ImageLoader for testing"""
    from dynamo.vllm.handlers import BaseWorkerHandler

    # Create a concrete implementation of the abstract class for testing
    class TestWorkerHandler(BaseWorkerHandler):
        async def generate(self, request, context):
            pass  # Not needed for these tests

    runtime = MagicMock()
    component = MagicMock()
    engine = MagicMock()
    default_sampling_params = {}

    handler = TestWorkerHandler(runtime, component, engine, default_sampling_params)
    return handler


@pytest.mark.asyncio
async def test_extract_no_multimodal_data(mock_handler):
    """Test that None is returned when no multimodal data present"""
    request = {"token_ids": [1, 2, 3]}

    result = await mock_handler._extract_multimodal_data(request)

    assert result is None


@pytest.mark.asyncio
async def test_extract_empty_multimodal_data(mock_handler):
    """Test that None is returned when multimodal data is empty"""
    request = {"token_ids": [1, 2, 3], "multi_modal_data": None}

    result = await mock_handler._extract_multimodal_data(request)

    assert result is None


@pytest.mark.asyncio
async def test_extract_base64_image(mock_handler):
    """Test extraction of base64 data URL image"""
    # Minimal 1x1 PNG as base64
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{tiny_png_b64}"

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Url": data_url}]},
    }

    result = await mock_handler._extract_multimodal_data(request)

    assert result is not None
    assert "image" in result
    assert isinstance(result["image"], Image.Image)
    # Verify it's a 1x1 image
    assert result["image"].size == (1, 1)


@pytest.mark.asyncio
async def test_extract_http_url(mock_handler):
    """Test extraction of HTTP URL image (mocked)"""
    # Mock the ImageLoader to avoid actual HTTP call
    mock_image = Image.new("RGB", (100, 100))

    with patch.object(mock_handler.image_loader, "load_image", return_value=mock_image):
        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {
                "image_url": [{"Url": "https://example.com/image.jpg"}]
            },
        }

        result = await mock_handler._extract_multimodal_data(request)

        assert result is not None
        assert "image" in result
        assert result["image"] == mock_image


@pytest.mark.asyncio
async def test_extract_multiple_images(mock_handler):
    """Test extraction of multiple images returns list"""
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    data_url_1 = f"data:image/png;base64,{tiny_png_b64}"
    data_url_2 = f"data:image/png;base64,{tiny_png_b64}"

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Url": data_url_1}, {"Url": data_url_2}]},
    }

    result = await mock_handler._extract_multimodal_data(request)

    assert result is not None
    assert "image" in result
    # Multiple images should return as list
    assert isinstance(result["image"], list)
    assert len(result["image"]) == 2
    assert all(isinstance(img, Image.Image) for img in result["image"])


@pytest.mark.asyncio
async def test_extract_single_image_not_list(mock_handler):
    """Test that single image is returned as Image, not list"""
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{tiny_png_b64}"

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Url": data_url}]},
    }

    result = await mock_handler._extract_multimodal_data(request)

    assert result is not None
    assert "image" in result
    # Single image should NOT be a list
    assert isinstance(result["image"], Image.Image)
    assert not isinstance(result["image"], list)


@pytest.mark.asyncio
async def test_extract_invalid_base64(mock_handler):
    """Test that invalid base64 raises appropriate error"""
    data_url = "data:image/png;base64,INVALID_BASE64!!!"

    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"image_url": [{"Url": data_url}]},
    }

    with pytest.raises(Exception):  # ImageLoader will raise ValueError
        await mock_handler._extract_multimodal_data(request)


@pytest.mark.asyncio
async def test_extract_decoded_format_future(mock_handler):
    """Test that Decoded format is handled gracefully (future support)"""
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {
            "image_url": [{"Decoded": {"nixl_metadata": "...", "shape": [224, 224, 3]}}]
        },
    }

    # Should not crash, just log warning and return None
    result = await mock_handler._extract_multimodal_data(request)

    # Since Decoded is not yet supported, no images extracted
    assert result is None or result == {}


@pytest.mark.asyncio
async def test_extract_mixed_url_formats(mock_handler):
    """Test extraction with mix of data: and http: URLs"""
    tiny_png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
    data_url = f"data:image/png;base64,{tiny_png_b64}"
    mock_image = Image.new("RGB", (50, 50))

    with patch.object(
        mock_handler.image_loader, "load_image", side_effect=[None, mock_image]
    ) as mock_load:
        # First call (data URL) will use real ImageLoader
        # Second call (http URL) will use mock
        mock_load.side_effect = [
            await mock_handler.image_loader.load_image(data_url),
            mock_image,
        ]

        request = {
            "token_ids": [1, 2, 3],
            "multi_modal_data": {
                "image_url": [
                    {"Url": data_url},
                    {"Url": "https://example.com/image.jpg"},
                ]
            },
        }

        result = await mock_handler._extract_multimodal_data(request)

        assert result is not None
        assert "image" in result
        assert isinstance(result["image"], list)
        assert len(result["image"]) == 2


@pytest.mark.asyncio
async def test_extract_video_url_warning(mock_handler):
    """Test that video URLs log warning (not yet supported)"""
    request = {
        "token_ids": [1, 2, 3],
        "multi_modal_data": {"video_url": [{"Url": "https://example.com/video.mp4"}]},
    }

    with patch("dynamo.vllm.handlers.logger") as mock_logger:
        result = await mock_handler._extract_multimodal_data(request)

        # Should log warning about video not supported
        mock_logger.warning.assert_called()
        # Should return None or empty dict
        assert result is None or result == {}
