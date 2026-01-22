# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for diffusion handler utility functions."""

import sys
from unittest.mock import Mock

import pytest

# Mock sglang module before importing handler
sys.modules["sglang"] = Mock()
sys.modules["sglang.multimodal_gen"] = Mock()

from dynamo.sglang.request_handlers.diffusion.diffusion_handler import (
    DiffusionWorkerHandler,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class TestDiffusionHandlerUtils:
    """Test class for DiffusionWorkerHandler utility methods."""

    def test_parse_size(self):
        """Test that size strings are correctly parsed to width and height tuples."""
        # Create a mock handler instance
        component = Mock()
        generator = Mock()
        config = Mock()
        config.dynamo_args = Mock()
        config.dynamo_args.diffusion_s3_bucket = None

        handler = DiffusionWorkerHandler(
            component=component,
            generator=generator,
            config=config,
        )

        # Test standard sizes
        assert handler._parse_size("1024x1024") == (1024, 1024)
        assert handler._parse_size("512x512") == (512, 512)
        assert handler._parse_size("768x512") == (768, 512)
        assert handler._parse_size("1920x1080") == (1920, 1080)

        # Test non-square sizes
        assert handler._parse_size("256x1024") == (256, 1024)
        assert handler._parse_size("1024x256") == (1024, 256)

    def test_encode_base64(self):
        """Test that image bytes are correctly encoded to base64 strings."""
        # Create a mock handler instance
        component = Mock()
        generator = Mock()
        config = Mock()
        config.dynamo_args = Mock()
        config.dynamo_args.diffusion_s3_bucket = None

        handler = DiffusionWorkerHandler(
            component=component,
            generator=generator,
            config=config,
        )

        # Test encoding simple bytes
        test_bytes = b"test image data"
        result = handler._encode_base64(test_bytes)

        # Check result is a string
        assert isinstance(result, str)

        # Check result is valid base64 (can decode it back)
        import base64

        decoded = base64.b64decode(result)
        assert decoded == test_bytes

        # Test with empty bytes
        empty_result = handler._encode_base64(b"")
        assert isinstance(empty_result, str)
        assert base64.b64decode(empty_result) == b""

        # Test with longer binary data (simulating PNG header)
        png_header = b"\x89PNG\r\n\x1a\n"
        png_result = handler._encode_base64(png_header)
        assert isinstance(png_result, str)
        assert base64.b64decode(png_result) == png_header
