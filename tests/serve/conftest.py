# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared pytest fixtures for serve tests.

This module provides fixtures used across multiple serve test modules
(test_vllm.py, test_sglang.py, test_trtllm.py) for testing inference serving.
"""

import os

import pytest
from pytest_httpserver import HTTPServer

# Shared constants for multimodal testing
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
IMAGE_SERVER_PORT = 8765


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("127.0.0.1", IMAGE_SERVER_PORT)


@pytest.fixture(scope="function")
def image_server(httpserver: HTTPServer):
    """
    Provide an HTTP server that serves test fixture images.

    This function-scoped fixture configures pytest-httpserver to serve
    images from the fixtures/ directory. It's designed for testing multimodal
    inference capabilities where models need to fetch images via HTTP.

    Currently serves:
        - /basketball.png - Basketball image for multimodal tests

    Usage:
        def test_multimodal(image_server):
            url = "http://localhost:8765/basketball.png"
            # ... use url in your test payload
    """
    # Load basketball image
    basketball_path = os.path.join(FIXTURES_DIR, "basketball.png")
    with open(basketball_path, "rb") as f:
        basketball_data = f.read()

    # Configure server endpoint
    httpserver.expect_request("/basketball.png").respond_with_data(
        basketball_data,
        content_type="image/png",
    )

    return httpserver
