# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import os

import pytest
from pytest_httpserver import HTTPServer

from dynamo.common.utils.paths import WORKSPACE_DIR

# Shared constants for multimodal testing
IMAGE_SERVER_PORT = 8765
MULTIMODAL_IMG_PATH = os.path.join(
    WORKSPACE_DIR, "lib/llm/tests/data/media/llm-optimize-deploy-graphic.png"
)
MULTIMODAL_IMG_URL = f"http://localhost:{IMAGE_SERVER_PORT}/llm-graphic.png"


@pytest.fixture(scope="session")
def httpserver_listen_address():
    return ("127.0.0.1", IMAGE_SERVER_PORT)


@pytest.fixture(scope="function")
def image_server(httpserver: HTTPServer):
    """
    Provide an HTTP server that serves test images for multimodal inference.

    This function-scoped fixture configures pytest-httpserver to serve
    the LLM optimization diagram image. It's designed for testing multimodal
    inference capabilities where models need to fetch images via HTTP.

    Currently serves:
        - /llm-graphic.png - LLM diagram image for multimodal tests

    Usage:
        def test_multimodal(image_server):
            url = "http://localhost:8765/llm-graphic.png"
            # ... use url in your test payload
    """
    # Load LLM graphic image from shared test data
    with open(MULTIMODAL_IMG_PATH, "rb") as f:
        image_data = f.read()

    # Configure server endpoint
    httpserver.expect_request("/llm-graphic.png").respond_with_data(
        image_data,
        content_type="image/png",
    )

    return httpserver
