# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest
from pytest_httpserver import HTTPServer

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.serve.lora_utils import MinioLoraConfig, MinioService

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


@pytest.fixture(scope="function")
def minio_lora_service():
    """
    Provide a MinIO service with a pre-uploaded LoRA adapter for testing.

    This fixture:
    1. Connects to existing MinIO or starts a Docker container
    2. Creates the required S3 bucket
    3. Downloads the LoRA adapter from Hugging Face Hub
    4. Uploads it to MinIO
    5. Yields the MinioLoraConfig with connection details
    6. Cleans up after the test (only stops container if we started it)

    Usage:
        def test_lora(minio_lora_service):
            config = minio_lora_service
            # Use config.get_env_vars() for environment setup
            # Use config.get_s3_uri() to get the S3 URI for loading LoRA
    """
    config = MinioLoraConfig()
    service = MinioService(config)

    try:
        # Start or connect to MinIO
        service.start()

        # Create bucket and upload LoRA
        service.create_bucket()
        local_path = service.download_lora()
        service.upload_lora(local_path)

        # Clean up downloaded files (keep MinIO data intact)
        service.cleanup_download()

        yield config

    finally:
        # Stop MinIO only if we started it, clean up temp dirs
        service.stop()
        service.cleanup_temp()
