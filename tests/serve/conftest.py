# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from io import BytesIO

import pytest
from pytest_httpserver import HTTPServer

from tests.serve.lora_utils import MinioLoraConfig, MinioService
from tests.utils.port_utils import allocate_port, deallocate_port

# Shared constants for multimodal testing
IMAGE_SERVER_PORT = allocate_port(8765)
MULTIMODAL_IMG_URL = f"http://localhost:{IMAGE_SERVER_PORT}/llm-graphic.png"


def get_multimodal_test_image_bytes() -> bytes:
    """Return a deterministic green PNG for multimodal serve tests."""

    # Lazy import so conftest loads in environments that don't have Pillow (e.g. pre-commit).
    from PIL import Image

    buf = BytesIO()
    # Keep the served test image synthetic so CI does not depend on Git LFS media
    # contents that prompt models to OCR logo text instead of answering with colors.
    Image.new("RGB", (2, 2), color="green").save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture(scope="session")
def httpserver_listen_address():
    yield ("127.0.0.1", IMAGE_SERVER_PORT)
    deallocate_port(IMAGE_SERVER_PORT)


@pytest.fixture(scope="function")
def image_server(httpserver: HTTPServer):
    """
    Provide an HTTP server that serves test images for multimodal inference.

    This function-scoped fixture configures pytest-httpserver to serve
    a deterministic synthetic image. It's designed for testing multimodal
    inference capabilities where models need to fetch images via HTTP.

    Currently serves:
        - /llm-graphic.png - 2x2 green PNG used by multimodal serve tests

    Usage:
        def test_multimodal(image_server):
            # Use MULTIMODAL_IMG_URL from this module
            # ... use url in your test payload
    """
    image_data = get_multimodal_test_image_bytes()

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
