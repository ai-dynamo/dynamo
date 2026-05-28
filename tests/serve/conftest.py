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
    """Return a deterministic PNG with an obvious green square."""

    # Lazy import so conftest loads in environments that don't have Pillow (e.g. pre-commit).
    from PIL import Image, ImageDraw

    buf = BytesIO()
    # Keep this synthetic so CI never depends on Git LFS media. The white
    # background plus large centered square gives VLMs a stronger signal than
    # an edge-to-edge flat color.
    img = Image.new("RGB", (512, 512), color="white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((96, 96, 416, 416), fill=(0, 180, 0), outline=(0, 90, 0), width=8)
    draw.text((214, 444), "GREEN", fill=(0, 90, 0))
    img.save(buf, format="PNG")
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
        - /llm-graphic.png - synthetic green-square PNG used by multimodal serve tests

    The handler honors `Range: bytes=A-B` and returns 206 Partial Content.
    The MM-routing dim-fetch path (`fetch_image_dims_uncached`) strictly
    requires 206 on Range probes so it never accidentally downloads a
    full image into memory; a bare `respond_with_data` would return 200
    and silently disable MM routing in the test.

    Usage:
        def test_multimodal(image_server):
            # Use MULTIMODAL_IMG_URL from this module
            # ... use url in your test payload
    """
    from werkzeug.wrappers import Request, Response

    image_data = get_multimodal_test_image_bytes()

    def _handler(request: Request) -> Response:
        range_hdr = request.headers.get("Range", "")
        if range_hdr.startswith("bytes="):
            spec = range_hdr[len("bytes=") :]
            lo_s, _, hi_s = spec.partition("-")
            try:
                lo = int(lo_s) if lo_s else 0
                hi = int(hi_s) if hi_s else len(image_data) - 1
            except ValueError:
                return Response(status=416)
            hi = min(hi, len(image_data) - 1)
            lo = max(lo, 0)
            if lo > hi:
                return Response(status=416)
            chunk = image_data[lo : hi + 1]
            resp = Response(chunk, status=206, content_type="image/png")
            resp.headers["Content-Range"] = f"bytes {lo}-{hi}/{len(image_data)}"
            resp.headers["Accept-Ranges"] = "bytes"
            return resp
        return Response(image_data, status=200, content_type="image/png")

    httpserver.expect_request("/llm-graphic.png").respond_with_handler(_handler)

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
