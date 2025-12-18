# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from dataclasses import dataclass
from typing import Generator

import pytest
from pytest_httpserver import HTTPServer

from dynamo.common.utils.paths import WORKSPACE_DIR
from tests.serve.lora_utils import MinioLoraConfig, MinioService
from tests.utils.constants import DefaultPort
from tests.utils.port_utils import allocate_port, allocate_ports, deallocate_ports

# Shared constants for multimodal testing
IMAGE_SERVER_PORT = 8765
MULTIMODAL_IMG_PATH = os.path.join(
    WORKSPACE_DIR, "lib/llm/tests/data/media/llm-optimize-deploy-graphic.png"
)
MULTIMODAL_IMG_URL = f"http://localhost:{IMAGE_SERVER_PORT}/llm-graphic.png"


@dataclass(frozen=True)
class ServicePorts:
    frontend_port: int
    system_port1: int
    system_port2: int


@pytest.fixture(scope="function")
def dynamo_dynamic_ports() -> Generator[ServicePorts, None, None]:
    """Allocate per-test ports for serve-style deployments.

    - frontend_port: OpenAI-compatible HTTP ingress (dynamo.frontend)
    - system_port1/system_port2: worker metrics/system ports (used by some scripts)

    Note: some disaggregated launch scripts can spawn more than two workers; if/when
    serve tests start exercising those scripts, we'll extend this fixture to allocate
    additional system ports (e.g. system_port3+ / DYN_SYSTEM_PORT3+).
    """

    frontend_port = allocate_port(DefaultPort.FRONTEND.value)
    system_ports = allocate_ports(2, DefaultPort.SYSTEM1.value)
    ports = [frontend_port, *system_ports]
    try:
        yield ServicePorts(
            frontend_port=frontend_port,
            system_port1=system_ports[0],
            system_port2=system_ports[1],
        )
    finally:
        deallocate_ports(ports)


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
