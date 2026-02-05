# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pytest configuration for CI deployment tests."""

import base64
import os
from pathlib import Path

import pytest


def pytest_addoption(parser):
    """Add command-line options for deploy tests."""
    parser.addoption(
        "--deploy-namespace",
        type=str,
        default=None,
        help="Kubernetes namespace for deploy tests (default: from DYNAMO_DEPLOY_NAMESPACE env var)",
    )
    parser.addoption(
        "--runtime-image",
        type=str,
        default=None,
        help="Runtime image to use for deployment (default: from DYNAMO_RUNTIME_IMAGE env var)",
    )


@pytest.fixture(scope="session")
def deploy_namespace(request):
    """Get Kubernetes namespace for deployment tests.

    Priority:
    1. --deploy-namespace CLI option
    2. DYNAMO_DEPLOY_NAMESPACE environment variable
    3. Default: "pytest-deploy"
    """
    namespace = request.config.getoption("--deploy-namespace")
    if namespace:
        return namespace

    namespace = os.environ.get("DYNAMO_DEPLOY_NAMESPACE")
    if namespace:
        return namespace

    return "pytest-deploy"


@pytest.fixture(scope="session")
def runtime_image(request):
    """Get runtime image for deployment tests.

    Priority:
    1. --runtime-image CLI option
    2. DYNAMO_RUNTIME_IMAGE environment variable
    3. Construct from framework: dynamo:{framework}-latest

    Returns:
        str: Full image path (e.g., registry.com/dynamo:sha-vllm-amd64)
    """
    image = request.config.getoption("--runtime-image")
    if image:
        return image

    image = os.environ.get("DYNAMO_RUNTIME_IMAGE")
    if image:
        return image

    # Default: construct from framework (will be overridden per-test)
    return None


@pytest.fixture(scope="session")
def kubeconfig_path(tmp_path_factory):
    """Setup kubeconfig from environment variable or use existing KUBECONFIG.

    Priority:
    1. Existing KUBECONFIG environment variable
    2. KUBECONFIG_B64 environment variable (base64 encoded, used in CI)
    3. Default: ~/.kube/config

    Returns:
        Path: Path to kubeconfig file
    """
    # If KUBECONFIG is already set, use it
    existing_kubeconfig = os.environ.get("KUBECONFIG")
    if existing_kubeconfig:
        return Path(existing_kubeconfig)

    # Check for base64-encoded kubeconfig (CI pattern)
    kubeconfig_b64 = os.environ.get("KUBECONFIG_B64")
    if kubeconfig_b64:
        tmp_dir = tmp_path_factory.mktemp("kubeconfig")
        kubeconfig_file = tmp_dir / "config"
        kubeconfig_file.write_bytes(base64.b64decode(kubeconfig_b64))
        # Set KUBECONFIG for subprocess calls
        os.environ["KUBECONFIG"] = str(kubeconfig_file)
        return kubeconfig_file

    # Default to standard location
    default_kubeconfig = Path.home() / ".kube" / "config"
    if default_kubeconfig.exists():
        return default_kubeconfig

    pytest.skip("No kubeconfig found. Set KUBECONFIG or KUBECONFIG_B64 environment variable.")


@pytest.fixture(scope="session")
def model_name():
    """Get model name for inference tests.

    Returns:
        str: Model name (default: Qwen/Qwen3-0.6B, matches CI)
    """
    return os.environ.get("MODEL_NAME", "Qwen/Qwen3-0.6B")


@pytest.fixture(scope="session")
def ingress_suffix():
    """Get ingress suffix for deployments.

    Returns:
        str: Ingress suffix (default: from DYNAMO_INGRESS_SUFFIX env var)
    """
    return os.environ.get("DYNAMO_INGRESS_SUFFIX", "")


@pytest.fixture(scope="session")
def azure_acr_hostname():
    """Get Azure ACR hostname from environment.

    Returns:
        str: ACR hostname or None
    """
    return os.environ.get("AZURE_ACR_HOSTNAME")
