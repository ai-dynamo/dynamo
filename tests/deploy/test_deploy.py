# SPDX-FileCopyrightText: Copyright (c) -2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Deployment tests for Kubernetes-based LLM deployments.

These tests verify that deployments can be created, become ready, and respond
to chat completion requests correctly.
"""

import logging
import tempfile
from typing import Any, Dict, Optional

import pytest
import requests

from tests.deploy.conftest import DeploymentTarget
from tests.utils.client import wait_for_model_availability
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

logger = logging.getLogger(__name__)

# Test prompt for realistic load testing
# This prompt is substantial enough to generate meaningful responses
TEST_PROMPT = """In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, \
lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried \
beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, \
known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at \
the city's location. Your journey will take you through treacherous deserts, enchanted forests, \
and across perilous mountain ranges. Describe your first steps into the ruins of Aeloria."""

# Default test parameters
DEFAULT_MAX_TOKENS = 30
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = 120
MIN_RESPONSE_CONTENT_LENGTH = 10


def send_test_request(
    base_url: str,
    model: str,
    prompt: str = TEST_PROMPT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    timeout: int = DEFAULT_REQUEST_TIMEOUT,
) -> requests.Response:
    """Send a chat completion request to the model endpoint.

    Args:
        base_url: Base URL for the service (e.g., "http://localhost:8000")
        model: Model name to use for the request
        prompt: User prompt for the chat completion
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds

    Returns:
        Response object from the API

    Raises:
        requests.RequestException: If the request fails
    """
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    logger.info(f"Sending test request to {url} with model '{model}'")
    response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    logger.info(f"Received response with status code {response.status_code}")

    return response


def validate_chat_response(
    response: requests.Response,
    expected_model: str,
    min_content_length: int = MIN_RESPONSE_CONTENT_LENGTH,
) -> Dict[str, Any]:
    """Validate the structure and content of a chat completion response.

    Args:
        response: HTTP response from the chat completion endpoint
        expected_model: Expected model name in the response
        min_content_length: Minimum required length for response content

    Returns:
        Parsed response JSON on success

    Raises:
        AssertionError: If validation fails
    """
    # Check HTTP status
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. "
        f"Response: {response.text[:500]}"
    )

    # Parse JSON
    try:
        data = response.json()
    except requests.JSONDecodeError as e:
        pytest.fail(f"Response is not valid JSON: {e}. Response: {response.text[:500]}")

    # Validate response structure
    assert "choices" in data, f"Response missing 'choices' field: {data}"
    assert len(data["choices"]) > 0, f"Response has empty 'choices': {data}"

    choice = data["choices"][0]
    assert "message" in choice, f"Choice missing 'message' field: {choice}"

    message = choice["message"]
    assert message.get("role") == "assistant", (
        f"Expected role 'assistant', got '{message.get('role')}'"
    )
    assert "content" in message, f"Message missing 'content' field: {message}"

    content = message["content"]
    assert len(content) >= min_content_length, (
        f"Response content too short: {len(content)} chars (min: {min_content_length}). "
        f"Content: {content[:200]}"
    )

    # Validate model name
    assert "model" in data, f"Response missing 'model' field: {data}"
    assert data["model"] == expected_model, (
        f"Expected model '{expected_model}', got '{data['model']}'"
    )

    logger.info(
        f"Response validation passed: model={data['model']}, "
        f"content_length={len(content)}"
    )

    return data


def _get_model_from_spec(deployment_spec: DeploymentSpec) -> Optional[str]:
    """Extract model name from deployment spec.

    Searches through services to find a service with a model configured.
    """
    for service in deployment_spec.services:
        model = service.model
        if model:
            return model
    return None


@pytest.mark.k8s
@pytest.mark.deploy
async def test_deployment(
    deployment_target: DeploymentTarget,
    deployment_spec: DeploymentSpec,
    namespace: str,
    skip_service_restart: bool,
) -> None:
    """Test Kubernetes deployment end-to-end.

    This test:
    1. Deploys the specified configuration to Kubernetes
    2. Waits for all pods to become ready
    3. Port-forwards to the frontend service
    4. Waits for the model to be available
    5. Sends a test chat completion request
    6. Validates the response structure and content

    Args:
        deployment_target: The deployment target containing path and metadata
        deployment_spec: Configured DeploymentSpec from fixture
        namespace: Kubernetes namespace for the deployment
        skip_service_restart: Whether to skip restarting NATS/etcd services (default: True)
    """
    # Extract identifying information from the target
    framework = deployment_target.framework
    profile = deployment_target.profile

    # Extract model name from deployment spec
    model = _get_model_from_spec(deployment_spec)
    if not model:
        pytest.fail(
            f"Could not determine model name from deployment spec for "
            f"{framework}/{profile}"
        )

    logger.info(
        f"Starting deployment test for {deployment_target.test_id} "
        f"(source: {deployment_target.source}, model: {model}, namespace: {namespace})"
    )

    # Create temporary directory for logs
    log_dir = tempfile.mkdtemp(prefix=f"deploy_test_{framework}_{profile}_")
    logger.info(f"Log directory: {log_dir}")

    # Deploy and test
    async with ManagedDeployment(
        log_dir=log_dir,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        # Get frontend pod for port forwarding
        frontend_pods = deployment.get_pods([deployment.frontend_service_name])
        frontend_pod_list = frontend_pods.get(deployment.frontend_service_name, [])

        assert len(frontend_pod_list) > 0, (
            f"No frontend pods found for deployment {deployment_spec.name}"
        )

        frontend_pod = frontend_pod_list[0]
        logger.info(f"Found frontend pod: {frontend_pod.name}")

        # Setup port forwarding
        port = deployment_spec.port
        port_forward = deployment.port_forward(frontend_pod, port)
        assert port_forward is not None, (
            f"Failed to establish port forward to {frontend_pod.name}:{port}"
        )

        base_url = f"http://localhost:{port_forward.local_port}"
        logger.info(f"Port forwarding established: {base_url}")

        # Wait for model to be available
        endpoint = deployment_spec.endpoint
        model_ready = wait_for_model_availability(
            url=base_url,
            endpoint=endpoint,
            model=model,
            logger=logger,
            max_attempts=30,
        )

        assert model_ready, (
            f"Model '{model}' did not become available within the timeout period"
        )

        # Send test request
        response = send_test_request(
            base_url=base_url,
            model=model,
            prompt=TEST_PROMPT,
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )

        # Validate response
        validate_chat_response(
            response=response,
            expected_model=model,
            min_content_length=MIN_RESPONSE_CONTENT_LENGTH,
        )

        logger.info(
            f"Deployment test PASSED for {deployment_target.test_id} "
            f"(source: {deployment_target.source}, model: {model}, namespace: {namespace})"
        )
