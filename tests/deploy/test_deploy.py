# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Deployment tests for Kubernetes-based LLM deployments.

These tests verify that deployments can be created, become ready, and respond
to chat completion requests correctly.
"""

import logging
import os
import subprocess
import time
from typing import Any, Dict

import kr8s
import pytest
import requests
import yaml

from tests.deploy.conftest import DeploymentTarget
from tests.utils.client import send_request, wait_for_model_availability
from tests.utils.managed_deployment import (
    DeploymentSpec,
    ManagedDeployment,
    _get_workspace_dir,
)

logger = logging.getLogger(__name__)

# Test prompt designed to validate model capabilities:
# - Long enough to test context handling (multiple sentences, ~150 words)
# - Descriptive content requiring multi-sentence responses
# - Consistent across test runs for reproducibility
# This prompt is maintained from the original shell-based deployment tests.
TEST_PROMPT = """In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, \
lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried \
beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, \
known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at \
the city's location. Your journey will take you through treacherous deserts, enchanted forests, \
and across perilous mountain ranges. Describe your first steps into the ruins of Aeloria."""

DEFAULT_MAX_TOKENS = 30
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = 120
# Minimum response content length to validate that the model is generating meaningful output.
# This matches the validation threshold from the original shell-based deployment tests.
MIN_RESPONSE_CONTENT_LENGTH = 100
GAIE_MODEL_NAME = "Qwen/Qwen3-0.6B"


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

    try:
        data = response.json()
    except ValueError as e:
        pytest.fail(f"Response is not valid JSON: {e}. Response: {response.text[:500]}")

    assert "choices" in data, f"Response missing 'choices' field: {data}"
    assert len(data["choices"]) > 0, f"Response has empty 'choices': {data}"

    choice = data["choices"][0]
    assert "message" in choice, f"Choice missing 'message' field: {choice}"

    message = choice["message"]
    assert (
        message.get("role") == "assistant"
    ), f"Expected role 'assistant', got '{message.get('role')}'"
    assert "content" in message, f"Message missing 'content' field: {message}"

    content = message["content"]
    assert len(content) >= min_content_length, (
        f"Response content too short: {len(content)} chars (min: {min_content_length}). "
        f"Content: {content[:200]}"
    )

    assert "model" in data, f"Response missing 'model' field: {data}"
    assert (
        data["model"] == expected_model
    ), f"Expected model '{expected_model}', got '{data['model']}'"

    logger.info(
        f"Response validation passed: model={data['model']}, "
        f"content_length={len(content)}"
    )

    return data


@pytest.mark.framework_only
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(1200)
async def test_deployment(
    deployment_target: DeploymentTarget,
    deployment_spec: DeploymentSpec,
    namespace: str,
    skip_service_restart: bool,
    request,
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
        skip_service_restart: Whether to skip restarting NATS/etcd services (default: True).
            Use --restart-services flag to restart services before deployment.
        request: Pytest request object for accessing test metadata
    """
    # Extract identifying information from the target
    framework = deployment_target.framework
    profile = deployment_target.profile

    model = next((s.model for s in deployment_spec.services if s.model), None)
    if not model:
        pytest.fail(
            f"Could not determine model name from deployment spec for "
            f"{framework}/{profile}"
        )

    logger.info(
        f"Starting deployment test for {deployment_target.test_id} "
        f"(source: {deployment_target.source}, model: {model}, namespace: {namespace})"
    )
    logger.info(f"Log directory: {request.node.name}")

    # Deploy and test
    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        # Get frontend pod for port forwarding
        frontend_pods = deployment.get_pods([deployment.frontend_service_name])
        frontend_pod_list = frontend_pods.get(deployment.frontend_service_name, [])

        assert (
            len(frontend_pod_list) > 0
        ), f"No frontend pods found for deployment {deployment_spec.name}"

        frontend_pod = frontend_pod_list[0]
        logger.info(f"Found frontend pod: {frontend_pod.name}")

        # Setup port forwarding
        port = deployment_spec.port
        port_forward = deployment.port_forward(frontend_pod, port)
        assert (
            port_forward is not None
        ), f"Failed to establish port forward to {frontend_pod.name}:{port}"

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

        assert (
            model_ready
        ), f"Model '{model}' did not become available within the timeout period"

        # Send test request
        url = f"{base_url}{endpoint}"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "stream": False,
        }
        response = send_request(
            url, payload, timeout=float(DEFAULT_REQUEST_TIMEOUT), method="POST"
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


# GAIE (Gateway API Inference Extension) deployment test
@pytest.mark.framework_with_gaie
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(900)
async def test_gaie_deployment(
    image: str,
    namespace: str,
    skip_service_restart: bool,
    request,
) -> None:
    """Test GAIE disaggregated deployment with vLLM workers.

    Applies the GAIE DynamoGraphDeployment (with CI-built images) and the
    companion HTTPRoute, then verifies inference works end-to-end through
    the full Gateway path.
    """
    frontend_image = request.config.getoption("--frontend-image")
    worker_image = image

    assert frontend_image, "--frontend-image is required for GAIE deploy test"
    assert worker_image, "--image is required for GAIE deploy test"
    assert namespace, "--namespace is required for GAIE deploy test"

    workspace = _get_workspace_dir()
    gaie_dir = os.path.join(workspace, "examples", "backends", "vllm", "deploy", "gaie")
    disagg_path = os.path.join(gaie_dir, "disagg.yaml")
    httproute_path = os.path.join(gaie_dir, "http-route.yaml")

    assert os.path.exists(disagg_path), f"disagg.yaml not found: {disagg_path}"
    assert os.path.exists(
        httproute_path
    ), f"http-route.yaml not found: {httproute_path}"

    deployment_spec = DeploymentSpec(disagg_path)
    deployment_spec.namespace = namespace

    logger.info(f"Frontend image: {frontend_image}")
    logger.info(f"Worker image: {worker_image}")

    deployment_spec.set_image(frontend_image, service_name="Epp")
    for worker in ("VllmPrefillWorker", "VllmDecodeWorker"):
        deployment_spec.set_image(worker_image, service_name=worker)
        deployment_spec.set_frontend_sidecar_image(frontend_image, service_name=worker)

    route_hostname = f"{namespace}.example.com"
    logger.info(f"HTTPRoute hostname: {route_hostname}")

    with open(httproute_path) as f:
        httproute_spec = yaml.safe_load(f)
    httproute_spec["spec"]["hostnames"] = [route_hostname]
    httproute_yaml = yaml.safe_dump(httproute_spec)

    logger.info("Applying GAIE HTTPRoute...")
    result = subprocess.run(
        ["kubectl", "apply", "-n", namespace, "-f", "-"],
        input=httproute_yaml,
        capture_output=True,
        text=True,
    )
    logger.info(f"HTTPRoute apply stdout: {result.stdout}")
    if result.stderr:
        logger.warning(f"HTTPRoute apply stderr: {result.stderr}")
    assert result.returncode == 0, f"Failed to apply HTTPRoute: {result.stderr}"

    # Debug: verify namespace state before creating DGD
    logger.info(f"Namespace: {namespace}")
    ns_check = subprocess.run(
        ["kubectl", "get", "namespace", namespace],
        capture_output=True,
        text=True,
    )
    logger.info(f"Namespace check: {ns_check.stdout.strip()}")
    if ns_check.returncode != 0:
        logger.error(f"Namespace not found: {ns_check.stderr}")

    # Debug: check if operator CRD is registered
    crd_check = subprocess.run(
        ["kubectl", "get", "crd", "dynamographdeployments.nvidia.com"],
        capture_output=True,
        text=True,
    )
    logger.info(f"CRD check: {crd_check.stdout.strip()}")
    if crd_check.returncode != 0:
        logger.error(f"CRD not found: {crd_check.stderr}")

    # Debug: check operator pod status
    operator_check = subprocess.run(
        [
            "kubectl",
            "get",
            "pods",
            "-n",
            namespace,
            "-l",
            "app.kubernetes.io/name=dynamo-operator",
        ],
        capture_output=True,
        text=True,
    )
    logger.info(f"Operator pods: {operator_check.stdout.strip()}")

    # Debug: log the full deployment spec being submitted
    logger.info(f"DGD name: {deployment_spec.name}")
    logger.info(f"DGD namespace: {deployment_spec.namespace}")
    logger.info(f"DGD services: {[s.name for s in deployment_spec.services]}")

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        frontend_service_name="Epp",
    ) as deployment:
        # Debug: check what DGDs exist after creation
        dgd_check = subprocess.run(
            ["kubectl", "get", "dynamographdeployments", "-n", namespace],
            capture_output=True,
            text=True,
        )
        logger.info(f"DGDs after creation: {dgd_check.stdout.strip()}")

        pod_check = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-o", "wide"],
            capture_output=True,
            text=True,
        )
        logger.info(f"Pods after creation: {pod_check.stdout.strip()}")
        epp_pods = deployment.get_pods(["Epp"])
        epp_pod_list = epp_pods.get("Epp", [])
        assert len(epp_pod_list) > 0, "No EPP pods found for GAIE deployment"
        logger.info(f"Found EPP pod: {epp_pod_list[0].name}")

        gateway_svcs = list(
            kr8s.get("services", "inference-gateway", namespace=namespace)
        )
        assert (
            len(gateway_svcs) > 0
        ), f"inference-gateway service not found in namespace {namespace}"
        gateway_pf = gateway_svcs[0].portforward(remote_port=80, local_port=0)
        gateway_pf.start()
        time.sleep(2)

        try:
            gateway_url = f"http://localhost:{gateway_pf.local_port}"
            logger.info(f"Gateway port-forward established: {gateway_url}")

            endpoint = deployment_spec.endpoint
            headers = {"Host": route_hostname}
            logger.info(f"Using Host header: {route_hostname}")

            model_ready = wait_for_model_availability(
                url=gateway_url,
                endpoint=endpoint,
                model=GAIE_MODEL_NAME,
                logger=logger,
                max_attempts=30,
                headers=headers,
            )
            assert model_ready, (
                f"Model '{GAIE_MODEL_NAME}' did not become available "
                f"within the timeout period"
            )

            url = f"{gateway_url}{endpoint}"
            payload = {
                "model": GAIE_MODEL_NAME,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "stream": False,
            }
            logger.info(f"Sending inference request to {url}")
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )

            validate_chat_response(
                response=response,
                expected_model=GAIE_MODEL_NAME,
                min_content_length=MIN_RESPONSE_CONTENT_LENGTH,
            )

            data = response.json()
            content = data["choices"][0]["message"]["content"]
            logger.info(
                f"GAIE deployment test PASSED | "
                f"model={data['model']}, status={response.status_code}, "
                f"response_length={len(content)} chars\n"
                f"Model response: {content}"
            )
        finally:
            gateway_pf.stop()
