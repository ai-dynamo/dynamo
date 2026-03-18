# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GAIE (Gateway API Inference Extension) deployment test.

Validates end-to-end deployment of disaggregated vLLM through the GAIE path:
  1. Patches the GAIE disagg manifest with CI-built images
  2. Applies the DynamoGraphDeployment and HTTPRoute
  3. Waits for EPP and worker pods to become ready
  4. Port-forwards to the inference gateway
  5. Sends a chat completion request and validates the response
"""

import logging
import os
import subprocess
import time

import pytest
import requests
import yaml

from tests.deploy.test_deploy import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    MIN_RESPONSE_CONTENT_LENGTH,
    TEST_PROMPT,
    validate_chat_response,
)
from tests.utils.managed_deployment import (
    DeploymentSpec,
    ManagedDeployment,
    _get_workspace_dir,
)

logger = logging.getLogger(__name__)

GAIE_MODEL_NAME = "Qwen/Qwen3-0.6B"


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(900)
async def test_gaie_deployment(
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
    vllm_image = request.config.getoption("--vllm-image")

    assert frontend_image, "--frontend-image is required for GAIE deploy test"
    assert vllm_image, "--vllm-image is required for GAIE deploy test"
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
    logger.info(f"vLLM image: {vllm_image}")

    deployment_spec.set_image(frontend_image, service_name="Epp")
    for worker in ("VllmPrefillWorker", "VllmDecodeWorker"):
        deployment_spec.set_image(vllm_image, service_name=worker)
        deployment_spec.set_frontend_sidecar_image(frontend_image, service_name=worker)

    # Each concurrent run gets a unique hostname so HTTPRoutes don't conflict
    # when multiple runs share the same gateway
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
    assert result.returncode == 0, f"Failed to apply HTTPRoute: {result.stderr}"

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        frontend_service_name="Epp",
    ) as deployment:
        epp_pods = deployment.get_pods(["Epp"])
        epp_pod_list = epp_pods.get("Epp", [])
        assert len(epp_pod_list) > 0, "No EPP pods found for GAIE deployment"
        logger.info(f"Found EPP pod: {epp_pod_list[0].name}")

        gateway_pf = subprocess.Popen(
            [
                "kubectl",
                "port-forward",
                "svc/inference-gateway",
                "8000:80",
                "-n",
                "kgateway-system",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        try:
            time.sleep(3)
            assert gateway_pf.poll() is None, (
                "Gateway port-forward exited unexpectedly: "
                f"{gateway_pf.stderr.read().decode() if gateway_pf.stderr else ''}"
            )

            gateway_url = "http://localhost:8000"
            logger.info(f"Gateway port-forward established: {gateway_url}")

            endpoint = deployment_spec.endpoint
            headers = {"Host": route_hostname}
            logger.info(f"Using Host header: {route_hostname}")

            test_url = f"{gateway_url}{endpoint}"
            model_ready = False
            for attempt in range(30):
                try:
                    r = requests.post(
                        test_url,
                        json={
                            "model": GAIE_MODEL_NAME,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_tokens": 1,
                            "stream": False,
                        },
                        headers=headers,
                        timeout=60,
                    )
                    if r.status_code == 200:
                        logger.info(
                            f"Model '{GAIE_MODEL_NAME}' is available and responding"
                        )
                        time.sleep(5)
                        model_ready = True
                        break
                except Exception as e:
                    logger.warning(
                        f"Model availability check failed (attempt {attempt + 1}): {e}"
                    )
                time.sleep(10 if attempt < 5 else 5)

            assert model_ready, (
                f"Model '{GAIE_MODEL_NAME}' did not become available "
                f"within the timeout period"
            )

            payload = {
                "model": GAIE_MODEL_NAME,
                "messages": [{"role": "user", "content": TEST_PROMPT}],
                "max_tokens": DEFAULT_MAX_TOKENS,
                "temperature": DEFAULT_TEMPERATURE,
                "stream": False,
            }
            logger.info(f"Sending inference request to {test_url}")
            response = requests.post(
                test_url,
                json=payload,
                headers=headers,
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )

            validate_chat_response(
                response=response,
                expected_model=GAIE_MODEL_NAME,
                min_content_length=MIN_RESPONSE_CONTENT_LENGTH,
            )

            logger.info("GAIE deployment test PASSED")
        finally:
            gateway_pf.terminate()
            gateway_pf.wait(timeout=5)
