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

from tests.deploy.test_deploy import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    MIN_RESPONSE_CONTENT_LENGTH,
    TEST_PROMPT,
    validate_chat_response,
)
from tests.utils.client import send_request, wait_for_model_availability
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

    # Remove any existing HTTPRoutes referencing the kgateway-system gateway
    # to avoid routing conflicts before applying ours
    logger.info("Cleaning up existing HTTPRoutes tied to kgateway...")
    existing = subprocess.run(
        [
            "kubectl",
            "get",
            "httproutes",
            "--all-namespaces",
            "-o",
            "jsonpath={range .items[?(@.spec.parentRefs[*].namespace=='kgateway-system')]}"
            "{.metadata.namespace}{' '}{.metadata.name}{'\\n'}{end}",
        ],
        capture_output=True,
        text=True,
    )
    if existing.returncode == 0 and existing.stdout.strip():
        for line in existing.stdout.strip().splitlines():
            parts = line.strip().split()
            if len(parts) == 2:
                ns, name = parts
                logger.info(f"Deleting HTTPRoute {ns}/{name}")
                subprocess.run(
                    [
                        "kubectl",
                        "delete",
                        "httproute",
                        name,
                        "-n",
                        ns,
                        "--ignore-not-found",
                    ],
                    capture_output=True,
                    text=True,
                )

    logger.info("Applying GAIE HTTPRoute...")
    result = subprocess.run(
        ["kubectl", "apply", "-n", namespace, "-f", httproute_path],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"Failed to apply HTTPRoute: {result.stderr}"

    try:
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
                model_ready = wait_for_model_availability(
                    url=gateway_url,
                    endpoint=endpoint,
                    model=GAIE_MODEL_NAME,
                    logger=logger,
                    max_attempts=30,
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
                response = send_request(
                    url,
                    payload,
                    timeout=float(DEFAULT_REQUEST_TIMEOUT),
                    method="POST",
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
    finally:
        logger.info("Cleaning up HTTPRoute...")
        subprocess.run(
            [
                "kubectl",
                "delete",
                "-n",
                namespace,
                "-f",
                httproute_path,
                "--ignore-not-found",
            ],
            capture_output=True,
            text=True,
        )
