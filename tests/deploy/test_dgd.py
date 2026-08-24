# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
DynamoGraphDeployment tests for Kubernetes-based LLM deployments.

These tests verify that deployments can be created, become ready, and respond
to chat completion requests correctly.
"""

import asyncio
import logging
import os
import subprocess
import time

import kr8s
import pytest
import yaml

from tests.deploy.conftest import SERVING_READY_TIMEOUT_S, DeploymentTarget
from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment, _get_workspace_dir
from tests.utils.inference_endpoint import InferenceEndpoint, wait_until_serving
from tests.utils.payload_builder import deployment_smoke_chat_payload
from tests.utils.verification import run_payloads

logger = logging.getLogger(__name__)

GAIE_MODEL_NAME = "Qwen/Qwen3-0.6B"
# The install script deploys the Gateway into agentgateway-system; the
# controller provisions the proxy Service in that same namespace.
GAIE_AGW_NAMESPACE = "agentgateway-system"


@pytest.mark.framework_only
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(1200)
async def test_deployment(
    deployed_endpoint: InferenceEndpoint,
    deployment_target: DeploymentTarget,
) -> None:
    """A deployed Dynamo frontend answers a chat completion.

    Deployment, readiness and teardown are the ``deployed_endpoint`` fixture's
    job; this test only sends a payload and asserts on the response, so the
    same assertion runs against any Dynamo frontend URL.
    """
    payload = deployment_smoke_chat_payload(model=deployed_endpoint.model)
    run_payloads([payload.bind(deployed_endpoint)], log=logger)

    logger.info(
        "Deployment test PASSED for %s (source: %s, model: %s, url: %s)",
        deployment_target.test_id,
        deployment_target.source,
        deployed_endpoint.model,
        deployed_endpoint.base_url,
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

        # Gateway Programmed != Service exists; poll until the controller catches up.
        # The proxy Service lives in GAIE_AGW_NAMESPACE (where the Gateway was created),
        # not in the workload namespace.
        gateway_svcs = []
        for attempt in range(30):
            gateway_svcs = list(
                kr8s.get(
                    "services",
                    "inference-gateway",
                    namespace=GAIE_AGW_NAMESPACE,
                )
            )
            if gateway_svcs:
                break
            logger.info(
                f"Waiting for inference-gateway service in namespace {GAIE_AGW_NAMESPACE}"
                f" (attempt {attempt + 1}/30)..."
            )
            if attempt < 29:
                await asyncio.sleep(10)
        assert (
            len(gateway_svcs) > 0
        ), f"inference-gateway service not found in namespace {GAIE_AGW_NAMESPACE}"
        gateway_pf = gateway_svcs[0].portforward(remote_port=80, local_port=0)
        gateway_pf.start()
        time.sleep(2)

        try:
            # The Gateway routes on Host, so it is part of the address rather
            # than of the request under test.
            gateway = InferenceEndpoint.from_port(
                gateway_pf.local_port,
                model=GAIE_MODEL_NAME,
                headers={"Host": route_hostname},
            )
            logger.info(
                "Gateway port-forward established: %s (Host: %s)",
                gateway.base_url,
                route_hostname,
            )

            wait_until_serving(gateway, timeout=SERVING_READY_TIMEOUT_S, log=logger)

            # Same payload and same assertions as the non-GAIE deployment test;
            # only the address differs.
            payload = deployment_smoke_chat_payload(model=GAIE_MODEL_NAME)
            run_payloads([payload.bind(gateway)], log=logger)

            logger.info("GAIE deployment test PASSED via %s", gateway.base_url)
        finally:
            gateway_pf.stop()
