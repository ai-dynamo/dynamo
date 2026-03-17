# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GAIE (Gateway API Inference Extension) deployment test.

Validates end-to-end deployment of disaggregated vLLM through the GAIE path:
  1. Patches the GAIE disagg manifest with CI-built images
  2. Applies the DynamoGraphDeployment and HTTPRoute
  3. Waits for EPP and worker pods to become ready
  4. Cleans up resources
"""

import logging
import os
import subprocess
import time

import pytest
import yaml

logger = logging.getLogger(__name__)

GAIE_DEPLOY_TIMEOUT_SECONDS = 600  # 10 minutes
POLL_INTERVAL_SECONDS = 10


def _get_workspace_dir() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _run_kubectl(*args, input_data=None):
    cmd = ["kubectl"] + list(args)
    logger.info(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, input=input_data)
    if result.stdout:
        logger.info(f"stdout: {result.stdout.strip()}")
    if result.stderr:
        logger.warning(f"stderr: {result.stderr.strip()}")
    return result


def _update_gaie_manifest(manifest_path, frontend_image, vllm_image):
    """Patch the GAIE disagg manifest with CI-built images.

    - EPP mainContainer → frontend_image
    - Worker frontendSidecar → frontend_image
    - Worker mainContainer → vllm_image
    """
    with open(manifest_path) as f:
        spec = yaml.safe_load(f)

    services = spec["spec"]["services"]

    services["Epp"]["extraPodSpec"]["mainContainer"]["image"] = frontend_image

    for worker_name in ("VllmPrefillWorker", "VllmDecodeWorker"):
        services[worker_name]["extraPodSpec"]["mainContainer"]["image"] = vllm_image
        services[worker_name]["frontendSidecar"]["image"] = frontend_image

    return spec


def _wait_for_dgd_ready(name, namespace, timeout=GAIE_DEPLOY_TIMEOUT_SECONDS):
    """Poll the DynamoGraphDeployment until Ready=True or timeout."""
    start = time.time()
    last_log = 0.0

    while True:
        elapsed = time.time() - start
        if elapsed >= timeout:
            break

        result = _run_kubectl(
            "get",
            "dynamographdeployments",
            name,
            "-n",
            namespace,
            "-o",
            "jsonpath={.status.conditions[?(@.type=='Ready')].status}",
        )
        if result.returncode == 0 and result.stdout.strip() == "True":
            logger.info(f"DGD {name} is Ready after {elapsed:.0f}s")
            return True

        if elapsed - last_log >= 60:
            logger.info(
                f"Waiting for DGD {name} to be Ready... ({elapsed:.0f}s / {timeout}s)"
            )
            last_log = elapsed

        time.sleep(POLL_INTERVAL_SECONDS)

    _run_kubectl("describe", "dynamographdeployments", name, "-n", namespace)
    return False


def _get_ready_pod_count(namespace, label_selector):
    """Return the number of pods matching label_selector that have Ready=True."""
    result = _run_kubectl(
        "get",
        "pods",
        "-n",
        namespace,
        "-l",
        label_selector,
        "-o",
        "jsonpath={range .items[*]}{.status.conditions[?(@.type=='Ready')].status}{' '}{end}",
    )
    if result.returncode != 0:
        return 0
    statuses = result.stdout.strip().split()
    return sum(1 for s in statuses if s == "True")


def _verify_pods(namespace, dgd_name, expected_services):
    """Check that every expected service has the right number of Ready pods."""
    all_ok = True
    for service_name, expected_count in expected_services.items():
        label = f"nvidia.com/selector={dgd_name}-{service_name.lower()}"
        count = _get_ready_pod_count(namespace, label)
        logger.info(f"  {service_name}: {count}/{expected_count} pods ready")
        if count < expected_count:
            all_ok = False
    return all_ok


def _dump_debug_info(namespace, dgd_name):
    """Collect debug information on failure."""
    logger.info("=== DEBUG: DGD status ===")
    _run_kubectl("describe", "dynamographdeployments", dgd_name, "-n", namespace)
    logger.info("=== DEBUG: Pods ===")
    _run_kubectl("get", "pods", "-n", namespace, "-o", "wide")
    logger.info("=== DEBUG: Events ===")
    _run_kubectl(
        "get",
        "events",
        "-n",
        namespace,
        "--sort-by=.lastTimestamp",
        "--field-selector",
        "type!=Normal",
    )


@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(900)
def test_gaie_deploy(request):
    """Test GAIE disaggregated deployment with vLLM workers.

    Applies the GAIE DynamoGraphDeployment and HTTPRoute manifests, then
    verifies the EPP pod and all worker pods reach Ready state within the
    timeout window.
    """
    frontend_image = request.config.getoption("--frontend-image")
    vllm_image = request.config.getoption("--vllm-image")
    namespace = request.config.getoption("--namespace")

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

    logger.info("Updating GAIE manifest images:")
    logger.info(f"  Frontend image: {frontend_image}")
    logger.info(f"  vLLM image: {vllm_image}")

    updated_manifest = _update_gaie_manifest(disagg_path, frontend_image, vllm_image)
    dgd_name = updated_manifest["metadata"]["name"]

    manifest_yaml = yaml.safe_dump(updated_manifest)

    logger.info(
        f"Applying DynamoGraphDeployment '{dgd_name}' to namespace '{namespace}'"
    )
    result = _run_kubectl("apply", "-n", namespace, "-f", "-", input_data=manifest_yaml)
    assert result.returncode == 0, f"Failed to apply disagg.yaml: {result.stderr}"

    logger.info("Applying GAIE HTTPRoute...")
    result = _run_kubectl("apply", "-n", namespace, "-f", httproute_path)
    assert result.returncode == 0, f"Failed to apply http-route.yaml: {result.stderr}"

    try:
        logger.info(
            f"Waiting up to {GAIE_DEPLOY_TIMEOUT_SECONDS}s for deployment to be ready..."
        )
        ready = _wait_for_dgd_ready(
            dgd_name, namespace, timeout=GAIE_DEPLOY_TIMEOUT_SECONDS
        )

        if not ready:
            _dump_debug_info(namespace, dgd_name)
            pytest.fail(
                f"DGD '{dgd_name}' did not become ready "
                f"within {GAIE_DEPLOY_TIMEOUT_SECONDS}s"
            )

        expected_services = {
            "Epp": 1,
            "VllmPrefillWorker": 1,
            "VllmDecodeWorker": 1,
        }

        pods_ok = _verify_pods(namespace, dgd_name, expected_services)
        if not pods_ok:
            _dump_debug_info(namespace, dgd_name)
            pytest.fail("Not all expected GAIE pods are ready")

        logger.info("GAIE deployment test PASSED")

    finally:
        logger.info("Cleaning up GAIE deployment resources...")
        _run_kubectl(
            "delete", "-n", namespace, "-f", httproute_path, "--ignore-not-found"
        )
        _run_kubectl(
            "delete",
            "dynamographdeployments",
            dgd_name,
            "-n",
            namespace,
            "--timeout=120s",
            "--ignore-not-found",
        )
