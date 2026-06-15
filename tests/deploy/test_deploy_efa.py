# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
EFA verification deploy test.

Verifies that an EFA-tagged image built from the commit under test can run
Dynamo with Elastic Fabric Adapter (EFA) fully enabled: a disaggregated vLLM
stack deploys on an EFA-capable cluster, serves a chat completion, and the
prefill->decode KV-cache transfer rides NIXL -> LIBFABRIC -> EFA.

This test is NOT part of the auto-discovered deploy-test matrix. It uses an
explicit manifest (tests/deploy/efa/disagg-efa.yaml) and the
``framework_with_efa`` marker, and only makes sense on a cluster with p5/EFA
nodes (the standard CI vCluster lacks RDMA/EFA, which is why the matrix test
skips vLLM disagg). Run it explicitly, e.g.:

    pytest tests/deploy/test_deploy_efa.py -m framework_with_efa \
        --image=<efa-vllm-runtime-image> --namespace=<ns> -v -s

There is no CI job wired for this yet (no EFA-capable runner/cluster is
connected to CI); see the EFA deploy-test plan for the CI-wiring follow-up.
"""

import logging
import os
from typing import Optional

import pytest

from tests.deploy.test_deploy import (
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

EFA_MODEL_NAME = "Qwen/Qwen3-0.6B"
PREFILL_SERVICE = "VllmPrefillWorker"
DECODE_SERVICE = "VllmDecodeWorker"

# Generate enough tokens to clear MIN_RESPONSE_CONTENT_LENGTH with margin.
# The shared DEFAULT_MAX_TOKENS=30 leaves a thin cushion above the 100-char
# minimum (a short, deterministic Qwen3-0.6B reply can land near the floor),
# so request a larger budget here to keep this single-completion test robust.
EFA_MAX_TOKENS = 64

# Substrings that prove NIXL registered memory regions with the EFA libfabric
# provider (i.e. the KV-cache transfer actually used LIBFABRIC -> EFA). These
# appear in the worker logs when FI_LOG_LEVEL>=info, e.g.:
#   libfabric:1234:5678:efa:mr:efa_mr_reg_impl():...
# See the test_efa_on_aws skill (Senthil's check #3) for the canonical signal.
LIBFABRIC_EFA_MARKERS = ("efa:mr:", "efa_mr_reg")
# If NIXL silently falls back to UCX (e.g. the kv-transfer-config lost the
# LIBFABRIC backend), the worker logs the UCX rcache tuning line instead and
# never emits the libfabric:efa:mr lines above.
UCX_FALLBACK_MARKER = "Setting UCX_RCACHE_MAX_UNRELEASED"

# NIXL Prometheus telemetry (enabled via NIXL_TELEMETRY_ENABLE=y in the manifest)
# is exposed on this port inside each worker pod. We scrape it with
# pod.exec(python3 ...) — python3 is the container entrypoint, so it is always
# present — which avoids depending on a named container port or port-forward.
NIXL_TELEMETRY_PORT = 19090
# With NIXL READ semantics (vLLM _read_blocks) the decode worker pulls KV from
# prefill, so transferred bytes register as rx on the decode side (prefill tx
# stays ~0). agent_rx_bytes is therefore the authoritative "bytes moved over the
# NIXL/EFA agent" counter. Metric name per the test_efa_on_aws skill; TP=1 here,
# so the rank-0-only telemetry limitation does not apply.
NIXL_RX_BYTES_METRIC = "agent_rx_bytes"


def _read_pod_logs(pod) -> str:
    """Return the concatenated logs of every container in a pod.

    Multi-container pods (worker + optional sidecar/init containers) reject
    ``pod.logs()`` without an explicit ``container=``, so iterate the manifest's
    containers like ManagedDeployment.get_pod_manifest_logs_metrics does.
    """
    container_names = []
    try:
        spec = pod.raw.get("spec", {}) if hasattr(pod, "raw") else {}
        for c in (spec.get("initContainers") or []) + (spec.get("containers") or []):
            if c.get("name"):
                container_names.append(c["name"])
    except Exception as e:  # noqa: BLE001 - diagnostics only
        logger.debug(f"Failed to resolve containers for {pod.name}: {e}")

    if not container_names:
        container_names = [""]

    chunks = []
    for container in container_names:
        try:
            lines = pod.logs(container=container) if container else pod.logs()
            chunks.append("\n".join(lines))
        except Exception as e:  # noqa: BLE001 - a container may have no logs yet
            logger.debug(
                f"Failed to fetch logs for {pod.name} "
                f"container={container or '<default>'}: {e}"
            )
    return "\n".join(chunks)


def assert_nixl_used_libfabric(deployment: ManagedDeployment) -> None:
    """Fail unless the worker logs prove NIXL used the LIBFABRIC/EFA backend.

    This is the cheap "EFA fully enabled" proof: a successful disaggregated
    completion shows KV transfer worked, and these log lines show it rode
    LIBFABRIC -> EFA rather than silently falling back to UCX/TCP.
    """
    worker_pods = deployment.get_pods([DECODE_SERVICE, PREFILL_SERVICE])
    all_pods = [p for pods in worker_pods.values() for p in pods]
    assert all_pods, "No prefill/decode worker pods found to verify EFA usage"

    combined = "\n".join(_read_pod_logs(p) for p in all_pods)

    found_libfabric = any(marker in combined for marker in LIBFABRIC_EFA_MARKERS)
    saw_ucx_fallback = UCX_FALLBACK_MARKER in combined

    assert found_libfabric, (
        "EFA NOT confirmed: worker logs contain no libfabric:efa memory-registration "
        f"lines ({LIBFABRIC_EFA_MARKERS}). "
        + (
            "Found UCX fallback marker instead — NIXL fell back to UCX; check that "
            "--kv-transfer-config still has kv_connector_extra_config.backends=['LIBFABRIC']."
            if saw_ucx_fallback
            else "Check FI_PROVIDER=efa, FI_LOG_LEVEL>=info, and the EFA device/resources."
        )
    )
    logger.info("EFA path confirmed: NIXL registered memory regions via LIBFABRIC/EFA")


def _read_nixl_rx_bytes(pod) -> Optional[float]:
    """Return the summed NIXL ``agent_rx_bytes`` counter from a worker pod.

    Scrapes the in-pod NIXL Prometheus endpoint via ``pod.exec`` and sums every
    ``agent_rx_bytes`` sample (one per NIXL agent/label set). Returns the total
    bytes NIXL has received (KV pulled from prefill), or ``None`` if telemetry is
    not reachable or the metric is absent.
    """
    snippet = (
        "import urllib.request;"
        "print(urllib.request.urlopen("
        f"'http://localhost:{NIXL_TELEMETRY_PORT}/metrics', timeout=5).read().decode())"
    )
    try:
        result = pod.exec(["python3", "-c", snippet])
        text = result.stdout.decode()
    except Exception as e:  # noqa: BLE001 - telemetry may not be up yet
        logger.warning(f"Could not scrape NIXL telemetry from {pod.name}: {e}")
        return None

    total = 0.0
    found = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # Samples look like: agent_rx_bytes{agent="..."} 1234 (also matches a
        # possible _total suffix). The value is the final whitespace field.
        if line.startswith(NIXL_RX_BYTES_METRIC):
            try:
                total += float(line.rsplit(maxsplit=1)[1])
                found = True
            except (IndexError, ValueError):
                continue
    return total if found else None


def assert_efa_rdma_traffic(
    rx_before: Optional[float], rx_after: Optional[float]
) -> None:
    """Fail unless the decode worker's NIXL rx-bytes counter grew across the request.

    This is the direct "traffic actually went over EFA RDMA" proof. Combined with
    assert_nixl_used_libfabric (which proves the *backend* is LIBFABRIC/EFA), a
    strictly increasing ``agent_rx_bytes`` proves KV bytes physically moved through
    the NIXL/EFA agent for this inference — not merely that the path was configured.
    """
    assert rx_after is not None, (
        "EFA RDMA traffic NOT confirmed: could not read NIXL "
        f"{NIXL_RX_BYTES_METRIC} from the decode worker. Check NIXL_TELEMETRY_ENABLE=y "
        f"and that the exporter is listening on :{NIXL_TELEMETRY_PORT}."
    )
    baseline = rx_before or 0.0
    assert rx_after > baseline, (
        f"EFA RDMA traffic NOT confirmed: NIXL {NIXL_RX_BYTES_METRIC} did not "
        f"increase across the completion (before={rx_before}, after={rx_after}). "
        "A disagg request must pull KV from prefill to decode; a flat counter means "
        "no KV moved through the NIXL/EFA agent."
    )
    logger.info(
        "EFA RDMA traffic confirmed: NIXL %s rose %s -> %s bytes across the request",
        NIXL_RX_BYTES_METRIC,
        rx_before,
        rx_after,
    )


@pytest.mark.framework_with_efa
@pytest.mark.vllm
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.timeout(1200)
async def test_efa_deployment(
    image: str,
    namespace: str,
    skip_service_restart: bool,
    request,
) -> None:
    """Deploy a disaggregated vLLM stack with EFA enabled and verify it serves.

    This test:
    1. Deploys tests/deploy/efa/disagg-efa.yaml with the EFA image under test
    2. Waits for the frontend and BOTH prefill and decode workers to be ready
    3. Port-forwards to the frontend and waits for the model to be available
    4. Baselines the decode worker's NIXL agent_rx_bytes telemetry counter
    5. Sends a chat completion (which requires prefill->decode KV transfer)
    6. Validates the response
    7. Asserts the worker logs prove NIXL used the LIBFABRIC/EFA backend, AND
       that agent_rx_bytes grew across the request — i.e. KV bytes physically
       moved over EFA RDMA, not just that the LIBFABRIC path was configured.
    """
    assert image, "--image is required for the EFA deploy test"
    assert namespace, "--namespace is required for the EFA deploy test"

    workspace = _get_workspace_dir()
    manifest_path = os.path.join(workspace, "tests", "deploy", "efa", "disagg-efa.yaml")
    assert os.path.exists(manifest_path), f"EFA manifest not found: {manifest_path}"

    deployment_spec = DeploymentSpec(manifest_path)
    deployment_spec.namespace = namespace
    # Single EFA-tagged image for every service (the vllm-runtime image also
    # provides the frontend entrypoint).
    deployment_spec.set_image(image)

    logger.info(
        f"Starting EFA deploy test (image: {image}, model: {EFA_MODEL_NAME}, "
        f"namespace: {namespace})"
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        # Both workers must be present — disaggregation is the whole point.
        worker_pods = deployment.get_pods([PREFILL_SERVICE, DECODE_SERVICE])
        for svc in (PREFILL_SERVICE, DECODE_SERVICE):
            assert worker_pods.get(svc), f"No pods found for worker service {svc}"
        # Decode worker is the NIXL READ sink — its agent_rx_bytes counter is what
        # we baseline and re-read to prove KV bytes actually moved over EFA.
        decode_pod = worker_pods[DECODE_SERVICE][0]

        frontend_pods = deployment.get_pods([deployment.frontend_service_name])
        frontend_pod_list = frontend_pods.get(deployment.frontend_service_name, [])
        assert frontend_pod_list, "No frontend pods found for EFA deployment"
        frontend_pod = frontend_pod_list[0]
        logger.info(f"Found frontend pod: {frontend_pod.name}")

        port = deployment_spec.port
        port_forward = deployment.port_forward(frontend_pod, port)
        assert (
            port_forward is not None
        ), f"Failed to establish port forward to {frontend_pod.name}:{port}"
        base_url = f"http://localhost:{port_forward.local_port}"
        logger.info(f"Port forwarding established: {base_url}")

        endpoint = deployment_spec.endpoint
        model_ready = wait_for_model_availability(
            url=base_url,
            endpoint=endpoint,
            model=EFA_MODEL_NAME,
            logger=logger,
            max_attempts=30,
        )
        assert (
            model_ready
        ), f"Model '{EFA_MODEL_NAME}' did not become available within the timeout"

        # Baseline the decode worker's NIXL rx-bytes counter before the request,
        # so we can prove the completion below makes it grow (KV pulled over EFA).
        rx_before = _read_nixl_rx_bytes(decode_pod)
        logger.info(f"NIXL {NIXL_RX_BYTES_METRIC} before request: {rx_before}")

        url = f"{base_url}{endpoint}"
        payload = {
            "model": EFA_MODEL_NAME,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "max_tokens": EFA_MAX_TOKENS,
            "temperature": DEFAULT_TEMPERATURE,
            "stream": False,
        }
        response = send_request(
            url, payload, timeout=float(DEFAULT_REQUEST_TIMEOUT), method="POST"
        )
        validate_chat_response(
            response=response,
            expected_model=EFA_MODEL_NAME,
            min_content_length=MIN_RESPONSE_CONTENT_LENGTH,
        )

        rx_after = _read_nixl_rx_bytes(decode_pod)
        logger.info(f"NIXL {NIXL_RX_BYTES_METRIC} after request: {rx_after}")

        # A successful disagg completion means KV moved prefill->decode. Prove it
        # (1) rode the LIBFABRIC/EFA backend rather than falling back to UCX, and
        # (2) physically moved bytes over EFA RDMA (the rx-bytes counter grew).
        assert_nixl_used_libfabric(deployment)
        assert_efa_rdma_traffic(rx_before, rx_after)

        logger.info(
            f"EFA deployment test PASSED (image: {image}, model: {EFA_MODEL_NAME}, "
            f"namespace: {namespace})"
        )
