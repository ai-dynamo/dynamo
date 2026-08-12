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

Runs nightly, against the nightly -efa image. EFA changes are sparse, so a
cluster-backed test on every commit would cost far more than the risk warrants;
tests/container/test_efa_image.py is the cheap per-image packaging gate and this
is the functional one.
"""

import logging
from pathlib import Path

import pytest

from tests.utils.client import send_request, wait_for_model_availability
from tests.utils.managed_deployment import DeploymentSpec, ManagedDeployment

logger = logging.getLogger(__name__)

EFA_MODEL_NAME = "Qwen/Qwen3-0.6B"
PREFILL_SERVICE = "VllmPrefillWorker"
DECODE_SERVICE = "VllmDecodeWorker"

# Deliberately self-contained rather than imported from test_deploy.py. Importing
# one test module from another breaks the pre-commit marker report, which
# collects only the changed files and cannot resolve the sibling module, and it
# would couple this test's pass/fail to edits in the deploy-matrix test.
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = 120
MIN_RESPONSE_CONTENT_LENGTH = 100

# A long prompt on purpose: the point of this test is that KV moves from prefill
# to decode, so there needs to be enough of it to register and transfer.
TEST_PROMPT = """In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, \
lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried \
beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, \
known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at \
the city's location. Your journey will take you through treacherous deserts, enchanted forests, \
and across perilous mountain ranges. Describe your first steps into the ruins of Aeloria."""


def validate_completion(response, expected_model: str) -> None:
    """Assert the chat completion is a well-formed, non-trivial answer."""
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. "
        f"Response: {response.text[:500]}"
    )
    try:
        data = response.json()
    except ValueError as e:
        pytest.fail(f"Response is not valid JSON: {e}. Response: {response.text[:500]}")

    choices = data.get("choices") or []
    assert choices, f"Response has no choices: {data}"
    content = (choices[0].get("message") or {}).get("content") or ""
    assert (
        data.get("model") == expected_model
    ), f"Expected model {expected_model!r}, got {data.get('model')!r}"
    assert len(content) >= MIN_RESPONSE_CONTENT_LENGTH, (
        f"Response content is {len(content)} chars, expected at least "
        f"{MIN_RESPONSE_CONTENT_LENGTH}: {content!r}"
    )
    logger.info(
        "Response validation passed: model=%s, content_length=%d",
        expected_model,
        len(content),
    )


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
        logger.debug("Failed to resolve containers for %s: %s", pod.name, e)

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


def log_nixl_layout(pod) -> None:
    """Record where NIXL actually lives in the worker image. Diagnostic only.

    Deliberately not asserted on. The layout moves between framework images and
    between releases -- vLLM and TRT-LLM expose the canonical
    /opt/nvidia/nvda_nixl tree via NIXL_PLUGIN_DIR, while the CUDA SGLang image
    gets NIXL from the pip wheel and exports no NIXL_* variables at all -- so a
    test that pins the layout breaks on the next repackaging without any EFA
    regression having occurred. Logging it keeps a failed run diagnosable
    ("NIXL was over here, and these plugins were present") while the assertions
    below stay on the observable outcome: bytes moved over LIBFABRIC/EFA.
    """
    snippet = (
        "import os;"
        "d=os.environ.get('NIXL_PLUGIN_DIR','');"
        "print('NIXL_PLUGIN_DIR=', d or '<unset>');"
        "print('NIXL_LIB_DIR=', os.environ.get('NIXL_LIB_DIR','') or '<unset>');"
        "print('LD_PRELOAD=', os.environ.get('LD_PRELOAD','') or '<unset>');"
        "print('EFA_VERSION=', os.environ.get('EFA_VERSION','') or '<unset>');"
        "print('plugins=', sorted(os.listdir(d)) if d and os.path.isdir(d) else '<no plugin dir>')"
    )
    try:
        result = pod.exec(["python3", "-c", snippet])
        logger.info("NIXL layout in %s:\n%s", pod.name, result.stdout.decode())
    except Exception as e:  # noqa: BLE001 - diagnostics only, never fail the test
        logger.warning("Could not read NIXL layout from %s: %s", pod.name, e)


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


def _read_nixl_rx_bytes(pod) -> float | None:
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
        logger.warning("Could not scrape NIXL telemetry from %s: %s", pod.name, e)
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


def assert_efa_rdma_traffic(rx_before: float | None, rx_after: float | None) -> None:
    """Assert the decode worker's NIXL rx-bytes counter grew across the request.

    This is the direct "traffic actually went over EFA RDMA" proof. Combined with
    assert_nixl_used_libfabric (which proves the *backend* is LIBFABRIC/EFA), a
    strictly increasing ``agent_rx_bytes`` proves KV bytes physically moved through
    the NIXL/EFA agent for this inference — not merely that the path was configured.

    Capability-gated rather than mandatory: the NIXL Prometheus exporter is not
    reachable on every platform we run this on (it has not been observed working
    on GB200), and the libfabric log evidence is the assertion that must hold
    everywhere. When the counter cannot be read at all we log loudly and leave
    the verdict to assert_nixl_used_libfabric; when it CAN be read, it must grow.
    """
    if rx_after is None:
        logger.warning(
            "NIXL %s not readable on this platform — skipping the RDMA-traffic "
            "assertion and relying on the libfabric log evidence. Check "
            "NIXL_TELEMETRY_ENABLE=y and the exporter on :%s if this is unexpected.",
            NIXL_RX_BYTES_METRIC,
            NIXL_TELEMETRY_PORT,
        )
        return
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
@pytest.mark.nightly
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

    # Resolved from this file rather than the workspace root, so the test does
    # not depend on a private helper or on where pytest was invoked from.
    manifest_path = Path(__file__).parent / "efa" / "disagg-efa.yaml"

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
        logger.info("Found frontend pod: %s", frontend_pod.name)

        port = deployment_spec.port
        port_forward = deployment.port_forward(frontend_pod, port)
        assert (
            port_forward is not None
        ), f"Failed to establish port forward to {frontend_pod.name}:{port}"
        base_url = f"http://localhost:{port_forward.local_port}"
        logger.info("Port forwarding established: %s", base_url)

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
        logger.info("NIXL %s before request: %s", NIXL_RX_BYTES_METRIC, rx_before)

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
        validate_completion(response, expected_model=EFA_MODEL_NAME)

        rx_after = _read_nixl_rx_bytes(decode_pod)
        logger.info("NIXL %s after request: %s", NIXL_RX_BYTES_METRIC, rx_after)

        # A successful disagg completion means KV moved prefill->decode. Prove it
        # (1) rode the LIBFABRIC/EFA backend rather than falling back to UCX, and
        # (2) physically moved bytes over EFA RDMA (the rx-bytes counter grew).
        log_nixl_layout(decode_pod)
        assert_nixl_used_libfabric(deployment)
        assert_efa_rdma_traffic(rx_before, rx_after)

        logger.info(
            f"EFA deployment test PASSED (image: {image}, model: {EFA_MODEL_NAME}, "
            f"namespace: {namespace})"
        )
