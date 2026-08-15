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

The deployment name is fixed, so the namespace must be clear of a previous
run before starting a new one -- back-to-back manual runs collide with the
prior teardown. CI is unaffected: every nightly gets a fresh vCluster.

Runs nightly, against the nightly -efa image. EFA changes are sparse, so a
cluster-backed test on every commit would cost far more than the risk warrants;
tests/container/test_efa_image.py is the cheap per-image packaging gate and this
is the functional one.
"""

import json
import logging
import shlex
import time
from pathlib import Path

import pytest

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment
from tests.utils.client import send_request, wait_for_model_availability

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

# NIXL states which backend it instantiated, per worker. Both strings were
# observed directly: a deployment with backends:["LIBFABRIC"] removed logs
# "Backend UCX was instantiated" (plus nixl_agent.cpp warning that EFA NICs were
# detected but UCX was configured), and a correctly pinned one logs LIBFABRIC.
# This is a positive statement of backend selection rather than an inference.
NIXL_BACKEND_LIBFABRIC = "Backend LIBFABRIC was instantiated"
NIXL_BACKEND_UCX = "Backend UCX was instantiated"

# libfabric's own memory-registration lines (FI_LOG_LEVEL>=info). These pin the
# transfer to the *efa* provider specifically -- choosing the LIBFABRIC backend
# alone would not rule out libfabric selecting shm or tcp underneath.
LIBFABRIC_EFA_MARKERS = ("efa:mr:", "efa_mr_reg")

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

# The efa-node-exporter DaemonSet (monitoring namespace, hostNetwork, :9102)
# publishes per-NIC Amazon EFA counters. Unlike agent_rx_bytes -- which is a
# NIXL agent-level counter and increments identically whichever backend moved
# the bytes -- these are read from the adapter, so a delta here is direct
# evidence that traffic crossed EFA. Measured idle noise on an assigned NIC is
# zero, and the delta matches agent_rx_bytes byte-for-byte.
EFA_EXPORTER_PORT = 9102
# decode issues RDMA READs; prefill serves them. Assert on the side that proves
# each worker actually moved bytes over its own adapter.
EFA_COUNTER_BY_ROLE = {
    DECODE_SERVICE: "node_amazonefa_rdma_read_bytes",
    PREFILL_SERVICE: "node_amazonefa_rdma_read_resp_bytes",
}


def _read_pod_logs(pod, tail_lines: int = 20000) -> str:
    """Return this pod's logs, including the previous container instance.

    ``previous`` matters both ways: a worker that restarted during startup keeps
    its EFA registration lines in the prior instance (gate would false-fail on a
    healthy deployment), and a fallback that happened before a restart would
    otherwise be invisible. ``tail_lines`` bounds memory -- FI_LOG_LEVEL=info is
    extremely chatty and both workers' logs are held at once.
    """
    chunks = []
    spec = pod.raw.get("spec", {}) if hasattr(pod, "raw") else {}
    containers = [c["name"] for c in (spec.get("containers") or []) if c.get("name")]
    for container in containers or [""]:
        for previous in (True, False):
            kwargs = {"tail_lines": tail_lines, "previous": previous}
            if container:
                kwargs["container"] = container
            try:
                chunks.append("\n".join(pod.logs(**kwargs)))
            except Exception as e:  # noqa: BLE001 - no previous instance is normal
                logger.debug(
                    "logs(previous=%s) unavailable for %s/%s: %s",
                    previous,
                    pod.name,
                    container or "<default>",
                    e,
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


def assert_nixl_used_libfabric(worker_pods: dict) -> None:
    """Every worker must have instantiated the LIBFABRIC backend, and no UCX.

    Evaluated per pod on purpose. Concatenating both workers' logs and asking
    ``any()`` passes when only one side used EFA -- and one-sided fallback (say
    decode landing without a usable EFA device) is exactly the regression this
    test exists to catch.
    """
    verdicts = {}
    for role, pods in worker_pods.items():
        for pod in pods:
            logs = _read_pod_logs(pod)
            verdicts[f"{role}/{pod.name}"] = {
                "libfabric_backend": NIXL_BACKEND_LIBFABRIC in logs,
                "ucx_backend": NIXL_BACKEND_UCX in logs,
                "efa_provider": any(m in logs for m in LIBFABRIC_EFA_MARKERS),
            }

    assert verdicts, "No prefill/decode worker pods found to verify EFA usage"
    for name, v in verdicts.items():
        logger.info("NIXL backend verdict %s: %s", name, v)

    no_libfabric = sorted(n for n, v in verdicts.items() if not v["libfabric_backend"])
    assert not no_libfabric, (
        f"EFA NOT confirmed: {len(no_libfabric)} of {len(verdicts)} workers never logged "
        f"{NIXL_BACKEND_LIBFABRIC!r}: {no_libfabric}. Check that --kv-transfer-config "
        "still pins kv_connector_extra_config.backends=['LIBFABRIC']."
    )

    on_ucx = sorted(n for n, v in verdicts.items() if v["ucx_backend"])
    assert not on_ucx, (
        f"EFA NOT confirmed: {len(on_ucx)} worker(s) instantiated the UCX backend "
        f"({NIXL_BACKEND_UCX!r}): {on_ucx}. Even alongside LIBFABRIC this makes the "
        "agent byte counter ambiguous about which transport moved the KV."
    )

    no_provider = sorted(n for n, v in verdicts.items() if not v["efa_provider"])
    assert not no_provider, (
        f"EFA NOT confirmed: {len(no_provider)} worker(s) show no libfabric EFA "
        f"memory-registration lines {LIBFABRIC_EFA_MARKERS}: {no_provider}. The "
        "LIBFABRIC backend was selected but libfabric may not have used the efa "
        "provider (shm/tcp). Check FI_PROVIDER=efa and FI_LOG_LEVEL>=info."
    )
    logger.info(
        "EFA path confirmed on all %d workers: LIBFABRIC backend, no UCX, efa provider",
        len(verdicts),
    )


def _read_nixl_rx_bytes(pod) -> tuple[str, float | None]:
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
    except Exception as e:  # noqa: BLE001 - classified as a scrape failure below
        logger.warning("Could not scrape NIXL telemetry from %s: %s", pod.name, e)
        return ("scrape_failed", None)

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
    return ("ok", total) if found else ("absent", None)


def read_efa_device_counters(pod) -> dict:
    """Read this pod's OWN EFA NIC counters from the node exporter.

    Resolves the assigned device (the pod sees all 32 NICs in sysfs but is given
    exactly one ``/dev/infiniband/uverbs*``), maps it to its ibdev name, and
    returns only that device's ``node_amazonefa_*`` series. Returns {} if the
    exporter is unreachable, so callers can capability-gate.
    """
    snippet = (
        "import glob,os,json,urllib.request\n"
        "u=[os.path.basename(p) for p in glob.glob('/dev/infiniband/uverbs*')]\n"
        "if not u: print('{}'); raise SystemExit\n"
        "dev=open('/sys/class/infiniband_verbs/%s/ibdev'%u[0]).read().strip()\n"
        "ip=os.environ.get('EFA_EXPORTER_HOST','')\n"
        f"t=urllib.request.urlopen('http://%s:{EFA_EXPORTER_PORT}/metrics'%ip,timeout=10).read().decode()\n"
        "o={'_ibdev':dev}\n"
        "for ln in t.splitlines():\n"
        "    ln=ln.strip()\n"
        "    if not ln.startswith('node_amazonefa_') or dev not in ln: continue\n"
        "    try: o[ln.split('{')[0]]=float(ln.rsplit(maxsplit=1)[1])\n"
        "    except Exception: pass\n"
        "print(json.dumps(o))"
    )
    try:
        host = pod.raw["status"]["hostIP"]
        result = pod.exec(
            ["sh", "-c", f"EFA_EXPORTER_HOST={host} python3 -c {shlex.quote(snippet)}"]
        )
        for line in result.stdout.decode().splitlines():
            line = line.strip()
            if line.startswith("{"):
                parsed = json.loads(line)
                parsed["_status"] = "ok"
                return parsed
    except Exception as e:  # noqa: BLE001 - classified as a scrape failure below
        logger.warning("EFA device counters unavailable for %s: %s", pod.name, e)
        return {"_status": "scrape_failed"}
    return {"_status": "empty"}


def assert_efa_device_traffic(before: dict, after: dict, min_bytes: int) -> None:
    """Assert each worker's own EFA adapter moved at least ``min_bytes``.

    This is the only backend-independent proof in the test: it is read from the
    adapter, not from NIXL, so it cannot be satisfied by a transfer that took a
    different transport. Measured idle noise on an assigned NIC is zero.
    """
    # Fails closed. This guards the only backend-independent proof in the test, so
    # an unreachable exporter or a renamed counter must not quietly turn it into a
    # no-op. The efa-node-exporter DaemonSet runs on both aws-dev-02 (8 nodes) and
    # aws-dev-01 (13 nodes), so there is no lane where leniency here is justified.
    failed = sorted(
        role
        for snap in (before, after)
        for role, c in snap.items()
        if c.get("_status") != "ok"
    )
    assert not failed, (
        f"EFA traffic NOT confirmed: could not read EFA device counters for {failed}. "
        f"The efa-node-exporter DaemonSet publishes them on :{EFA_EXPORTER_PORT} of each "
        "node; an unreachable exporter is an infrastructure failure, not a platform "
        "without EFA telemetry."
    )

    for role, counter in EFA_COUNTER_BY_ROLE.items():
        b, a = before.get(role, {}), after.get(role, {})
        assert counter in b and counter in a, (
            f"EFA traffic NOT confirmed: counter {counter} missing for {role} "
            f"(device {a.get('_ibdev') or b.get('_ibdev')}). Skipping it would leave "
            "the adapter-level proof unasserted."
        )
        delta = a[counter] - b[counter]
        assert delta >= min_bytes, (
            f"EFA traffic NOT confirmed on {role} ({a.get('_ibdev')}): {counter} rose "
            f"{delta:,.0f} bytes across the completion, expected at least "
            f"{min_bytes:,}. The KV transfer did not cross this worker's EFA adapter."
        )
        logger.info(
            "EFA adapter traffic confirmed on %s (%s): %s +%s bytes",
            role,
            a.get("_ibdev"),
            counter,
            f"{delta:,.0f}",
        )


def assert_efa_rdma_traffic(
    before: tuple[str, float | None], after: tuple[str, float | None]
) -> None:
    """Assert the decode worker's NIXL rx-bytes counter grew across the request.

    Combined with assert_nixl_used_libfabric (which proves the *backend* is
    LIBFABRIC/EFA), a strictly increasing ``agent_rx_bytes`` proves KV bytes
    physically moved through the NIXL/EFA agent for this inference -- not merely
    that the path was configured.

    Fails closed. This lane is pinned to aws-dev-02/H100, where the exporter is
    known to work, so a scrape that errors is an infrastructure failure and must
    not silently delete the only direct proof that bytes moved. Only a
    *successful* scrape that genuinely lacks the metric is treated as a platform
    without NIXL telemetry -- and even that is reported loudly, since on this
    lane it is not expected either. A GB200 lane can add an explicit capability
    gate when it exists.
    """
    before_status, rx_before = before
    after_status, rx_after = after

    failed = [s for s in (before_status, after_status) if s == "scrape_failed"]
    assert not failed, (
        f"EFA RDMA traffic NOT confirmed: scraping NIXL {NIXL_RX_BYTES_METRIC} failed "
        f"(before={before_status}, after={after_status}). The exporter on "
        f":{NIXL_TELEMETRY_PORT} is expected to work on this lane, so this is an "
        "infrastructure failure, not a platform without telemetry."
    )

    if before_status == "absent" and after_status == "absent":
        logger.warning(
            "NIXL %s absent from a successfully scraped endpoint on both samples — "
            "skipping the RDMA-traffic assertion and relying on the libfabric log "
            "evidence. This is unexpected on this lane; check NIXL_TELEMETRY_ENABLE=y.",
            NIXL_RX_BYTES_METRIC,
        )
        return

    assert rx_before is not None and rx_after is not None, (
        f"EFA RDMA traffic NOT confirmed: NIXL {NIXL_RX_BYTES_METRIC} was present on "
        f"only one of the two samples (before={before_status}, after={after_status}). "
        "Treating the missing one as zero would be a false pass."
    )

    assert rx_after > rx_before, (
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


def assert_deployment_cleaned_up(
    deployment: ManagedDeployment, timeout: int = 180
) -> None:
    """Fail if the worker pods survive teardown.

    A pod wedged in Terminating keeps both its GPU and its VPC IP until the
    sandbox is torn down, so a teardown that quietly leaves pods behind degrades
    the shared cluster for everyone else on it -- later runs hit
    "failed to assign an IP address" or simply cannot get GPUs, and none of it
    points back here. This has been observed taking 20+ minutes on these
    clusters (Grove podclique finalizers), so it is a real failure mode, not a
    theoretical one. Better to fail this test loudly than to leak quietly.
    """
    deadline = time.monotonic() + timeout
    remaining: list[str] = []
    while time.monotonic() < deadline:
        pods = deployment.get_pods([PREFILL_SERVICE, DECODE_SERVICE])
        remaining = [pod.name for pod_list in pods.values() for pod in pod_list]
        if not remaining:
            logger.info("Teardown verified: no worker pods remain")
            return
        time.sleep(5)

    pytest.fail(
        f"Deployment teardown left {len(remaining)} worker pod(s) after {timeout}s: "
        f"{remaining}. They hold GPUs and VPC IPs until removed; force-delete them "
        "before re-running."
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
        efa_before = {
            role: read_efa_device_counters(pods[0])
            for role, pods in worker_pods.items()
            if pods
        }
        for role, c in efa_before.items():
            logger.info(
                "EFA counters before (%s, %s): %s",
                role,
                c.get("_ibdev"),
                {k: v for k, v in c.items() if k.endswith("_bytes")},
            )

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
        efa_after = {
            role: read_efa_device_counters(pods[0])
            for role, pods in worker_pods.items()
            if pods
        }

        # A successful disagg completion means KV moved prefill->decode. Prove it
        # (1) rode the LIBFABRIC/EFA backend rather than falling back to UCX, and
        # (2) physically moved bytes over EFA RDMA (the rx-bytes counter grew).
        log_nixl_layout(decode_pod)
        assert_nixl_used_libfabric(worker_pods)
        assert_efa_rdma_traffic(rx_before, rx_after)
        # Backend-independent proof, read from the adapter rather than from NIXL.
        # 1 MiB floor: the observed KV volume for this prompt is ~12.8 MB, so this
        # catches "nothing moved" without being brittle about the exact figure.
        assert_efa_device_traffic(efa_before, efa_after, min_bytes=1 << 20)

        logger.info(
            "EFA deployment test PASSED (image: %s, model: %s, namespace: %s)",
            image,
            EFA_MODEL_NAME,
            namespace,
        )

    # Outside the context manager: ManagedDeployment has now torn the deployment
    # down, so verify it actually went away rather than assuming it did.
    assert_deployment_cleaned_up(deployment)
