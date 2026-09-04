# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared machinery for the per-framework EFA verification deploy tests.

One deployment flow and one set of assertions, parameterised by an
:class:`EfaFrameworkProfile`, so that ``test_deploy_efa.py`` (vLLM) and
``test_deploy_efa_sglang.py`` (SGLang) prove the same thing about their own
framework and a fix to the evidence rules cannot land in one copy only.

What differs between frameworks is only *how the KV moves*, and that is exactly
what the profile carries: which worker initiates the NIXL transfer, which NIXL
telemetry counter therefore grows, and which pair of EFA adapter counters the
transfer shows up in. The evidence rules themselves -- LIBFABRIC backend on
every worker, no UCX anywhere, libfabric's own EFA memory-registration lines,
adapter counters that moved -- are identical and live here.
"""

import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

# The request defaults, prompt and response validation are shared with
# tests/deploy/test_dgd.py via dgd_utils, a non-test module -- so these tests
# assert the same response contract as every other deploy test, and a fix to
# that contract cannot land in one copy only.
from tests.deploy.dgd_utils import (
    DEFAULT_REQUEST_TIMEOUT,
    DEFAULT_TEMPERATURE,
    TEST_PROMPT,
    DeploymentSpec,
    ManagedDeployment,
    validate_chat_response,
)
from tests.utils.client import send_request, wait_for_model_availability

logger = logging.getLogger(__name__)

# Same model for every framework: the smallest thing that still forces a real
# prefill->decode KV transfer, so the lanes stay comparable and cheap.
EFA_MODEL_NAME = "Qwen/Qwen3-0.6B"

MANIFEST_DIR = Path(__file__).parent / "efa"

# Comfortably under the tests' own pytest timeout, so a deployment that never
# becomes ready loses the race to _wait_for_condition, which raises with pod
# statuses, conditions and warning events appended. Left at the 1800s default it
# would be pytest-timeout that fires, killing the test mid-wait with a traceback
# that says nothing about why the pods were not ready. Observed readiness on
# aws-dev-02 is ~90-150s for vLLM and ~150-250s for SGLang, so this is not a
# tight budget.
EFA_READINESS_TIMEOUT = 900

# Generate enough tokens to clear MIN_RESPONSE_CONTENT_LENGTH with margin.
# The shared DEFAULT_MAX_TOKENS=30 leaves a thin cushion above the 100-char
# minimum (a short, deterministic Qwen3-0.6B reply can land near the floor),
# so request a larger budget here to keep these single-completion tests robust.
EFA_MAX_TOKENS = 64

# NIXL states which backend it instantiated, per worker. Both strings were
# observed directly: a deployment with the LIBFABRIC pin removed logs
# "Backend UCX was instantiated" (plus nixl_agent.cpp warning that EFA NICs were
# detected but UCX was configured), and a correctly pinned one logs LIBFABRIC.
# This is a positive statement of backend selection rather than an inference.
NIXL_BACKEND_LIBFABRIC = "Backend LIBFABRIC was instantiated"
NIXL_BACKEND_UCX = "Backend UCX was instantiated"

# libfabric's own memory-registration lines (FI_LOG_LEVEL>=info). These pin the
# transfer to the *efa* provider specifically -- choosing the LIBFABRIC backend
# alone would not rule out libfabric selecting shm or tcp underneath.
LIBFABRIC_EFA_MARKERS = ("efa:mr:", "efa_mr_reg")

# NIXL Prometheus telemetry (enabled via NIXL_TELEMETRY_ENABLE=y in the
# manifests) is exposed on this port inside each worker pod. We scrape it with
# pod.exec(python3 ...) — python3 is the container entrypoint, so it is always
# present — which avoids depending on a named container port or port-forward.
NIXL_TELEMETRY_PORT = 19090

# The efa-node-exporter DaemonSet (monitoring namespace, hostNetwork, :9102)
# publishes per-NIC Amazon EFA counters. Unlike the NIXL agent counters -- which
# are agent-level and increment identically whichever backend moved the bytes --
# these are read from the adapter, so a delta here is direct evidence that
# traffic crossed EFA. Measured idle noise on an assigned NIC is zero, and the
# delta matches the NIXL counter byte-for-byte.
EFA_EXPORTER_PORT = 9102

# 1 MiB floor for the adapter-counter gate: the observed KV volume for this
# prompt is ~12.8 MB, so this catches "nothing moved" without being brittle
# about the exact figure.
EFA_MIN_TRANSFER_BYTES = 1 << 20


@dataclass(frozen=True)
class EfaFrameworkProfile:
    """Everything that differs between one framework's EFA lane and another's.

    ``nixl_counter_service``/``nixl_counter_metric`` follow the direction of the
    transfer: NIXL counts bytes on the agent that *initiates* it, so a framework
    whose decode worker pulls KV (READ) is proved by rx bytes on decode, and one
    whose prefill worker pushes KV (WRITE) by tx bytes on prefill. Asserting the
    wrong side reads a permanently-zero counter and fails a healthy deployment.
    ``efa_counter_by_role`` follows the same split at the adapter level.
    """

    name: str
    manifest_name: str
    prefill_service: str
    decode_service: str
    # How this framework is told to use LIBFABRIC, quoted back in the failure
    # message so a regression names the knob that has to be checked.
    backend_pin_hint: str
    nixl_counter_service: str
    nixl_counter_metric: str
    efa_counter_by_role: Mapping[str, str]

    @property
    def manifest_path(self) -> Path:
        """Resolved from this file rather than from where pytest was invoked."""
        return MANIFEST_DIR / self.manifest_name

    @property
    def worker_services(self) -> tuple[str, str]:
        return (self.prefill_service, self.decode_service)


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
    stay on the observable outcome: bytes moved over LIBFABRIC/EFA.
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


def assert_nixl_used_libfabric(worker_pods: dict, profile: EfaFrameworkProfile) -> None:
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
        f"{NIXL_BACKEND_LIBFABRIC!r}: {no_libfabric}. Check that {profile.backend_pin_hint}"
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


def read_nixl_agent_bytes(pod, metric: str) -> tuple[str, float | None]:
    """Return the summed NIXL byte counter ``metric`` from a worker pod.

    Scrapes the in-pod NIXL Prometheus endpoint via ``pod.exec`` and sums every
    matching sample (one per NIXL agent/label set). Returns the total, or
    ``None`` with a status of ``scrape_failed``/``absent`` when telemetry is not
    reachable or the metric is missing -- the caller decides what that means.
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
        if line.startswith(metric):
            try:
                total += float(line.rsplit(maxsplit=1)[1])
                found = True
            except (IndexError, ValueError):
                continue
    return ("ok", total) if found else ("absent", None)


def _parse_prometheus_labels(label_block: str) -> dict:
    """Parse a Prometheus label block into a dict.

    Naive on commas inside label values, which node_amazonefa_* never has --
    device names are PCI-derived (rdmap<bus>s<slot>) and port is numeric.
    """
    out = {}
    for part in label_block.split(","):
        key, sep, value = part.partition("=")
        if sep:
            out[key.strip()] = value.strip().strip('"')
    return out


def parse_efa_device_metrics(metrics_lines, dev: str) -> dict:
    """Return ``{metric_name: value}`` for exactly the device ``dev``.

    Matches the ``device`` label by equality rather than by substring. A node
    publishes one series per metric per NIC, and the result here is keyed only by
    metric name, so a loose match would let a second NIC's sample overwrite the
    assigned one -- attributing another adapter's traffic to this worker and
    passing the load-bearing gate in assert_efa_device_traffic on evidence from
    the wrong NIC. Device names can nest (rdmap16s2 is a prefix of rdmap16s27),
    so substring matching is not safe even though the current p5 fleet happens
    to name every EFA device rdmap<bus>s0.
    """
    out = {}
    for line in metrics_lines:
        line = line.strip()
        if not line.startswith("node_amazonefa_"):
            continue
        name, sep, rest = line.partition("{")
        if not sep:
            # No label block at all, so the sample cannot be attributed to a NIC.
            continue
        label_block, sep, value = rest.rpartition("}")
        if not sep:
            continue
        if _parse_prometheus_labels(label_block).get("device") != dev:
            continue
        try:
            out[name] = float(value.split()[0])
        except (ValueError, IndexError):
            continue
    return out


def read_efa_device_counters(pod) -> dict:
    """Read this pod's OWN EFA NIC counters from the node exporter.

    Resolves the assigned device (the pod sees all 32 NICs in sysfs but is given
    exactly one ``/dev/infiniband/uverbs*``) and maps it to its ibdev name. The
    in-pod snippet only fetches and coarse-filters to node_amazonefa_* lines;
    device attribution happens in parse_efa_device_metrics above, so the part
    that has to be exact is ordinary local code with unit tests rather than a
    string executed over kubectl exec.
    """
    snippet = (
        "import glob,os,json,urllib.request\n"
        "u=[os.path.basename(p) for p in glob.glob('/dev/infiniband/uverbs*')]\n"
        "if not u: print('{}'); raise SystemExit\n"
        "dev=open('/sys/class/infiniband_verbs/%s/ibdev'%u[0]).read().strip()\n"
        "ip=os.environ.get('EFA_EXPORTER_HOST','')\n"
        f"t=urllib.request.urlopen('http://%s:{EFA_EXPORTER_PORT}/metrics'%ip,timeout=10).read().decode()\n"
        "lines=[l for l in t.splitlines() if l.startswith('node_amazonefa_')]\n"
        "print(json.dumps({'_ibdev':dev,'_lines':lines}))"
    )
    try:
        host = pod.raw["status"]["hostIP"]
        result = pod.exec(
            ["sh", "-c", f"EFA_EXPORTER_HOST={host} python3 -c {shlex.quote(snippet)}"]
        )
        for line in result.stdout.decode().splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            payload = json.loads(line)
            dev = payload.get("_ibdev")
            if not dev:
                break
            parsed = parse_efa_device_metrics(payload.get("_lines", []), dev)
            parsed["_ibdev"] = dev
            parsed["_status"] = "ok"
            return parsed
    except Exception as e:  # noqa: BLE001 - classified as a scrape failure below
        logger.warning("EFA device counters unavailable for %s: %s", pod.name, e)
        return {"_status": "scrape_failed"}
    return {"_status": "empty"}


def assert_efa_device_traffic(
    before: dict,
    after: dict,
    profile: EfaFrameworkProfile,
    min_bytes: int = EFA_MIN_TRANSFER_BYTES,
) -> None:
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

    for role, counter in profile.efa_counter_by_role.items():
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
    before: tuple[str, float | None],
    after: tuple[str, float | None],
    profile: EfaFrameworkProfile,
) -> None:
    """Assert the initiating worker's NIXL byte counter grew across the request.

    Combined with assert_nixl_used_libfabric (which proves the *backend* is
    LIBFABRIC/EFA), a strictly increasing counter proves KV bytes physically
    moved through the NIXL/EFA agent for this inference -- not merely that the
    path was configured.

    Fails closed. These lanes are pinned to aws-dev-02/H100, where the exporter
    is known to work, so a scrape that errors is an infrastructure failure and
    must not silently delete the only direct proof that bytes moved. An exporter
    that is up but omits its principal counter is broken, not a platform without
    telemetry, so both cases fail here rather than being tolerated. A GB200 lane
    can add an explicit capability gate when it exists.
    """
    before_status, bytes_before = before
    after_status, bytes_after = after
    metric = profile.nixl_counter_metric

    not_ok = {
        k: v
        for k, v in (("before", before_status), ("after", after_status))
        if v != "ok"
    }
    assert not not_ok, (
        f"EFA RDMA traffic NOT confirmed: NIXL {metric} unreadable ({not_ok}) on the "
        f"{profile.nixl_counter_service} worker. 'scrape_failed' means the exporter on "
        f":{NIXL_TELEMETRY_PORT} was unreachable; 'absent' means it responded without "
        "the counter, which indicates NIXL_TELEMETRY_ENABLE did not take effect or the "
        "metric was renamed. Either way there is no evidence KV moved."
    )

    assert bytes_after > bytes_before, (
        f"EFA RDMA traffic NOT confirmed: NIXL {metric} did not increase across the "
        f"completion (before={bytes_before}, after={bytes_after}). A disagg request "
        f"must move KV between prefill and decode; a flat counter on the "
        f"{profile.nixl_counter_service} worker means no KV moved through the NIXL/EFA "
        "agent."
    )
    logger.info(
        "EFA RDMA traffic confirmed: NIXL %s rose %s -> %s bytes across the request",
        metric,
        bytes_before,
        bytes_after,
    )


async def run_efa_deployment_check(
    profile: EfaFrameworkProfile,
    image: str,
    namespace: str,
    skip_service_restart: bool,
    request,
) -> None:
    """Deploy this framework's EFA manifest and prove the KV rode EFA.

    1. Deploys the profile's manifest with the EFA image under test
    2. Waits for the frontend and BOTH prefill and decode workers to be ready
    3. Port-forwards to the frontend and waits for the model to be available
    4. Baselines the NIXL agent byte counter on the worker that initiates the
       transfer, and both workers' own EFA adapter counters
    5. Sends a chat completion (which requires a prefill->decode KV transfer)
    6. Validates the response
    7. Asserts the worker logs prove NIXL used the LIBFABRIC/EFA backend, that
       the NIXL counter grew, and that each worker's own EFA adapter moved the
       bytes -- i.e. KV physically crossed EFA, not just that the path was
       configured.
    """
    assert image, "--image is required for the EFA deploy test"
    assert namespace, "--namespace is required for the EFA deploy test"

    deployment_spec = DeploymentSpec(profile.manifest_path)
    deployment_spec.namespace = namespace
    # Single EFA-tagged image for every service (the framework runtime image also
    # provides the frontend entrypoint).
    deployment_spec.set_image(image)

    logger.info(
        f"Starting {profile.name} EFA deploy test (image: {image}, "
        f"model: {EFA_MODEL_NAME}, namespace: {namespace})"
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        readiness_timeout=EFA_READINESS_TIMEOUT,
    ) as deployment:
        # Both workers must be present — disaggregation is the whole point.
        worker_pods = deployment.get_pods(list(profile.worker_services))
        for svc in profile.worker_services:
            assert worker_pods.get(svc), f"No pods found for worker service {svc}"
        # The worker that initiates the NIXL transfer is the one whose agent
        # counts the bytes; see EfaFrameworkProfile.
        counter_pod = worker_pods[profile.nixl_counter_service][0]

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

        # Baseline the NIXL byte counter before the request, so we can prove the
        # completion below makes it grow (KV moved over EFA).
        nixl_before = read_nixl_agent_bytes(counter_pod, profile.nixl_counter_metric)
        logger.info(
            "NIXL %s before request: %s", profile.nixl_counter_metric, nixl_before
        )
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
        validate_chat_response(response=response, expected_model=EFA_MODEL_NAME)

        nixl_after = read_nixl_agent_bytes(counter_pod, profile.nixl_counter_metric)
        logger.info(
            "NIXL %s after request: %s", profile.nixl_counter_metric, nixl_after
        )
        efa_after = {
            role: read_efa_device_counters(pods[0])
            for role, pods in worker_pods.items()
            if pods
        }

        # A successful disagg completion means KV moved prefill->decode. Prove it
        # (1) rode the LIBFABRIC/EFA backend rather than falling back to UCX, and
        # (2) physically moved bytes over EFA RDMA.
        log_nixl_layout(counter_pod)
        assert_nixl_used_libfabric(worker_pods, profile)
        assert_efa_rdma_traffic(nixl_before, nixl_after, profile)
        # Backend-independent proof, read from the adapter rather than from NIXL.
        assert_efa_device_traffic(efa_before, efa_after, profile)

        logger.info(
            "EFA deployment test PASSED (framework: %s, image: %s, model: %s, "
            "namespace: %s)",
            profile.name,
            image,
            EFA_MODEL_NAME,
            namespace,
        )
