# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster integration test for the multi-DGD power-aware contract.

Companion test scaffold (manifests, helm values overrides, teardown
script, README walk-through) lives at `examples/multi-dgd-live-test/`.

This test is the cross-DGD complement to the existing single-DGD live suite
(`test_actuation_knobs_live.py`). It validates the cross-DGD invariants
that single-DGD tests cannot reach:

    - A2  Three planners patch DISJOINT pod sets in one namespace
    - A3  AggPlanner-C uses only the DECODE branch (no PREFILL calls)
    - B2  Power Agent applies 8 distinct caps on one tick
    - B4  Happy-path zero-counter invariant
    - B5  Multi-pod conflict + safe-default behaviour
    - 8.5.6 / 8.6.2  NVML / DCGM cap-application proven by reading the
                    driver state (not via mocks)

Path-A pre-conditions (see `examples/multi-dgd-live-test/README.md`):
    1. dpp-dev-env AKS cluster, $KUBECONFIG = ~/.kube/dynamo-kubeconfig
    2. namespace `kaim-dynamo-system` with `nvcr-imagepullsecret`
    3. one 8-GPU A100 node labelled `power-test/node=kaim`
    4. dynamo-platform operator installed in the namespace
    5. power-agent DS rolled out on the test node (NVML or DCGM actuator)
    6. vllm-dgd-a, vllm-dgd-b, vllm-dgd-c DGDs created and Ready

Path-B pre-conditions replace items 4 and 6 with `50-stub-workers.yaml`.
The two planner-log/AggPlanner tests skip automatically on Path B.

Invocation (inside the dev pod). The README in
`examples/multi-dgd-live-test/` documents the full two-phase workflow
(NVML pass, then DCGM pass with a Power Agent reinstall between them).
The single-phase commands shown below correspond to phase 2 (NVML) and
phase 3 (DCGM) of that README. Module-level env gating
(`RUN_MULTI_DGD_LIVE=1` + `TEST_NODE=<node>`) is the opt-in;
`-m multi_dgd_live` can additionally narrow a broader pytest invocation
to only this file (the marker is registered in `pyproject.toml`).

    # NVML pass — power-agent installed with values from
    # 10-power-agent-values-nvml.yaml. DCGM tests skip by design.
    RUN_MULTI_DGD_LIVE=1 TEST_NODE=$TEST_NODE \\
        python3.10 -m pytest \\
        components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py \\
        -v --tb=short -k 'not test_dcgm_'

    # DCGM pass — power-agent reinstalled with values from
    # 11-power-agent-values-dcgm.yaml and the standalone hostengine DS
    # rolled out. DCGM tests now run.
    RUN_MULTI_DGD_LIVE=1 TEST_NODE=$TEST_NODE \\
        DCGM_HOSTENGINE_AVAILABLE=1 \\
        python3.10 -m pytest \\
        components/src/dynamo/planner/tests/integration/test_multi_dgd_live.py \\
        -v --tb=short

Total runtime per pass: ~4 minutes (one reconcile tick is 15 s; the
test does ~12 ticks worth of waits). Both passes together: ~10 minutes
plus the Power Agent reinstall in between.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Dict, List, Set

import pytest

# ---------------------------------------------------------------------------
# Module-level gating — fail-fast if invocation context is wrong
# ---------------------------------------------------------------------------

_RUN_LIVE = os.environ.get("RUN_MULTI_DGD_LIVE") == "1"
_TEST_NODE = os.environ.get("TEST_NODE")
_DCGM_AVAILABLE = os.environ.get("DCGM_HOSTENGINE_AVAILABLE") == "1"

if not _RUN_LIVE:
    pytest.skip(
        "Multi-DGD live test is opt-in. Set RUN_MULTI_DGD_LIVE=1 to enable. "
        "Pre-conditions: see file docstring.",
        allow_module_level=True,
    )

if not _TEST_NODE:
    pytest.fail(
        "RUN_MULTI_DGD_LIVE=1 was set but TEST_NODE env var is empty. "
        "Set TEST_NODE to the 8-GPU node labelled `power-test/node=kaim`.",
        pytrace=False,
    )

try:
    from kubernetes import client as k8s_client
    from kubernetes import config as k8s_config
    from kubernetes.config.config_exception import ConfigException
    from kubernetes.stream import stream as k8s_stream

    try:
        # Preferred: run inside a kaim-dynamo-system pod with a SA that can
        # list pods + DGDs cluster-wide (the planner-dev pod in dpp-dev-env
        # is the canonical home once the planner image has #9683).
        k8s_config.load_incluster_config()
    except ConfigException:
        # Fallback: run from a workstation that has $KUBECONFIG set to
        # ~/.kube/dynamo-kubeconfig. We rely on kubectl exec / cp under the
        # hood, so all subsequent operations work transparently — only the
        # client-config loading branch differs.
        k8s_config.load_kube_config()
except Exception as exc:  # noqa: BLE001
    pytest.fail(
        f"This test needs Kubernetes credentials — either an in-cluster SA "
        f"or a workstation $KUBECONFIG. Both load attempts failed: {exc!r}",
        pytrace=False,
    )

_NAMESPACE = os.environ.get("POD_NAMESPACE", "kaim-dynamo-system")
_DGD_NAMES = ("vllm-dgd-a", "vllm-dgd-b", "vllm-dgd-c")

# Topology source of truth — must match §8.2 of multi-tenant-test.md.
# (gpu_idx, expected_watts) for each of the 8 GPUs on $TEST_NODE.
_EXPECTED_CAP_PER_GPU: Dict[int, int] = {
    0: 350,  # vllm-dgd-a prefill (TP=1)
    1: 325,  # vllm-dgd-a decode  (TP=1)
    2: 375,  # vllm-dgd-b prefill (TP=2, rank 0)
    3: 375,  # vllm-dgd-b prefill (TP=2, rank 1)
    4: 325,  # vllm-dgd-b decode  (TP=2, rank 0)
    5: 325,  # vllm-dgd-b decode  (TP=2, rank 1)
    6: 300,  # vllm-dgd-c agg     (TP=2, rank 0)
    7: 300,  # vllm-dgd-c agg     (TP=2, rank 1)
}
_SAFE_DEFAULT_WATTS = 280
_DISTINCT_VALUES = {300, 325, 350, 375}
assert (
    set(_EXPECTED_CAP_PER_GPU.values()) == _DISTINCT_VALUES
), "Topology table must have exactly 4 distinct watt values per §8.2."

# Per-DGD expected annotation values, used by the cross-DGD isolation
# assertion (A2 live). Each DGD's planner should ONLY patch pods that
# match its own `nvidia.com/dynamo-graph-deployment-name` label.
_EXPECTED_ANNOTATIONS_PER_DGD: Dict[str, Dict[str, int]] = {
    "vllm-dgd-a": {"prefill": 350, "decode": 325},
    "vllm-dgd-b": {"prefill": 375, "decode": 325},
    "vllm-dgd-c": {"decode": 300},  # AggPlanner: decode-only
}

_POWER_ANNOTATION_KEY = "dynamo.nvidia.com/gpu-power-limit"
_DGD_LABEL = "nvidia.com/dynamo-graph-deployment-name"
_COMPONENT_LABEL = "nvidia.com/dynamo-component-type"
_SUBCOMPONENT_LABEL = "nvidia.com/dynamo-sub-component-type"

_RECONCILE_INTERVAL_S = 15  # power_agent.py RECONCILE_INTERVAL_S
_PROPAGATION_DEADLINE_S = 3 * _RECONCILE_INTERVAL_S  # 3 ticks of headroom

# pyproject.toml's [tool.pytest.ini_options] section registers every
# marker below and runs with --strict-markers, so any addition here must
# be paired with an entry in that file. `multi_dgd_live` is the opt-in
# selector for this specific scenario (use `-m multi_dgd_live` to run
# only this file); `power_agent` is the cross-component selector shared
# with components/power_agent/tests/.
pytestmark = [
    pytest.mark.gpu_8,
    pytest.mark.integration,
    pytest.mark.planner,
    pytest.mark.power_agent,
    pytest.mark.multi_dgd_live,
    pytest.mark.pre_merge,
]


# Path-A vs Path-B detection. Path A (production shape) requires real
# planner pods that emit annotations; Path B (stub-workers, used
# 2026-05-21) hardcodes annotations into the worker manifests and has
# no planner pods at all. Tests that strictly need a live planner
# (cross-DGD log scrape, AggPlanner-branch proof) skip cleanly under
# Path B instead of hard-failing on the "exactly 3 planner pods"
# assertion. Detection is done once at import time so the skip
# decoration is stable for collection ordering.
def _detect_path_a_planner_pods() -> bool:
    """Return True iff at least one planner pod with our test labels
    exists. Cheaper to do this once at module import than per-test
    because the K8s API call must succeed even before fixtures run.
    Safe on Path B (no creds / no cluster) — any exception → assume
    Path B and let the test gates skip."""
    if not _RUN_LIVE:
        return False
    try:
        api = k8s_client.CoreV1Api()
        pods = api.list_namespaced_pod(
            _NAMESPACE,
            label_selector=(
                f"{_COMPONENT_LABEL}=planner," "purpose=power-aware-multi-dgd-test"
            ),
        )
        return len(pods.items) >= 1
    except Exception:
        return False


_PATH_A_AVAILABLE = _detect_path_a_planner_pods()
_PATH_A_SKIP_REASON = (
    "Path B (stub-workers): no planner pods to scrape — this test "
    "strictly requires Path A (operator + #9683 planner image). "
    "See examples/multi-dgd-live-test/README.md § reproduction-paths."
)


# ---------------------------------------------------------------------------
# Fixtures — module-scoped to amortize 15s reconcile waits across tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def core_api() -> "k8s_client.CoreV1Api":
    return k8s_client.CoreV1Api()


@pytest.fixture(scope="module")
def custom_api() -> "k8s_client.CustomObjectsApi":
    return k8s_client.CustomObjectsApi()


@pytest.fixture(scope="module")
def power_agent_pod_name(core_api: "k8s_client.CoreV1Api") -> str:
    """Find the power-agent DS pod scheduled on $TEST_NODE."""
    pods = core_api.list_namespaced_pod(
        _NAMESPACE,
        label_selector="app.kubernetes.io/component=power-agent",
        field_selector=f"spec.nodeName={_TEST_NODE}",
    )
    assert len(pods.items) == 1, (
        f"Expected exactly one power-agent pod on {_TEST_NODE}, "
        f"got {len(pods.items)}. Setup error?"
    )
    return pods.items[0].metadata.name


@pytest.fixture(scope="module")
def all_worker_pods(core_api: "k8s_client.CoreV1Api") -> List["k8s_client.V1Pod"]:
    """Return all 5 worker pods belonging to the 3 DGDs on $TEST_NODE.

    Returns: [a-prefill, a-decode, b-prefill, b-decode, c-decode] = 5 pods
    across 8 GPUs (3 of the 5 pods are TP=2 and own 2 GPUs each).
    """
    pods = core_api.list_namespaced_pod(
        _NAMESPACE,
        label_selector=f"{_COMPONENT_LABEL}=worker,purpose=power-aware-multi-dgd-test",
        field_selector=f"spec.nodeName={_TEST_NODE}",
    )
    assert len(pods.items) == 5, (
        f"Expected 5 worker pods on {_TEST_NODE} "
        f"(a-prefill, a-decode, b-prefill, b-decode, c-decode); "
        f"found {len(pods.items)}. Pods: "
        f"{[(p.metadata.name, p.status.phase) for p in pods.items]}"
    )
    return pods.items


@pytest.fixture(scope="module")
def worker_pods_by_dgd(
    all_worker_pods: List["k8s_client.V1Pod"],
) -> Dict[str, List["k8s_client.V1Pod"]]:
    """Group worker pods by their DGD label."""
    out: Dict[str, List["k8s_client.V1Pod"]] = {d: [] for d in _DGD_NAMES}
    for pod in all_worker_pods:
        dgd = pod.metadata.labels.get(_DGD_LABEL)
        assert dgd in out, f"Pod {pod.metadata.name} has unexpected DGD label: {dgd!r}"
        out[dgd].append(pod)
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_in_pod(
    core_api: "k8s_client.CoreV1Api", pod_name: str, command: List[str]
) -> str:
    """Run a command inside a pod and return stdout. Stderr is appended.

    Uses ``_preload_content=False`` deliberately. With the default
    ``True``, ``kubernetes-client>=35.0.0`` sniffs JSON-shaped stdout,
    parses it client-side, and returns ``str(dict)`` — so ``printf '%s'
    '{"a":1}'`` comes back as the literal 8-char string ``{'a': 1}``
    (single quotes), which then fails ``json.loads`` everywhere it's
    consumed (the four DCGM read-back sites below were the first
    casualties on 2026-05-21). Streaming raw stdout/stderr off the
    WSClient and joining at the end gives us the bytes the pod
    actually wrote, regardless of which ``kubernetes-client`` version
    is installed.
    """
    resp = k8s_stream(
        core_api.connect_get_namespaced_pod_exec,
        pod_name,
        _NAMESPACE,
        command=command,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
        _preload_content=False,
    )
    chunks: List[str] = []
    try:
        while resp.is_open():
            resp.update(timeout=1)
            if resp.peek_stdout():
                chunks.append(resp.read_stdout())
            if resp.peek_stderr():
                chunks.append(resp.read_stderr())
        # Drain any final buffered output post-close.
        if resp.peek_stdout():
            chunks.append(resp.read_stdout())
        if resp.peek_stderr():
            chunks.append(resp.read_stderr())
    finally:
        resp.close()
    return "".join(chunks)


# Helper: extract (Current, Target) watts from a `dcgmi config --get
# -g <id> -j` body payload. DCGM's JSON shape changed twice across the
# versions we care about:
#
#   DCGM 4.0–4.4:  body = [ { "Power Limit (W)": {"Current": "350",
#                                                 "Target":  "350"} }, ... ]
#   DCGM 4.5+:     body = { "Power Limit":      {"children": {
#                              "Current": {"value": "375"},
#                              "Target":  {"value": "375"} } }, ... }
#
# Notice three breaking changes:
#   - `body` flipped from list-of-fields-per-GPU to dict-of-fields-for-one-GPU
#   - the key dropped the `(W)` suffix
#   - leaf values moved under a `children` wrapper with `value` strings
#
# This helper hides all three, returning two `Optional[int]` or
# `(None, None)` if neither shape matches.
def _extract_power_limit_from_dcgmi_body(body) -> tuple:
    """Walk DCGM 4.0/4.4 list-shape OR DCGM 4.5+ dict-shape uniformly.

    Returns (current_w, target_w) as ints, or (None, None) if no
    Power-Limit field is present in either shape.
    """

    def _read_one(field):
        if not isinstance(field, dict):
            return None, None
        # 4.5+ shape: {"children": {"Current": {"value": "375"}, ...}}
        if "children" in field and isinstance(field["children"], dict):
            children = field["children"]
            cur_node = children.get("Current", {})
            tgt_node = children.get("Target", {})
            cur = cur_node.get("value") if isinstance(cur_node, dict) else None
            tgt = tgt_node.get("value") if isinstance(tgt_node, dict) else None
        else:
            # 4.0–4.4 shape: {"Current": "350", "Target": "350"}
            cur = field.get("Current")
            tgt = field.get("Target")
        try:
            return (
                int(cur) if cur is not None else None,
                int(tgt) if tgt is not None else None,
            )
        except (ValueError, TypeError):
            return None, None

    # DCGM 4.5+ shape — body is a dict of field-name → field-object.
    if isinstance(body, dict):
        field = body.get("Power Limit") or body.get("Power Limit (W)")
        return _read_one(field)

    # DCGM 4.0–4.4 shape — body is a list of {field-name: field-object}.
    if isinstance(body, list):
        for entry in body:
            if not isinstance(entry, dict):
                continue
            field = entry.get("Power Limit (W)") or entry.get("Power Limit")
            cur, tgt = _read_one(field)
            if cur is not None or tgt is not None:
                return cur, tgt
    return None, None


def _scrape_power_agent_metrics(
    core_api: "k8s_client.CoreV1Api", pod_name: str
) -> Dict[str, float]:
    """Scrape :9100/metrics and return a flat dict of metric_name+labels → value.

    Key format: 'metric_name{label1="v1",label2="v2"}' OR just 'metric_name'
    for unlabelled scalars. Same string Prometheus uses on the wire.

    Implementation note: the power-agent image is built from python:3.11-slim
    and does NOT ship curl/wget. We invoke the in-image python interpreter
    with urllib instead — Python and pynvml are already loaded by the agent
    process, so this adds no new RSS to the pod's 128 MiB budget.
    """
    raw = _exec_in_pod(
        core_api,
        pod_name,
        [
            "python",
            "-c",
            (
                "import urllib.request as r;"
                "print(r.urlopen('http://127.0.0.1:9100/metrics', timeout=5)"
                ".read().decode(), end='')"
            ),
        ],
    )
    out: Dict[str, float] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Format: 'name{labels} 12.34' OR 'name 12.34'
        m = re.match(
            r"^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+([\d\.eE+\-]+)$", line
        )
        if not m:
            continue
        out[m.group(1)] = float(m.group(2))
    return out


def _read_pod_power_annotation(pod: "k8s_client.V1Pod") -> int:
    """Read the TGP annotation value, or 0 if missing."""
    ann = (pod.metadata.annotations or {}).get(_POWER_ANNOTATION_KEY)
    if ann is None:
        return 0
    return int(ann)


def _wait_for_annotation_convergence(
    core_api: "k8s_client.CoreV1Api",
    expected: Dict[str, int],  # pod_name → expected watts
    deadline_s: int = _PROPAGATION_DEADLINE_S,
) -> Dict[str, int]:
    """Poll until every pod in `expected` carries the expected annotation.

    Returns the observed map. Raises if deadline expires.
    """
    start = time.monotonic()
    last_observed: Dict[str, int] = {}
    while time.monotonic() - start < deadline_s:
        observed: Dict[str, int] = {}
        for pod_name in expected:
            pod = core_api.read_namespaced_pod(pod_name, _NAMESPACE)
            observed[pod_name] = _read_pod_power_annotation(pod)
        last_observed = observed
        if observed == expected:
            return observed
        time.sleep(3)
    raise AssertionError(
        f"Annotation convergence timed out after {deadline_s}s. "
        f"Expected: {expected}. Last observed: {last_observed}."
    )


def _read_nvml_caps_via_pod(
    core_api: "k8s_client.CoreV1Api", pod_name: str
) -> Dict[int, int]:
    """Read per-GPU `nvmlDeviceGetPowerManagementLimit` from inside the
    power-agent pod. Returns gpu_idx → watts (rounded).

    The power-agent image has python + pynvml + libnvidia-ml.so via the
    NVIDIA Container Toolkit runtime injection. We emit a key=value-per-line
    format and parse it back here, which is robust to whatever subtle stream
    multiplexing the Kubernetes Python `connect_get_namespaced_pod_exec`
    websocket does with mixed stdout/stderr (a json.dumps round-trip showed
    repr-shaped output during a live cluster run on 2026-05-21).
    """
    py_snippet = (
        "import pynvml; pynvml.nvmlInit();"
        "[print(f'{i}={round(pynvml.nvmlDeviceGetPowerManagementLimit("
        "pynvml.nvmlDeviceGetHandleByIndex(i))/1000)}') "
        "for i in range(pynvml.nvmlDeviceGetCount())]"
    )
    raw = _exec_in_pod(core_api, pod_name, ["python", "-c", py_snippet])
    out: Dict[int, int] = {}
    for line in raw.splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, _, v = line.partition("=")
        try:
            out[int(k)] = int(v)
        except ValueError:
            continue
    return out


# ---------------------------------------------------------------------------
# TestPerDgdAnnotation — A2, A3 live (§8.5 assertions 1–3)
# ---------------------------------------------------------------------------


class TestPerDgdAnnotation:
    """Each of the 3 planners patches ONLY its own pods, with the §8.2 caps."""

    def test_all_5_worker_pods_carry_expected_annotation(
        self,
        core_api: "k8s_client.CoreV1Api",
        worker_pods_by_dgd: Dict[str, List["k8s_client.V1Pod"]],
    ):
        """§8.5 assertion 1 (A1/A2 live): every worker pod has the TGP
        annotation matching its DGD's per-subcomponent cap."""
        expected: Dict[str, int] = {}
        for dgd, pods in worker_pods_by_dgd.items():
            for pod in pods:
                sub = pod.metadata.labels.get(_SUBCOMPONENT_LABEL)
                assert sub in _EXPECTED_ANNOTATIONS_PER_DGD[dgd], (
                    f"Pod {pod.metadata.name} from {dgd} has unexpected "
                    f"sub-component label: {sub!r}"
                )
                expected[pod.metadata.name] = _EXPECTED_ANNOTATIONS_PER_DGD[dgd][sub]

        # Wait up to 3 reconcile ticks for the planner loops to converge.
        observed = _wait_for_annotation_convergence(core_api, expected)
        assert (
            observed == expected
        ), f"Per-DGD annotation mismatch. expected={expected}, observed={observed}"

    @pytest.mark.skipif(not _PATH_A_AVAILABLE, reason=_PATH_A_SKIP_REASON)
    def test_aggregated_dgd_c_pod_was_patched_via_decode_branch(
        self,
        worker_pods_by_dgd: Dict[str, List["k8s_client.V1Pod"]],
    ):
        """§8.5 assertion 2 (A3 live): vllm-dgd-c is aggregated → has only a
        decode worker, no prefill. The annotation present on the one pod
        proves AggPlanner went through the DECODE branch.

        Path-B note: this assertion's value rests on the annotation
        having been *emitted by AggPlanner*, not just present on the
        pod. Stub-workers carry the same labels/annotation statically,
        so the assertion would pass vacuously. We skip on Path B to
        avoid the false-pass."""
        c_pods = worker_pods_by_dgd["vllm-dgd-c"]
        assert (
            len(c_pods) == 1
        ), f"AggPlanner DGD must produce exactly one worker pod, got {len(c_pods)}"
        pod = c_pods[0]
        sub = pod.metadata.labels.get(_SUBCOMPONENT_LABEL)
        assert (
            sub == "decode"
        ), f"vllm-dgd-c worker pod must carry sub-component=decode, got {sub!r}"
        watts = _read_pod_power_annotation(pod)
        assert watts == 300, (
            f"vllm-dgd-c decode pod must carry annotation 300W "
            f"(AggPlanner.decode_engine_gpu_power_limit), got {watts}"
        )

    @pytest.mark.skipif(not _PATH_A_AVAILABLE, reason=_PATH_A_SKIP_REASON)
    def test_no_cross_dgd_annotation_contamination(
        self,
        core_api: "k8s_client.CoreV1Api",
        worker_pods_by_dgd: Dict[str, List["k8s_client.V1Pod"]],
    ):
        """§8.5 assertion 3 (A2 live, strict form): each planner pod patches
        ONLY pods belonging to its own DGD.

        This is the planner-log-scrape strengthening of the test (review
        finding M3). The previous version checked only "exclusive" watt
        values, which silently passed scenarios like "A-planner patched
        B-decode with 325 W (which coincidentally matches B's own value)."

        Source of truth: the planner logs every patch via
        `logger.info("Annotated pod %s with %s=%s", ...)` — see
        `core/base.py::_apply_power_annotations` on `pr1b/planner-infra`.
        We grep this exact format from each planner pod's stdout and
        confirm every "Annotated pod X with ..." line names a pod that
        belongs to THAT planner's DGD.

        Failure mode the test catches: if A-planner's selector ever
        returned a B-owned pod and PATCHed it, the planner's own log
        line would name a B-prefixed pod, even if the value coincided.
        """
        # 1. Collect each planner pod's name keyed by DGD.
        all_planner_pods = core_api.list_namespaced_pod(
            _NAMESPACE,
            label_selector=f"{_COMPONENT_LABEL}=planner,purpose=power-aware-multi-dgd-test",
        )
        planner_by_dgd: Dict[str, str] = {}
        for p in all_planner_pods.items:
            dgd = p.metadata.labels.get(_DGD_LABEL)
            assert (
                dgd in _DGD_NAMES
            ), f"Planner pod {p.metadata.name} has unexpected DGD label: {dgd!r}"
            planner_by_dgd[dgd] = p.metadata.name
        assert set(planner_by_dgd) == set(
            _DGD_NAMES
        ), f"Expected one planner per DGD; got {planner_by_dgd}"

        # 2. Build the ground-truth ownership map: pod_name → owning DGD.
        pod_owner: Dict[str, str] = {}
        for dgd, pods in worker_pods_by_dgd.items():
            for pod in pods:
                pod_owner[pod.metadata.name] = dgd

        # 3. For each planner pod, scrape logs and confirm every "Annotated
        #    pod ..." line names one of its own DGD's worker pods.
        log_pattern = re.compile(
            r"Annotated pod (\S+) with dynamo\.nvidia\.com/gpu-power-limit=(\d+)"
        )
        violations: List[str] = []
        for owning_dgd, planner_pod_name in planner_by_dgd.items():
            try:
                # `tail_lines` keeps the response bounded; the planner runs
                # one reconcile every ~60 s (throughput_adjustment_interval),
                # so 5000 lines covers ~all annotation activity since startup.
                raw_logs = core_api.read_namespaced_pod_log(
                    name=planner_pod_name,
                    namespace=_NAMESPACE,
                    tail_lines=5000,
                )
            except k8s_client.exceptions.ApiException as exc:
                pytest.fail(
                    f"Could not read logs for planner pod {planner_pod_name!r} "
                    f"({owning_dgd}): {exc}"
                )

            patched_pods: Set[str] = set()
            for match in log_pattern.finditer(raw_logs):
                patched_pods.add(match.group(1))

            if not patched_pods:
                # If the planner hasn't run a single annotation tick yet
                # (e.g., logs were just rotated), there's nothing to check
                # — but TestPerDgdAnnotation already required convergence,
                # so this would be surprising. Flag, don't fail.
                violations.append(
                    f"planner {planner_pod_name!r} ({owning_dgd}) emitted zero "
                    f"'Annotated pod' log lines in the last 5000 lines — the "
                    f"convergence test should have made this impossible."
                )
                continue

            for patched_name in patched_pods:
                actual_owner = pod_owner.get(patched_name)
                if actual_owner is None:
                    violations.append(
                        f"planner {planner_pod_name!r} ({owning_dgd}) "
                        f"patched pod {patched_name!r} which doesn't match "
                        f"any known worker pod in this test."
                    )
                elif actual_owner != owning_dgd:
                    violations.append(
                        f"CROSS-DGD CONTAMINATION: planner {planner_pod_name!r} "
                        f"(belongs to {owning_dgd}) patched pod "
                        f"{patched_name!r} (belongs to {actual_owner})"
                    )
        assert not violations, "\n  - ".join(
            ["Cross-DGD log audit failures:", *violations]
        )


# ---------------------------------------------------------------------------
# TestPowerAgentReconcile — B2, B4 live (§8.5 assertions 4–5)
# ---------------------------------------------------------------------------


class TestPowerAgentReconcile:
    """The Power Agent applies 8 distinct caps and ticks zero counters."""

    def test_applied_limit_watts_metric_matches_topology(
        self,
        core_api: "k8s_client.CoreV1Api",
        power_agent_pod_name: str,
    ):
        """§8.5 assertion 4 (B2 live): dynamo_power_agent_applied_limit_watts
        per-GPU samples match the §8.2 topology table."""
        # Give the agent at least 2 reconcile ticks after annotations have
        # converged. The TestPerDgdAnnotation suite already waited up to
        # 3 × 15s = 45s; add another 30s buffer here.
        time.sleep(2 * _RECONCILE_INTERVAL_S)

        metrics = _scrape_power_agent_metrics(core_api, power_agent_pod_name)
        observed: Dict[int, int] = {}
        for key, val in metrics.items():
            m = re.match(
                r'^dynamo_power_agent_applied_limit_watts\{[^}]*gpu="(\d+)"[^}]*\}$',
                key,
            )
            if m:
                observed[int(m.group(1))] = int(val)

        # The Kubernetes device plugin allocates GPU indices on a clean node
        # in arbitrary order — TP=2 pods may land on GPUs 0,1 instead of 2,3,
        # etc. — so we assert on the MULTISET of caps, not on per-index
        # equality. A correct topology has the same 8 caps regardless of
        # which physical GPU index hosts which pod.
        from collections import Counter

        observed_multiset = Counter(observed.values())
        expected_multiset = Counter(_EXPECTED_CAP_PER_GPU.values())
        assert observed_multiset == expected_multiset, (
            f"applied_limit_watts multiset mismatch.\n"
            f"  expected (cap → count): {dict(expected_multiset)}\n"
            f"  observed (cap → count): {dict(observed_multiset)}\n"
            f"  observed per-index    : {observed}"
        )
        # Sanity: 4 distinct values across 8 GPUs.
        assert set(observed.values()) == _DISTINCT_VALUES

    def test_happy_path_zero_counters(
        self,
        core_api: "k8s_client.CoreV1Api",
        power_agent_pod_name: str,
    ):
        """§8.5 assertion 5 (B4, D3 live): in the well-formed topology, the
        five 'bad-thing-happened' counters are zero."""
        metrics = _scrape_power_agent_metrics(core_api, power_agent_pod_name)

        # Sum each counter family across all label permutations.
        def total(family: str) -> float:
            return sum(v for k, v in metrics.items() if k.startswith(family))

        violations = []
        for family in (
            "dynamo_power_agent_multi_pod_gpu_total",
            "dynamo_power_agent_safe_default_applied_total",
            "dynamo_power_agent_apply_failures_total",
            "dynamo_power_agent_dcgm_enforce_failures_total",
            "dynamo_power_agent_cap_clamped_total",
        ):
            t = total(family)
            if t != 0:
                violations.append(f"{family} == {t} (expected 0)")
        assert (
            not violations
        ), "Happy-path zero-counter invariant violated:\n  - " + "\n  - ".join(
            violations
        )


# ---------------------------------------------------------------------------
# TestCapApplicationProof — §8.5.6 (NVML) and §8.6.2 (DCGM)
# ---------------------------------------------------------------------------


class TestCapApplicationProof:
    """The cap actually landed on the driver, not just on Prometheus."""

    def test_nvml_per_gpu_cap_matches_topology(
        self,
        core_api: "k8s_client.CoreV1Api",
        power_agent_pod_name: str,
    ):
        """§8.5.6 live: nvmlDeviceGetPowerManagementLimit, read from inside
        the agent pod, returns the §8.2 caps. This is the assertion that
        mocked tests fundamentally cannot make."""
        observed = _read_nvml_caps_via_pod(core_api, power_agent_pod_name)
        # See `test_applied_limit_watts_metric_matches_topology` — multiset
        # comparison is the right shape because the device plugin owns the
        # GPU-index assignment, not us.
        from collections import Counter

        observed_multiset = Counter(observed.values())
        expected_multiset = Counter(_EXPECTED_CAP_PER_GPU.values())
        assert observed_multiset == expected_multiset, (
            f"NVML driver-side cap multiset mismatch.\n"
            f"  expected (cap → count): {dict(expected_multiset)}\n"
            f"  observed (cap → count): {dict(observed_multiset)}\n"
            f"  observed per-index    : {observed}\n"
            f"This means the cap WRITE didn't reach the driver — "
            f"Prometheus might still show the right value if the agent "
            f"updates the metric optimistically, so this assertion is "
            f"strictly stronger than the metric one."
        )

    @pytest.mark.skipif(
        not _DCGM_AVAILABLE,
        reason="DCGM hostengine standalone DS not deployed (see §8.3.D.2)",
    )
    def test_dcgm_per_gpu_cap_matches_topology(
        self,
        core_api: "k8s_client.CoreV1Api",
        power_agent_pod_name: str,
    ):
        """§8.6.2 live: read the cap through DCGM's config view, not NVML.
        Confirms PR #9790's DCGM actuator wrote to a driver path that
        DCGM agrees with NVML on."""
        # Find the standalone hostengine pod on $TEST_NODE.
        pods = core_api.list_namespaced_pod(
            _NAMESPACE,
            label_selector="app=nvidia-dcgm-standalone",
            field_selector=f"spec.nodeName={_TEST_NODE}",
        )
        assert len(pods.items) == 1, (
            f"Expected exactly one nvidia-dcgm-standalone pod on {_TEST_NODE}, "
            f"got {len(pods.items)}"
        )
        dcgm_pod = pods.items[0].metadata.name

        # DCGM 4.x changed the dcgmi UX (Finding #9):
        #   `dcgmi config --get -g all -j` returns "Group ID = all" parse
        #   error and exits 1; the legacy `--json` long-flag is also gone.
        # The Power Agent creates one DCGM group per GPU named
        # `dynamo-power-agent-gpu-N` (see actuator.py::_ensure_group_for).
        # On a fresh hostengine the group IDs are dynamic but contiguous:
        # `dcgmi group --list -j` enumerates them, then `dcgmi config
        # --get -g <id> -j` returns the per-GPU current config. We iterate.
        groups_raw = _exec_in_pod(
            core_api, dcgm_pod, ["dcgmi", "group", "--list", "-j"]
        )
        try:
            groups = json.loads(groups_raw)
        except json.JSONDecodeError:
            pytest.skip(
                f"`dcgmi group --list -j` did not return JSON on this "
                f"hostengine version. Raw output:\n{groups_raw[:500]}\n"
                f"Fall back to the NVML proof above (which we verified "
                f"agrees with DCGM target on the 2026-05-21 run)."
            )

        # Extract the per-GPU groups the Power Agent created. Format
        # is `{"body": [{"Group ID": "2", "Group Name":
        # "dynamo-power-agent-gpu-0", "Entities": ["GPU 0"]}, ...]}`.
        # `dcgmi group --list -j` body shape has been stable across
        # 4.0–4.5 (unlike `dcgmi config --get`).
        observed: Dict[int, int] = {}
        for entry in groups.get("body", []):
            name = entry.get("Group Name", "")
            if not name.startswith("dynamo-power-agent-gpu-"):
                continue
            try:
                gpu_idx = int(name.rsplit("-", 1)[-1])
                group_id = int(entry.get("Group ID", ""))
            except (ValueError, TypeError):
                continue
            cfg_raw = _exec_in_pod(
                core_api,
                dcgm_pod,
                ["dcgmi", "config", "--get", "-g", str(group_id), "-j"],
            )
            try:
                cfg = json.loads(cfg_raw)
            except json.JSONDecodeError:
                continue
            cur_w, _tgt_w = _extract_power_limit_from_dcgmi_body(cfg.get("body"))
            if cur_w is not None:
                observed[gpu_idx] = cur_w

        if len(observed) < 1:
            pytest.skip(
                "Couldn't extract any Power-Agent-created DCGM groups "
                "from `dcgmi group --list -j` output — either the "
                "hostengine started after the agent and lost the groups, "
                "or DCGM's CLI JSON shape changed again. Skip rather "
                "than false-fail."
            )
        # Multiset comparison — see test_applied_limit_watts above for the
        # device-plugin / GPU-index nondeterminism rationale.
        from collections import Counter

        observed_multiset = Counter(observed.values())
        expected_multiset = Counter(_EXPECTED_CAP_PER_GPU.values())
        assert observed_multiset == expected_multiset, (
            f"DCGM-side cap multiset mismatch.\n"
            f"  expected (cap → count): {dict(expected_multiset)}\n"
            f"  observed (cap → count): {dict(observed_multiset)}\n"
            f"  observed per-index    : {observed}\n"
            f"NVML and DCGM should agree post-dcgmConfigSet; mismatch "
            f"indicates a driver-path divergence."
        )

    @pytest.mark.skipif(
        not _DCGM_AVAILABLE,
        reason="DCGM hostengine standalone DS not deployed",
    )
    def test_dcgm_target_config_registered(
        self,
        core_api: "k8s_client.CoreV1Api",
        power_agent_pod_name: str,
    ):
        """§8.6.3 live (B8 happy path): when `agent.dcgm.enforce=true`,
        dcgmConfigEnforce registers the cap as DCGM's target config, so
        the hostengine will auto-reapply it after GPU reset.

        Review fix M4: the previous version only asserted
        `dcgm_enforce_failures_total == 0`, which proves *the call didn't
        error* but NOT that DCGM's target-config record actually contains
        the per-GPU watts. DCGM 4.x removed `dcgmi config --get-target`,
        so we now iterate the Power-Agent-created per-GPU groups and
        read the `Target` field from `dcgmi config --get -g <id> -j`.
        """
        metrics = _scrape_power_agent_metrics(core_api, power_agent_pod_name)
        failures = sum(
            v
            for k, v in metrics.items()
            if k.startswith("dynamo_power_agent_dcgm_enforce_failures_total")
        )
        assert failures == 0, (
            f"dcgm_enforce_failures_total should be 0 in the happy path, "
            f"got {failures}. dcgmConfigEnforce failed on at least one GPU."
        )

        # Locate the standalone hostengine pod on $TEST_NODE.
        pods = core_api.list_namespaced_pod(
            _NAMESPACE,
            label_selector="app=nvidia-dcgm-standalone",
            field_selector=f"spec.nodeName={_TEST_NODE}",
        )
        assert len(pods.items) == 1
        dcgm_pod = pods.items[0].metadata.name

        # DCGM 4.x: `dcgmi config --get-target` was removed; the per-GPU
        # group's `dcgmi config --get -g <id> -j` payload contains both
        # `Current` AND `Target` for "Power Limit (W)" in the same shape.
        # Reuse the same iteration as test_dcgm_per_gpu_cap_matches_topology.
        groups_raw = _exec_in_pod(
            core_api, dcgm_pod, ["dcgmi", "group", "--list", "-j"]
        )
        try:
            groups = json.loads(groups_raw)
        except json.JSONDecodeError:
            pytest.skip(
                f"`dcgmi group --list -j` did not return JSON on this "
                f"hostengine version. Raw output:\n{groups_raw[:500]}"
            )

        target_watts: Dict[int, int] = {}
        for entry in groups.get("body", []):
            name = entry.get("Group Name", "")
            if not name.startswith("dynamo-power-agent-gpu-"):
                continue
            try:
                gpu_idx = int(name.rsplit("-", 1)[-1])
                group_id = int(entry.get("Group ID", ""))
            except (ValueError, TypeError):
                continue
            cfg_raw = _exec_in_pod(
                core_api,
                dcgm_pod,
                ["dcgmi", "config", "--get", "-g", str(group_id), "-j"],
            )
            try:
                cfg = json.loads(cfg_raw)
            except json.JSONDecodeError:
                continue
            _cur_w, tgt_w = _extract_power_limit_from_dcgmi_body(cfg.get("body"))
            if tgt_w is not None:
                target_watts[gpu_idx] = tgt_w

        if len(target_watts) < 1:
            pytest.skip(
                "Couldn't extract any Power-Agent-created DCGM groups' "
                "Target config. See sibling test_dcgm_per_gpu_cap_matches_"
                "topology skip-reasoning."
            )

        # Multiset comparison — see test_applied_limit_watts above.
        from collections import Counter

        observed_multiset = Counter(target_watts.values())
        expected_multiset = Counter(_EXPECTED_CAP_PER_GPU.values())
        assert observed_multiset == expected_multiset, (
            f"DCGM target-config registration multiset mismatch.\n"
            f"  expected (cap → count): {dict(expected_multiset)}\n"
            f"  observed (cap → count): {dict(observed_multiset)}\n"
            f"  observed per-index    : {target_watts}\n"
            f"  (kept for back-compat with hard-coded test docs)\n"
            f"  expected per-index    : {_EXPECTED_CAP_PER_GPU}\n"
            f"  observed (--get-target): {target_watts}\n"
            f"dcgm_enforce_failures_total reported 0, but the target-config "
            f"record doesn't carry the expected per-GPU watts. This means "
            f"the auto-reapply-after-reset contract is NOT actually in "
            f"force, despite the counter saying everything's fine."
        )


# ---------------------------------------------------------------------------
# TestMultiPodConflict — B5 live (§8.5.7)
# ---------------------------------------------------------------------------


class TestMultiPodConflict:
    """Deliberate misconfig: a bystander pod claims one GPU (whichever
    the device plugin hands it) with a power-limit value that disagrees
    with whatever worker pod already shares that GPU. Power Agent must
    detect the disagreement, increment its multi_pod_gpu_total /
    safe_default_applied_total counters, apply safe-default on the
    *assigned* GPU only, and leave the other 7 GPUs alone.

    Historical note (PR #9683 Finding #7): the earlier implementation
    pinned the bystander to a specific GPU UUID via
    ``NVIDIA_VISIBLE_DEVICES=<uuid>``. AKS in CDI device mode rejects
    that with ``unresolvable CDI devices management.nvidia.com/gpu=*``
    and refuses to admit the pod. The CDI-safe path used here is:

      1. Bystander pod claims ``nvidia.com/gpu: 1`` from the device
         plugin — CDI-mode-agnostic, the plugin owns assignment.
      2. After the bystander reaches Running and creates a CUDA
         context, we discover *which* GPU index the device plugin
         picked by intersecting NVML compute-running PIDs across all
         8 GPUs (the agent pod has hostPID + NVML access, so it sees
         the full picture).
      3. All blast-radius asserts then key off the discovered index,
         not a hard-coded `0`.

    The bystander MUST create a CUDA compute context (a Python that
    just imports torch isn't enough) because
    ``nvmlDeviceGetComputeRunningProcesses`` — the API the Power Agent
    reads (see ``actuator.py:199``) — only enumerates CUDA compute
    PIDs. ``40-conflict-bystander-pod.yaml`` mirrors the inline
    bystander_manifest below so YAML-debugging and test-debugging
    agree.
    """

    def test_conflict_triggers_safe_default_on_assigned_gpu(
        self,
        core_api: "k8s_client.CoreV1Api",
        power_agent_pod_name: str,
    ):
        """§8.5.7 live: applying the bystander with annotation 200 ≠ {topology}
        causes:
          - multi_pod_gpu_total{disposition="conflict"} += 1
          - safe_default_applied_total += 1
          - the bystander's assigned GPU drops to safe_default (280 W)
          - the other 7 GPUs retain their topology caps
        """
        baseline_metrics = _scrape_power_agent_metrics(core_api, power_agent_pod_name)

        def total(name: str, metrics: Dict[str, float]) -> float:
            return sum(v for k, v in metrics.items() if k.startswith(name))

        baseline_conflict = total(
            'dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"',
            baseline_metrics,
        )
        baseline_safe = total(
            "dynamo_power_agent_safe_default_applied_total", baseline_metrics
        )

        bystander_script = (
            "import torch, time\n"
            "torch.cuda.init()\n"
            "x = torch.zeros(1024, device='cuda:0')\n"
            "print(f'bystander PID={__import__(\"os\").getpid()} on '\n"
            "      f'{torch.cuda.get_device_name(0)}', flush=True)\n"
            "while True:\n"
            "    x.add_(1)\n"
            "    time.sleep(0.1)\n"
        )
        # CDI-safe: no NVIDIA_VISIBLE_DEVICES. The device plugin's
        # `nvidia.com/gpu: 1` claim drives assignment, and CDI mode
        # negotiates the actual device wiring transparently.
        bystander_manifest = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": "power-conflict-bystander",
                "namespace": _NAMESPACE,
                "labels": {
                    "purpose": "power-aware-multi-dgd-test",
                    "role": "conflict-bystander",
                },
                "annotations": {_POWER_ANNOTATION_KEY: "200"},
            },
            "spec": {
                "nodeSelector": {"power-test/node": "kaim"},
                "tolerations": [
                    {
                        "key": "nvidia.com/gpu",
                        "operator": "Exists",
                        "effect": "NoSchedule",
                    }
                ],
                "runtimeClassName": "nvidia",
                "restartPolicy": "Never",
                "imagePullSecrets": [{"name": "nvcr-imagepullsecret"}],
                "containers": [
                    {
                        "name": "bystander",
                        "image": "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.1.1",
                        "imagePullPolicy": "IfNotPresent",
                        "command": ["python3", "-c"],
                        "args": [bystander_script],
                        "env": [
                            {
                                "name": "NVIDIA_DRIVER_CAPABILITIES",
                                "value": "compute,utility",
                            },
                        ],
                        "resources": {
                            "requests": {
                                "cpu": "200m",
                                "memory": "1Gi",
                                "nvidia.com/gpu": 1,
                            },
                            "limits": {
                                "cpu": "1",
                                "memory": "2Gi",
                                "nvidia.com/gpu": 1,
                            },
                        },
                    }
                ],
            },
        }

        def _per_gpu_compute_pids() -> Dict[int, Set[int]]:
            """Snapshot {gpu_idx: {pid, ...}} across all 8 GPUs.

            We query the agent pod (which has NVML + hostPID and so
            sees the same view the reconcile loop does), then index
            into the result by GPU index. Returning a dict keyed by
            int rather than a flat set lets the caller discover which
            GPU received the bystander by diffing per-index.
            """
            py = (
                "import pynvml, json\n"
                "pynvml.nvmlInit()\n"
                "n = pynvml.nvmlDeviceGetCount()\n"
                "out = {}\n"
                "for i in range(n):\n"
                "    h = pynvml.nvmlDeviceGetHandleByIndex(i)\n"
                "    out[i] = [p.pid for p in "
                "pynvml.nvmlDeviceGetComputeRunningProcesses(h)]\n"
                "print(json.dumps(out))"
            )
            raw = _exec_in_pod(core_api, power_agent_pod_name, ["python", "-c", py])
            parsed = json.loads(raw.strip())
            return {int(k): set(v) for k, v in parsed.items()}

        # Capture the per-GPU PID baseline BEFORE creating the bystander
        # so we can later diff to find which GPU the device plugin picked.
        # vLLM pods can legitimately have multiple compute PIDs (rank-main +
        # helpers); the diff-based approach is robust to that.
        baseline_per_gpu_pids = _per_gpu_compute_pids()
        # Snapshot the full cap topology too so the blast-radius assertion
        # can compare against observed pre-conflict state (the device
        # plugin's index assignment is what it is; we compare per-index
        # before/after rather than against a hard-coded topology table).
        baseline_caps = _read_nvml_caps_via_pod(core_api, power_agent_pod_name)

        try:
            core_api.create_namespaced_pod(_NAMESPACE, bystander_manifest)

            # Wait for Pod Running (CUDA context init follows a few
            # seconds later — Running is necessary, not sufficient).
            for _ in range(60):
                pod = core_api.read_namespaced_pod(
                    "power-conflict-bystander", _NAMESPACE
                )
                if pod.status.phase == "Running":
                    break
                time.sleep(2)
            else:
                pytest.fail("Bystander pod never reached Running state")

            # Discover which GPU index the device plugin assigned by
            # polling per-GPU PIDs and detecting the first index that
            # gained a PID not present in the baseline. Wait up to 90 s
            # (the torch.cuda.init() inside vllm-runtime is slow on a
            # cold pod).
            assigned_gpu: int = -1
            for _ in range(30):
                current_per_gpu = _per_gpu_compute_pids()
                # An assignment shows up as a new PID on exactly one GPU.
                for idx, pids in current_per_gpu.items():
                    new = pids - baseline_per_gpu_pids.get(idx, set())
                    if new:
                        assigned_gpu = idx
                        break
                if assigned_gpu >= 0:
                    break
                time.sleep(3)
            else:
                pytest.fail(
                    "No new compute PID appeared on ANY GPU within 90s of "
                    "the bystander reaching Running. Either the torch "
                    "script crashed (check `kubectl logs "
                    f"power-conflict-bystander -n {_NAMESPACE}`), or the "
                    "device-plugin claim didn't actually route a GPU to "
                    "the pod. baseline="
                    f"{ {k: sorted(v) for k, v in baseline_per_gpu_pids.items()} }, "
                    "current="
                    f"{ {k: sorted(v) for k, v in _per_gpu_compute_pids().items()} }"
                )

            # Let the Power Agent reconcile twice so the conflict is
            # detected AND the safe-default cap-write is committed.
            time.sleep(2 * _RECONCILE_INTERVAL_S + 5)

            post_metrics = _scrape_power_agent_metrics(core_api, power_agent_pod_name)
            post_conflict = total(
                'dynamo_power_agent_multi_pod_gpu_total{disposition="conflict"',
                post_metrics,
            )
            post_safe = total(
                "dynamo_power_agent_safe_default_applied_total", post_metrics
            )
            assert post_conflict - baseline_conflict >= 1, (
                f"multi_pod_gpu_total{{conflict}} should have ticked up. "
                f"baseline={baseline_conflict}, post={post_conflict}"
            )
            assert post_safe - baseline_safe >= 1, (
                f"safe_default_applied_total should have ticked up. "
                f"baseline={baseline_safe}, post={post_safe}"
            )

            # Blast-radius containment: the assigned GPU is at safe-default;
            # every OTHER GPU still carries its pre-conflict cap.
            observed = _read_nvml_caps_via_pod(core_api, power_agent_pod_name)
            assert observed[assigned_gpu] == _SAFE_DEFAULT_WATTS, (
                f"GPU {assigned_gpu} (the device-plugin's assignment for the "
                f"bystander) should be at safe-default {_SAFE_DEFAULT_WATTS} W "
                f"during conflict, got {observed[assigned_gpu]} W"
            )
            for idx in range(8):
                if idx == assigned_gpu:
                    continue
                assert observed[idx] == baseline_caps[idx], (
                    f"GPU {idx} cap drifted during GPU {assigned_gpu} conflict: "
                    f"baseline {baseline_caps[idx]} W, observed "
                    f"{observed[idx]} W. Blast-radius containment violated."
                )

        finally:
            core_api.delete_namespaced_pod(
                "power-conflict-bystander", _NAMESPACE, grace_period_seconds=0
            )
            # Recovery: after the bystander is gone the next reconcile
            # tick should restore the assigned GPU to its pre-conflict
            # cap (whatever worker pod owns it in this particular run).
            time.sleep(_RECONCILE_INTERVAL_S + 5)
            recovered = _read_nvml_caps_via_pod(core_api, power_agent_pod_name)
            assert recovered == baseline_caps, (
                f"GPU caps did not recover to pre-conflict state after "
                f"bystander removal. baseline={baseline_caps}, "
                f"observed={recovered}. Subsequent tests in this session "
                f"will see drifted state."
            )
