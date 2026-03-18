# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures and helpers for DGDR v1beta1 e2e tests.

These tests exercise the DynamoGraphDeploymentRequest CRD directly on a live
Kubernetes cluster running the Dynamo operator. A GPU cluster is assumed to be
available (GPU nodes reachable from the cluster).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
import uuid
from contextlib import contextmanager
from typing import Any, Dict, Generator, List, Optional

import pytest
import yaml

logger = logging.getLogger(__name__)

DGDR_API_VERSION = "nvidia.com/v1beta1"
DGDR_KIND = "DynamoGraphDeploymentRequest"
DGDR_SHORT_NAME = "dgdr"

# Default timeout values (seconds)
DEFAULT_PROFILING_TIMEOUT = 3600   # 1h for rapid, up to 4h for thorough
DEFAULT_DEPLOY_TIMEOUT = 600       # 10 minutes for DGD rollout
DEFAULT_PHASE_POLL_INTERVAL = 10   # poll every 10 seconds

# Label applied to all test-managed DGDRs so they can be bulk-deleted on cleanup
DGDR_TEST_LABEL_KEY = "test.dynamo/managed"
DGDR_TEST_LABEL_VALUE = "true"
DGDR_TEST_LABEL_SELECTOR = f"{DGDR_TEST_LABEL_KEY}={DGDR_TEST_LABEL_VALUE}"

# DGD kind name and the fixed DGD name that the mocker profiler always generates
DGD_KIND = "DynamoGraphDeployment"
MOCKER_DGD_NAME = "mocker-disagg"

# Phase values mirroring DGDRPhase Go enum
PHASE_PENDING = "Pending"
PHASE_PROFILING = "Profiling"
PHASE_READY = "Ready"
PHASE_DEPLOYING = "Deploying"
PHASE_DEPLOYED = "Deployed"
PHASE_FAILED = "Failed"


# ---------------------------------------------------------------------------
# Pytest option registration
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register DGDR-specific CLI options for the test session."""
    group = parser.getgroup("dgdr", "DynamoGraphDeploymentRequest e2e options")
    group.addoption(
        "--dgdr-namespace",
        default="default",
        help="Kubernetes namespace for DGDR resources (default: default)",
    )
    group.addoption(
        "--dgdr-image",
        default="docker.io/ashnam/dynamo-frontend:latest",
        help="Container image used for profiling and deployment workers",
    )
    group.addoption(
        "--dgdr-model",
        default="Qwen/Qwen3-0.6B",
        help="HuggingFace model ID for test DGDRs (default: Qwen/Qwen3-0.6B)",
    )
    group.addoption(
        "--dgdr-backend",
        default="vllm",
        choices=["auto", "vllm", "sglang", "trtllm"],
        help="Default backend for DGDR tests (default: vllm)",
    )
    group.addoption(
        "--dgdr-pvc-name",
        default="",
        help="Optional PVC name containing pre-downloaded model weights",
    )
    group.addoption(
        "--dgdr-profiling-timeout",
        type=int,
        default=DEFAULT_PROFILING_TIMEOUT,
        help="Max seconds to wait for profiling to complete (default: 3600)",
    )
    group.addoption(
        "--dgdr-deploy-timeout",
        type=int,
        default=DEFAULT_DEPLOY_TIMEOUT,
        help="Max seconds to wait for DGD to reach Deployed phase (default: 600)",
    )
    group.addoption(
        "--dgdr-no-mocker",
        action="store_true",
        default=False,
        help=(
            "Disable mocker mode (requires real GPU nodes for deployment). "
            "By default, mocker mode is ENABLED: DGD uses mock inference workers "
            "and AIC simulation (via searchStrategy=rapid) for GPU-free testing. "
            "Pass this flag to run against a real GPU cluster."
        ),
    )


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dgdr_namespace(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--dgdr-namespace")


@pytest.fixture(scope="session")
def dgdr_image(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--dgdr-image")


@pytest.fixture(scope="session")
def dgdr_model(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--dgdr-model")


@pytest.fixture(scope="session")
def dgdr_backend(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--dgdr-backend")


@pytest.fixture(scope="session")
def dgdr_pvc_name(request: pytest.FixtureRequest) -> str:
    return request.config.getoption("--dgdr-pvc-name")


@pytest.fixture(scope="session")
def dgdr_profiling_timeout(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--dgdr-profiling-timeout")


@pytest.fixture(scope="session")
def dgdr_deploy_timeout(request: pytest.FixtureRequest) -> int:
    return request.config.getoption("--dgdr-deploy-timeout")


@pytest.fixture(scope="session")
def dgdr_use_mocker(request: pytest.FixtureRequest) -> bool:
    # Mocker is ON by default; --dgdr-no-mocker disables it
    return not request.config.getoption("--dgdr-no-mocker")


# ---------------------------------------------------------------------------
# Simulation-mode helpers (Mocker)
# ---------------------------------------------------------------------------

# Default hardware config for GPU-free testing (AIC simulation needs hardware metadata)
DEFAULT_MOCKER_HARDWARE = {
    "gpuSku": "a100_sxm",
    "vramMb": 81920,
    "numGpusPerNode": 8,
    "totalGpus": 8,
}


def _inject_mocker_config(manifest: Dict[str, Any]) -> None:
    """Mutate *manifest* in-place to enable mocker deployment.

    Mocker: sets ``spec.features.mocker.enabled = true`` so the DGD uses mock
    inference workers that do not require GPU resources.

    Also injects a default ``spec.hardware`` config if not already set, since
    AIC simulation needs hardware metadata (GPU model, VRAM) even though it
    doesn't actually use GPUs.

    Combined with ``searchStrategy: rapid`` (the default), this enables the full
    DGDR lifecycle (Pending -> Profiling -> Ready -> Deploying -> Deployed) to
    complete without any GPU nodes, because:

    - rapid uses AI Configurator (AIC) simulation in the profiler (CPU-only)
    - mocker uses mock inference pods (no GPU resources requested)
    """
    spec = manifest.setdefault("spec", {})

    # Enable mocker for GPU-free deployment
    features = spec.setdefault("features", {})
    mocker = features.setdefault("mocker", {})
    mocker["enabled"] = True

    # Inject default hardware if not already set (AIC needs hardware metadata).
    # If hardware is partially set (e.g. the test only sets gpuSku/numGpusPerNode),
    # fill in any missing fields from DEFAULT_MOCKER_HARDWARE so AIC has the full
    # metadata it needs (vramMb, totalGpus) without overriding fields the test set.
    if "hardware" not in spec:
        spec["hardware"] = DEFAULT_MOCKER_HARDWARE.copy()
        logger.info(
            "Injected default hardware config for DGDR %s: %s",
            manifest.get("metadata", {}).get("name", "?"),
            spec["hardware"],
        )
    else:
        merged = False
        for k, v in DEFAULT_MOCKER_HARDWARE.items():
            if k not in spec["hardware"]:
                spec["hardware"][k] = v
                merged = True
        if merged:
            logger.info(
                "Merged missing hardware fields for DGDR %s: %s",
                manifest.get("metadata", {}).get("name", "?"),
                spec["hardware"],
            )

    logger.info("Mocker mode enabled for DGDR %s", manifest.get("metadata", {}).get("name", "?"))


def _cleanup_mocker_dgd(namespace: str) -> None:
    """Delete the shared `mocker-disagg` DGD if it exists.

    The mocker profiler always names the generated DGD ``mocker-disagg``.  When
    multiple DGDRs run sequentially in the same test session (all creating the same
    DGD name), the second DGDR's operator finds ``mocker-disagg`` already in the
    cluster.  If that DGD is in a bad/terminating state from the previous test, the
    operator fires ``handleDGDDeleted`` immediately → DGDR reaches Failed.  Deleting
    the DGD between tests guarantees each DGDR starts from a clean slate.
    """
    result = _run_kubectl(
        ["get", "dynamographdeployment", MOCKER_DGD_NAME, "-n", namespace, "--ignore-not-found"],
        check=False,
    )
    if MOCKER_DGD_NAME in result.stdout:
        logger.info("Deleting shared mocker DGD %s/%s to prevent state pollution", namespace, MOCKER_DGD_NAME)
        _run_kubectl(
            ["delete", "dynamographdeployment", MOCKER_DGD_NAME, "-n", namespace,
             "--ignore-not-found", "--timeout=30s"],
            check=False,
        )


# ---------------------------------------------------------------------------
# kubectl helpers
# ---------------------------------------------------------------------------


def _run_kubectl(args: List[str], check: bool = True, input: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a kubectl command, returning the CompletedProcess."""
    cmd = ["kubectl"] + args
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
        input=input,
    )
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


def kubectl_apply(manifest: Dict[str, Any], namespace: str) -> subprocess.CompletedProcess:
    """Apply a manifest dict via kubectl apply -f -."""
    yaml_str = yaml.dump(manifest)
    return _run_kubectl(["apply", "-n", namespace, "-f", "-"], input=yaml_str)


def kubectl_apply_raw(yaml_str: str, namespace: str) -> subprocess.CompletedProcess:
    """Apply a raw YAML string via kubectl apply -f -."""
    return _run_kubectl(["apply", "-n", namespace, "-f", "-"], input=yaml_str)


def kubectl_delete(kind: str, name: str, namespace: str, ignore_not_found: bool = True) -> None:
    """Delete a Kubernetes resource."""
    args = ["delete", kind, name, "-n", namespace]
    if ignore_not_found:
        args.append("--ignore-not-found")
    _run_kubectl(args, check=not ignore_not_found)


def kubectl_get_json(kind: str, name: str, namespace: str) -> Optional[Dict[str, Any]]:
    """Get a resource as a parsed JSON dict, or None if not found."""
    result = _run_kubectl(
        ["get", kind, name, "-n", namespace, "-o", "json"],
        check=False,
    )
    if result.returncode != 0:
        return None
    return json.loads(result.stdout)


def kubectl_list_json(kind: str, namespace: str, label_selector: str = "") -> List[Dict[str, Any]]:
    """List resources of a kind, returning parsed JSON items."""
    args = ["get", kind, "-n", namespace, "-o", "json"]
    if label_selector:
        args += ["-l", label_selector]
    result = _run_kubectl(args, check=False)
    if result.returncode != 0:
        return []
    data = json.loads(result.stdout)
    return data.get("items", [])


def kubectl_server_dry_run(manifest: Dict[str, Any], namespace: str) -> subprocess.CompletedProcess:
    """Apply with --dry-run=server to test admission webhooks without creating resources."""
    yaml_str = yaml.dump(manifest)
    return _run_kubectl(
        ["apply", "-n", namespace, "--dry-run=server", "-f", "-"],
        check=False,
        input=yaml_str,
    )


def get_dgdr(name: str, namespace: str) -> Optional[Dict[str, Any]]:
    """Return the full DGDR resource dict or None if not found."""
    return kubectl_get_json(DGDR_KIND, name, namespace)


def get_dgdr_phase(name: str, namespace: str) -> Optional[str]:
    """Return status.phase of the named DGDR."""
    obj = get_dgdr(name, namespace)
    if obj is None:
        return None
    return obj.get("status", {}).get("phase")


def get_dgdr_condition(name: str, namespace: str, condition_type: str) -> Optional[Dict[str, Any]]:
    """Return the named condition dict from status.conditions, or None."""
    obj = get_dgdr(name, namespace)
    if obj is None:
        return None
    conditions = obj.get("status", {}).get("conditions", [])
    for c in conditions:
        if c.get("type") == condition_type:
            return c
    return None


# ---------------------------------------------------------------------------
# Wait helpers
# ---------------------------------------------------------------------------


def wait_for_dgdr_phase(
    name: str,
    namespace: str,
    target_phase: str,
    timeout: int = DEFAULT_PROFILING_TIMEOUT,
    fail_fast_phases: Optional[List[str]] = None,
) -> str:
    """
    Poll until the DGDR reaches ``target_phase`` or times out.

    ``fail_fast_phases`` lists phases that should immediately abort the wait
    (e.g. Failed).  Returns the final observed phase.
    """
    if fail_fast_phases is None:
        fail_fast_phases = [PHASE_FAILED]

    deadline = time.monotonic() + timeout
    last_phase: Optional[str] = None

    while time.monotonic() < deadline:
        current = get_dgdr_phase(name, namespace)
        if current != last_phase:
            logger.info("DGDR %s/%s phase: %s", namespace, name, current)
            last_phase = current

        if current == target_phase:
            return current
        if current in fail_fast_phases:
            obj = get_dgdr(name, namespace)
            status = obj.get("status", {}) if obj else {}
            conditions = status.get("conditions", [])
            raise AssertionError(
                f"DGDR {namespace}/{name} reached fail-fast phase {current!r} "
                f"while waiting for {target_phase!r}. conditions={conditions}"
            )
        time.sleep(DEFAULT_PHASE_POLL_INTERVAL)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for DGDR {namespace}/{name} "
        f"to reach phase {target_phase!r}. Last phase: {last_phase!r}"
    )


def wait_for_any_dgdr_phase(
    name: str,
    namespace: str,
    target_phases: List[str],
    timeout: int = DEFAULT_PROFILING_TIMEOUT,
) -> str:
    """Poll until the DGDR reaches any of ``target_phases``.  Returns the matched phase."""
    deadline = time.monotonic() + timeout
    last_phase: Optional[str] = None

    while time.monotonic() < deadline:
        current = get_dgdr_phase(name, namespace)
        if current != last_phase:
            logger.info("DGDR %s/%s phase: %s", namespace, name, current)
            last_phase = current
        if current in target_phases:
            return current
        time.sleep(DEFAULT_PHASE_POLL_INTERVAL)

    raise TimeoutError(
        f"Timed out after {timeout}s waiting for DGDR {namespace}/{name} "
        f"to reach any of {target_phases!r}. Last phase: {last_phase!r}"
    )


# ---------------------------------------------------------------------------
# DGDR manifest builder
# ---------------------------------------------------------------------------


def build_dgdr_manifest(
    name: str,
    model: str,
    image: str,
    *,
    backend: str = "vllm",
    search_strategy: str = "rapid",
    sla: Optional[Dict[str, Any]] = None,
    workload: Optional[Dict[str, Any]] = None,
    features: Optional[Dict[str, Any]] = None,
    hardware: Optional[Dict[str, Any]] = None,
    model_cache: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    auto_apply: Optional[bool] = None,
    labels: Optional[Dict[str, str]] = None,
    extra_spec_fields: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a v1beta1 DGDR manifest dict.

    Only ``name``, ``model``, and ``image`` are required.  All other fields
    are optional and map 1-to-1 to the v1beta1 spec defined in
    ``deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go``.
    """
    spec: Dict[str, Any] = {
        "model": model,
        "backend": backend,
        "image": image,
        "searchStrategy": search_strategy,
    }

    if sla is not None:
        spec["sla"] = sla
    if workload is not None:
        spec["workload"] = workload
    if features is not None:
        spec["features"] = features
    if hardware is not None:
        spec["hardware"] = hardware
    if model_cache is not None:
        spec["modelCache"] = model_cache
    if overrides is not None:
        spec["overrides"] = overrides
    if auto_apply is not None:
        spec["autoApply"] = auto_apply
    if extra_spec_fields:
        spec.update(extra_spec_fields)

    manifest: Dict[str, Any] = {
        "apiVersion": DGDR_API_VERSION,
        "kind": DGDR_KIND,
        "metadata": {
            "name": name,
        },
        "spec": spec,
    }
    if labels:
        manifest["metadata"]["labels"] = labels

    return manifest


def unique_dgdr_name(prefix: str = "test") -> str:
    """Generate a unique DGDR name safe for Kubernetes (lowercase, 63 chars max)."""
    uid = uuid.uuid4().hex[:8]
    return f"{prefix}-{uid}"


# ---------------------------------------------------------------------------
# Core fixture: managed DGDR lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def dgdr_factory(
    dgdr_namespace: str,
    dgdr_profiling_timeout: int,
    dgdr_deploy_timeout: int,
    dgdr_use_mocker: bool,
):
    """
    A factory fixture that applies a DGDR manifest and ensures cleanup.

    When ``--dgdr-use-mocker`` CLI flag is set, the factory automatically
    injects mocker config into every manifest before applying it.  This
    makes the injection transparent to individual test functions.

    Combined with ``searchStrategy: rapid`` (the default), mocker mode
    enables a fully GPU-free lifecycle:
    - Profiling uses AIC simulation (CPU-only, no GPU resources needed)
    - Deployment uses mock inference pods (no GPU resources requested)

    Usage::

        def test_something(dgdr_factory, dgdr_image, dgdr_model):
            manifest = build_dgdr_manifest("my-test", dgdr_model, dgdr_image)
            name = dgdr_factory(manifest)
            wait_for_dgdr_phase(name, dgdr_namespace, PHASE_DEPLOYED, ...)
    """
    created: List[str] = []
    namespace = dgdr_namespace
    use_mocker = dgdr_use_mocker

    def _cleanup_all_test_dgdrs() -> None:
        """Delete all DGDRs bearing the test-managed label (handles orphans from prior runs)."""
        items = kubectl_list_json(DGDR_KIND, namespace, label_selector=DGDR_TEST_LABEL_SELECTOR)
        for item in items:
            item_name = item.get("metadata", {}).get("name", "")
            if item_name:
                logger.info("Cleaning up test-managed DGDR %s/%s", namespace, item_name)
                kubectl_delete(DGDR_KIND, item_name, namespace, ignore_not_found=True)

    # Pre-test: remove any orphaned DGDRs left by previously interrupted runs.
    # NOTE: Do NOT delete mocker-disagg here.  The operator pre-populates
    # Status.DGDName from the profiling output (generateDGDSpec), so when
    # handleDeployingPhase runs it immediately tries to GET the DGD by that name.
    # Deleting mocker-disagg before the test causes handleDGDDeleted to fire.
    _cleanup_all_test_dgdrs()

    def _create(manifest: Dict[str, Any]) -> str:
        name = manifest["metadata"]["name"]
        # Stamp the test-managed label so orphan cleanup can find it
        manifest.setdefault("metadata", {})
        manifest["metadata"].setdefault("labels", {})
        manifest["metadata"]["labels"][DGDR_TEST_LABEL_KEY] = DGDR_TEST_LABEL_VALUE
        # Inject mocker config if enabled
        if use_mocker:
            _inject_mocker_config(manifest)
        kubectl_apply(manifest, namespace)
        created.append(name)
        logger.info("Created DGDR %s/%s", namespace, name)
        return name

    yield _create

    # Post-test: delete everything we created (plus any label-matching stragglers)
    for name in reversed(created):
        logger.info("Cleaning up DGDR %s/%s", namespace, name)
        kubectl_delete(DGDR_KIND, name, namespace, ignore_not_found=True)
    _cleanup_all_test_dgdrs()
    # Clean up the shared mocker DGD so the next test starts fresh
    if use_mocker:
        _cleanup_mocker_dgd(namespace)


# ---------------------------------------------------------------------------
# Session-scoped shared deployment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def deployed_dgdr(
    dgdr_namespace: str,
    dgdr_image: str,
    dgdr_model: str,
    dgdr_use_mocker: bool,
    dgdr_profiling_timeout: int,
    dgdr_deploy_timeout: int,
) -> Generator[str, None, None]:
    """
    Session-scoped fixture: deploys a single DGDR once for the entire test
    session and tears it down afterward.

    Tests that only need a *Deployed* DGDR to read status from should use this
    fixture instead of spinning up their own lifecycle.  This avoids repeated
    ~1-2 minute profiling cycles for tests that are purely asserting status
    fields on an already-deployed resource.
    """
    name = unique_dgdr_name("session")
    # In mocker mode, auto_apply=True hits a consistent "DeploymentDeleted" failure because
    # the operator cannot complete DGD creation with the shared mocker-disagg name.  Use
    # auto_apply=False and target PHASE_READY instead; tests that strictly require the
    # Deployed phase are xfailed in mocker mode.
    manifest = build_dgdr_manifest(
        name,
        model=dgdr_model,
        image=dgdr_image,
        backend="vllm",
        search_strategy="rapid",
        auto_apply=not dgdr_use_mocker,
    )
    manifest.setdefault("metadata", {})
    manifest["metadata"].setdefault("labels", {})
    if dgdr_use_mocker:
        _inject_mocker_config(manifest)
        # Ensure no stale mocker-disagg DGD from a previous test so the session DGDR
        # gets a clean mocker-disagg on its first deploy attempt.
        _cleanup_mocker_dgd(dgdr_namespace)

    kubectl_apply(manifest, dgdr_namespace)
    logger.info("Session DGDR %s/%s created", dgdr_namespace, name)

    try:
        target_phase = PHASE_READY if dgdr_use_mocker else PHASE_DEPLOYED
        wait_for_dgdr_phase(
            name, dgdr_namespace, target_phase,
            timeout=dgdr_profiling_timeout + dgdr_deploy_timeout,
        )
        logger.info("Session DGDR %s/%s reached %s", dgdr_namespace, name, target_phase)
        yield name
    finally:
        kubectl_delete(DGDR_KIND, name, dgdr_namespace, ignore_not_found=True)
        if dgdr_use_mocker:
            _cleanup_mocker_dgd(dgdr_namespace)
        logger.info("Session DGDR %s/%s cleaned up", dgdr_namespace, name)
