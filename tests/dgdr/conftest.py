# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest fixtures and helpers for DGDR v1beta1 e2e tests.

These tests exercise the DynamoGraphDeploymentRequest CRD directly on a live
Kubernetes cluster running the Dynamo operator. A GPU cluster is assumed to be
available (GPU nodes reachable from the cluster).
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
import uuid
from typing import Any, Dict, Generator, List, Optional

import pytest

from tests.utils.managed_deployment import ManagedDGDR

logger = logging.getLogger(__name__)

DGDR_API_VERSION = "nvidia.com/v1beta1"
DGDR_KIND = "DynamoGraphDeploymentRequest"
DGDR_SHORT_NAME = "dgdr"

# Default timeout values (seconds)
DEFAULT_PROFILING_TIMEOUT = 3600  # 1h for rapid, up to 4h for thorough
DEFAULT_DEPLOY_TIMEOUT = 600  # 10 minutes for DGD rollout

# Label applied to all test-managed DGDRs so they can be bulk-deleted on cleanup
DGDR_TEST_LABEL_KEY = "test.dynamo/managed"

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
        default=None,
        help="Kubernetes namespace for DGDR resources (required to run tests)",
    )
    group.addoption(
        "--dgdr-image",
        default=None,
        help="Container image used for profiling and deployment workers (required to run tests)",
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
# Skip DGDR tests gracefully when required CLI args are not provided
# (e.g. when the whole test suite is run by CI without --dgdr-* flags)
# ---------------------------------------------------------------------------


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip all DGDR tests when --dgdr-namespace or --dgdr-image are not supplied.

    This prevents a session-aborting failure when the global CI runner collects
    and executes ``tests/`` without passing the DGDR-specific CLI options.
    """
    missing = []
    if not config.getoption("--dgdr-namespace", default=None):
        missing.append("--dgdr-namespace")
    if not config.getoption("--dgdr-image", default=None):
        missing.append("--dgdr-image")
    if not missing:
        return
    reason = f"DGDR tests require: {', '.join(missing)}"
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "dgdr" in str(item.fspath):
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def dgdr_namespace(request: pytest.FixtureRequest) -> str:
    value = request.config.getoption("--dgdr-namespace")
    if not value:
        pytest.skip("--dgdr-namespace is required to run DGDR tests")
    return value


@pytest.fixture(scope="session")
def dgdr_image(request: pytest.FixtureRequest) -> str:
    value = request.config.getoption("--dgdr-image")
    if not value:
        pytest.skip("--dgdr-image is required to run DGDR tests")
    return value


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
# Session-scoped ManagedDGDR client
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def _dgdr_event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Session-scoped event loop shared by all DGDR async helpers."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def dgdr_test_label() -> str:
    """Generate a unique label value for this test session to avoid cross-run cleanup races."""
    return f"test-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="session")
def dgdr_test_label_selector(dgdr_test_label: str) -> str:
    return f"{DGDR_TEST_LABEL_KEY}={dgdr_test_label}"


@pytest.fixture(scope="session")
def managed_dgdr(
    dgdr_namespace: str, _dgdr_event_loop: asyncio.AbstractEventLoop
) -> Generator[ManagedDGDR, None, None]:
    """Session-scoped async K8s client for DGDR operations."""
    mgr = ManagedDGDR(namespace=dgdr_namespace, loop=_dgdr_event_loop)
    mgr.run(mgr.init())
    yield mgr
    mgr.run(mgr.close())


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

    logger.info(
        "Mocker mode enabled for DGDR %s", manifest.get("metadata", {}).get("name", "?")
    )


async def _cleanup_mocker_dgd(mgr: ManagedDGDR) -> None:
    """Delete the shared `mocker-disagg` DGD if it exists.

    The mocker profiler always names the generated DGD ``mocker-disagg``.  When
    multiple DGDRs run sequentially in the same test session (all creating the same
    DGD name), the second DGDR's operator finds ``mocker-disagg`` already in the
    cluster.  If that DGD is in a bad/terminating state from the previous test, the
    operator fires ``handleDGDDeleted`` immediately → DGDR reaches Failed.  Deleting
    the DGD between tests guarantees each DGDR starts from a clean slate.
    """
    obj = await mgr.get_dgd(MOCKER_DGD_NAME)
    if obj is not None:
        logger.info(
            "Deleting shared mocker DGD %s/%s to prevent state pollution",
            mgr.namespace,
            MOCKER_DGD_NAME,
        )
        await mgr.delete_dgd(MOCKER_DGD_NAME)


# ---------------------------------------------------------------------------
# kubectl helper (kept only for tests that validate CLI behaviour itself)
# ---------------------------------------------------------------------------


def _run_kubectl(
    args: List[str], check: bool = True, input: Optional[str] = None
) -> subprocess.CompletedProcess:
    """Run a kubectl command, returning the CompletedProcess.

    This is intentionally kept for the small number of tests that validate
    kubectl CLI behaviour (short-names, custom-columns, CRD metadata).
    All DGDR CRUD and phase-polling should use :class:`ManagedDGDR` instead.
    """
    cmd = ["kubectl"] + args
    logger.debug("Running: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            input=input,
            timeout=60,
        )
    except subprocess.TimeoutExpired as e:
        logger.error("kubectl timed out: %s", e)
        pytest.fail(f"kubectl timed out after 60s: {e}")
    if check and result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode, cmd, result.stdout, result.stderr
        )
    return result


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
    managed_dgdr: ManagedDGDR,
    dgdr_namespace: str,
    dgdr_profiling_timeout: int,
    dgdr_deploy_timeout: int,
    dgdr_use_mocker: bool,
    dgdr_test_label: str,
    dgdr_test_label_selector: str,
):
    """
    A factory fixture that applies a DGDR manifest and ensures cleanup.

    When mocker mode is enabled (the default), the factory automatically
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
            managed_dgdr.run(managed_dgdr.wait_for_phase(name, PHASE_DEPLOYED, ...))
    """
    created: List[str] = []
    use_mocker = dgdr_use_mocker

    def _cleanup_all_test_dgdrs() -> None:
        """Delete all DGDRs bearing the test-managed label (handles orphans from prior runs)."""
        items = managed_dgdr.run(
            managed_dgdr.list(label_selector=dgdr_test_label_selector)
        )
        for item in items:
            item_name = item.get("metadata", {}).get("name", "")
            if item_name:
                logger.info(
                    "Cleaning up test-managed DGDR %s/%s", dgdr_namespace, item_name
                )
                managed_dgdr.run(managed_dgdr.delete(item_name))

    # Pre-test: remove any orphaned DGDRs left by previously interrupted runs.
    _cleanup_all_test_dgdrs()

    def _create(manifest: Dict[str, Any]) -> str:
        name = manifest["metadata"]["name"]
        # Stamp the test-managed label so orphan cleanup can find it
        manifest.setdefault("metadata", {})
        manifest["metadata"].setdefault("labels", {})
        manifest["metadata"]["labels"][DGDR_TEST_LABEL_KEY] = dgdr_test_label
        # Inject mocker config if enabled
        if use_mocker:
            _inject_mocker_config(manifest)
        managed_dgdr.run(managed_dgdr.create(manifest))
        created.append(name)
        logger.info("Created DGDR %s/%s", dgdr_namespace, name)
        return name

    def _register_for_cleanup(name: str) -> None:
        """Register an externally-created DGDR name for teardown cleanup."""
        if name not in created:
            created.append(name)

    _create.register_for_cleanup = _register_for_cleanup  # type: ignore[attr-defined]

    yield _create

    # Post-test: delete everything we created (plus any label-matching stragglers)
    for name in reversed(created):
        logger.info("Cleaning up DGDR %s/%s", dgdr_namespace, name)
        managed_dgdr.run(managed_dgdr.delete(name))
    _cleanup_all_test_dgdrs()
    # Clean up the shared mocker DGD so the next test starts fresh
    if use_mocker:
        managed_dgdr.run(_cleanup_mocker_dgd(managed_dgdr))


# ---------------------------------------------------------------------------
# Session-scoped shared deployment
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def deployed_dgdr(
    managed_dgdr: ManagedDGDR,
    dgdr_namespace: str,
    dgdr_image: str,
    dgdr_model: str,
    dgdr_use_mocker: bool,
    dgdr_profiling_timeout: int,
    dgdr_deploy_timeout: int,
    dgdr_test_label: str,
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
    # Stamp session DGDR with the per-session test label so it is cleaned up
    # together with other test-managed resources.
    manifest["metadata"]["labels"][DGDR_TEST_LABEL_KEY] = dgdr_test_label
    if dgdr_use_mocker:
        _inject_mocker_config(manifest)
        # Ensure no stale mocker-disagg DGD from a previous test so the session DGDR
        # gets a clean mocker-disagg on its first deploy attempt.
        managed_dgdr.run(_cleanup_mocker_dgd(managed_dgdr))

    managed_dgdr.run(managed_dgdr.create(manifest))
    logger.info("Session DGDR %s/%s created", dgdr_namespace, name)

    try:
        target_phase = PHASE_READY if dgdr_use_mocker else PHASE_DEPLOYED
        managed_dgdr.run(
            managed_dgdr.wait_for_phase(
                name,
                target_phase,
                timeout=dgdr_profiling_timeout + dgdr_deploy_timeout,
            )
        )
        logger.info("Session DGDR %s/%s reached %s", dgdr_namespace, name, target_phase)
        yield name
    finally:
        managed_dgdr.run(managed_dgdr.delete(name))
        if dgdr_use_mocker:
            managed_dgdr.run(_cleanup_mocker_dgd(managed_dgdr))
        logger.info("Session DGDR %s/%s cleaned up", dgdr_namespace, name)
