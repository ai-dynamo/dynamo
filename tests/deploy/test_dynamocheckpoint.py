# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster DGD checkpoint/restore deploy test."""

import asyncio
import copy
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import aiohttp
import pytest
import requests
from kubernetes_asyncio.client import exceptions as k8s_exceptions

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment, _get_workspace_dir
from tests.utils.client import send_request, wait_for_model_availability

logger = logging.getLogger(__name__)

# kr8s port-forward teardown runs in background threads; on pod termination it
# can surface expected OSErrors (e.g. EADDRINUSE for a local port still in
# TIME_WAIT) via threading.excepthook. Under filterwarnings=error those would
# fail this live-cluster test, so scope the suppression to this module only
# rather than globally hiding unrelated background-thread crashes.
pytestmark = pytest.mark.filterwarnings(
    "ignore::pytest.PytestUnhandledThreadExceptionWarning"
)

TRANSIENT_K8S_EXCEPTIONS = (
    aiohttp.ClientError,
    asyncio.TimeoutError,
    k8s_exceptions.ApiException,
)

DGD_PLURAL = "dynamographdeployments"
POD_SNAPSHOT_PLURAL = "podsnapshots"
SNAPSHOT_JOB_PLURAL = "snapshotjobs"

FRONTEND_COMPONENT = "Frontend"
TARGET_CONTAINER = "main"
CHECKPOINT_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT_STORAGE_MOUNT_PATH = "/checkpoints"
TRTLLM_HF_HOME = f"{CHECKPOINT_STORAGE_MOUNT_PATH}/trtllm-hf-cache"

SNAPSHOT_JOB_OWNER_LABEL = "nvidia.com/snapshot-job"
RESTORE_FROM_ANNOTATION = "nvidia.com/restore-from"
RESTORED_CONDITION = "nvidia.com/Restored"
RESTORE_FAILURE_REASONS = frozenset({"RestoreFailed", "RestorePartiallySucceeded"})

# CUDA checkpointing can OOM on 10GB MIG slices; run this test on full GPUs.
GPU_NODE_SELECTOR = {
    "nvidia.com/gpu.present": "true",
    "nvidia.com/mig.config": "all-disabled",
}
GPU_TOLERATIONS = [
    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
    {"key": "dedicated", "operator": "Exists", "effect": "NoSchedule"},
]

TEST_PROMPT = "Reply with one short sentence confirming this restored worker can serve."
DEFAULT_MAX_TOKENS = 24
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = 120
MODEL_READY_MAX_ATTEMPTS = 6
MODEL_READY_ATTEMPT_TIMEOUTS = [60.0, 30.0, 10.0, 10.0, 10.0, 10.0]
CHECKPOINT_READY_TIMEOUT = 300
RESTORE_READY_TIMEOUT = 300
DECODE_SCALE_TIMEOUT = 60
RESTORED_DEPLOYMENT_READY_TIMEOUT = 180
# Phase caps are deliberately non-additive: successful phases normally finish
# well below their individual ceilings. TEST_TIMEOUT is the global 28-minute
# budget and leaves two minutes beneath the workflow limit for final cleanup.
DEPLOYMENT_READY_TIMEOUT = 900
IMMEDIATE_DEPLOYMENT_READY_TIMEOUT = 600
TEST_TIMEOUT = 1680


@dataclass(frozen=True)
class CheckpointBackendConfig:
    name: str
    manifest: tuple[str, ...]
    decode_component: str
    frontend_component: str
    target_container: str
    model: str
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    extra_volumes: tuple[dict[str, Any], ...] = ()
    extra_volume_mounts: tuple[dict[str, Any], ...] = ()
    pod_spec_updates: dict[str, Any] | None = None
    container_resources: dict[str, Any] | None = None
    checkpoint_startup_policy: str | None = None


CHECKPOINT_BACKENDS = {
    # Exercise the default Immediate policy end to end. The stable automatic
    # restore-candidate metadata prevents capture readiness from rolling the
    # initial worker.
    "vllm": CheckpointBackendConfig(
        name="vllm",
        manifest=("examples", "backends", "vllm", "deploy", "agg.yaml"),
        decode_component="VllmDecodeWorker",
        frontend_component=FRONTEND_COMPONENT,
        target_container=TARGET_CONTAINER,
        model=CHECKPOINT_MODEL,
        args=(
            "--model",
            CHECKPOINT_MODEL,
            "--max-model-len",
            "2048",
            "--gpu-memory-utilization",
            "0.30",
        ),
    ),
    "sglang": CheckpointBackendConfig(
        name="sglang",
        manifest=(
            "examples",
            "backends",
            "sglang",
            "deploy",
            "agg.yaml",
        ),
        decode_component="decode",
        frontend_component=FRONTEND_COMPONENT,
        target_container=TARGET_CONTAINER,
        model=CHECKPOINT_MODEL,
        args=(
            "--model-path",
            CHECKPOINT_MODEL,
            "--served-model-name",
            CHECKPOINT_MODEL,
            "--page-size",
            "16",
            "--tp",
            "1",
            "--trust-remote-code",
            "--skip-tokenizer-init",
        ),
        # Keep SGLang capture and restore sequential on one GPU. vLLM exercises
        # the Immediate policy in this shared test.
        checkpoint_startup_policy="WaitForCheckpoint",
    ),
    "trtllm": CheckpointBackendConfig(
        name="trtllm",
        manifest=(
            "examples",
            "backends",
            "trtllm",
            "deploy",
            "agg.yaml",
        ),
        decode_component="TRTLLMWorker",
        frontend_component=FRONTEND_COMPONENT,
        target_container=TARGET_CONTAINER,
        model=CHECKPOINT_MODEL,
        # Only the CI-sizing overrides that differ from TensorRT-LLM defaults
        # are passed. The remaining single-GPU snapshot settings from
        # examples/backends/trtllm/engine_configs/qwen3/snapshot.yaml are already
        # defaults or no-ops for dense Qwen3-0.6B: tensor/pipeline parallel = 1,
        # no expert parallel or attention DP, pytorch backend (forced by the
        # worker), and chunked prefill (inert at max-batch-size 1).
        args=(
            "--model-path",
            CHECKPOINT_MODEL,
            "--served-model-name",
            CHECKPOINT_MODEL,
            "--max-num-tokens",
            "1024",
            "--max-batch-size",
            "1",
            "--free-gpu-memory-fraction",
            "0.10",
        ),
        # UCX_TLS is always set. HF_HOME defaults to the snapshot PVC so restore
        # pods keep weights without a model-cache PVC; when CI passes
        # --model-cache-pvc, _new_checkpoint_spec skips this HF_HOME so the
        # shared cache mount can own it (same as regular deploy tests).
        env=(("UCX_TLS", "tcp,self"), ("HF_HOME", TRTLLM_HF_HOME)),
        # Match the base TRTLLM snapshot recipe and avoid cold-worker/restore
        # rollout overlap during initial DGD startup.
        checkpoint_startup_policy="WaitForCheckpoint",
        pod_spec_updates={
            "runtimeClassName": "nvidia",
            "securityContext": {
                "fsGroup": 1000,
                "fsGroupChangePolicy": "OnRootMismatch",
            },
        },
        container_resources={
            "requests": {
                "cpu": "4",
                "memory": "16Gi",
                "nvidia.com/gpu": "1",
                "ephemeral-storage": "10Gi",
            },
            "limits": {
                "cpu": "8",
                "memory": "32Gi",
                "nvidia.com/gpu": "1",
            },
        },
        extra_volumes=(
            {"name": "criu-work", "emptyDir": {}},
            {
                "name": "dev-net-tun",
                "hostPath": {"path": "/dev/net/tun", "type": "CharDevice"},
            },
        ),
        extra_volume_mounts=(
            {"name": "criu-work", "mountPath": "/var/criu-work"},
            {"name": "dev-net-tun", "mountPath": "/dev/net/tun"},
        ),
    ),
}


def _checkpoint_backend(request: pytest.FixtureRequest) -> CheckpointBackendConfig:
    backend_name = request.config.getoption("--checkpoint-backend")
    try:
        return CHECKPOINT_BACKENDS[backend_name]
    except KeyError as exc:
        raise AssertionError(
            f"unsupported checkpoint backend {backend_name!r}"
        ) from exc


def _component(spec: dict[str, Any], name: str) -> dict[str, Any]:
    for component in spec["spec"].get("components", []):
        if component.get("name") == name:
            return component
    raise AssertionError(f"component {name!r} not found in DGD spec")


def _new_checkpoint_spec(
    backend: CheckpointBackendConfig,
    name: str,
    namespace: str,
    image: str,
    frontend_image: str,
    *,
    model_cache_pvc: str | None = None,
    model_cache_mount: str | None = None,
) -> DeploymentSpec:
    spec_path = Path(_get_workspace_dir()).joinpath(*backend.manifest)
    deployment_spec = DeploymentSpec(str(spec_path))
    deployment_spec.name = name
    deployment_spec.namespace = namespace
    deployment_spec.set_image(frontend_image, backend.frontend_component)
    deployment_spec.set_image(image, backend.decode_component)
    deployment_spec.set_model(backend.model, backend.decode_component)

    raw_spec = deployment_spec.spec()
    decode = _component(raw_spec, backend.decode_component)
    pod_spec = decode.setdefault("podTemplate", {}).setdefault("spec", {})
    containers = pod_spec.setdefault("containers", [])
    if not containers:
        raise AssertionError(
            f"component {backend.decode_component!r} has no containers"
        )
    pod_spec["nodeSelector"] = dict(GPU_NODE_SELECTOR)
    pod_spec["tolerations"] = list(GPU_TOLERATIONS)
    if backend.pod_spec_updates:
        pod_spec.update(copy.deepcopy(backend.pod_spec_updates))
    container = containers[0]
    container["args"] = list(backend.args)
    if backend.container_resources:
        container["resources"] = copy.deepcopy(backend.container_resources)

    # vCluster adds host-side resources that make the physical Pod Burstable.
    # Keep the virtual Pod in the same QoS class so status synchronization can
    # propagate Snapshot's restore condition without an immutable-field error.
    requests = container.setdefault("resources", {}).setdefault("requests", {})
    requests.setdefault("cpu", "1")
    requests.setdefault("memory", "2Gi")

    if backend.extra_volumes:
        pod_spec.setdefault("volumes", []).extend(
            copy.deepcopy(volume) for volume in backend.extra_volumes
        )
    if backend.extra_volume_mounts:
        container.setdefault("volumeMounts", []).extend(
            copy.deepcopy(mount) for mount in backend.extra_volume_mounts
        )
    if backend.env:
        env = container.setdefault("env", [])
        for name, value in backend.env:
            # Container HF_HOME would shadow the deployment-level value that
            # mount_model_cache_pvc sets; skip it when the shared cache is used.
            if name == "HF_HOME" and model_cache_pvc:
                continue
            for item in env:
                if item.get("name") == name:
                    item["value"] = value
                    break
            else:
                env.append({"name": name, "value": value})

    checkpoint = decode.setdefault("experimental", {}).setdefault("checkpoint", {})
    checkpoint["enabled"] = True
    checkpoint["targetContainerName"] = backend.target_container
    if backend.checkpoint_startup_policy is not None:
        checkpoint["startupPolicy"] = backend.checkpoint_startup_policy

    if model_cache_pvc:
        mount = model_cache_mount or "/models"
        deployment_spec.mount_model_cache_pvc(model_cache_pvc, mount)

    return deployment_spec


async def _wait_for(
    description: str,
    fn: Callable[[], Any],
    predicate: Callable[[Any], bool],
    *,
    timeout_s: int = 600,
    interval_s: float = 2.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    last_value: Any = None
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            last_value = fn()
            if hasattr(last_value, "__await__"):
                last_value = await last_value
            last_error = None
            if predicate(last_value):
                return last_value
        except TRANSIENT_K8S_EXCEPTIONS as exc:
            last_error = exc
            logger.warning("Transient error while waiting for %s: %s", description, exc)
        await asyncio.sleep(interval_s)
    message = f"timed out waiting for {description}; last={last_value!r}"
    if last_error is not None:
        message += f"; last_error={last_error!r}"
    raise AssertionError(message)


async def _get_dgd(deployment: ManagedDeployment) -> dict[str, Any]:
    if deployment._custom_api is None:
        raise RuntimeError("Kubernetes API not initialized")
    return await deployment._custom_api.get_namespaced_custom_object(
        group="nvidia.com",
        version=deployment.deployment_spec.api_version,
        namespace=deployment.namespace,
        plural=DGD_PLURAL,
        name=deployment.deployment_spec.name,
    )


async def _get_snapshot_resource(
    deployment: ManagedDeployment, plural: str, name: str
) -> dict[str, Any]:
    if deployment._custom_api is None:
        raise RuntimeError("Kubernetes API not initialized")
    return await deployment._custom_api.get_namespaced_custom_object(
        group="nvidia.com",
        version="v1alpha1",
        namespace=deployment.namespace,
        plural=plural,
        name=name,
    )


async def _wait_for_checkpoint_ready(
    deployment: ManagedDeployment,
    backend: CheckpointBackendConfig,
) -> str:
    async def fetch_status() -> dict[str, Any]:
        dgd = await _get_dgd(deployment)
        status = (
            dgd.get("status", {})
            .get("checkpoints", {})
            .get(backend.decode_component, {})
        )
        snapshot_name = status.get("checkpointName")
        snapshot = None
        snapshot_job = None
        if snapshot_name:
            snapshot = await _get_snapshot_resource(
                deployment, POD_SNAPSHOT_PLURAL, snapshot_name
            )
            snapshot_job_name = (
                snapshot.get("metadata", {})
                .get("labels", {})
                .get(SNAPSHOT_JOB_OWNER_LABEL)
            )
            if snapshot_job_name:
                snapshot_job = await _get_snapshot_resource(
                    deployment, SNAPSHOT_JOB_PLURAL, snapshot_job_name
                )
        return {
            "dgd_status": status,
            "snapshot": snapshot,
            "snapshot_job": snapshot_job,
        }

    value = await _wait_for(
        f"{backend.name} DGD auto checkpoint to become Ready",
        fetch_status,
        _automatic_snapshot_is_ready,
        timeout_s=CHECKPOINT_READY_TIMEOUT,
        interval_s=5,
    )
    snapshot = value["snapshot"]
    snapshot_name = snapshot["metadata"]["name"]
    logger.info("Automatic PodSnapshot is Ready: %s", snapshot_name)
    return snapshot_name


def _condition(resource: dict[str, Any], condition_type: str) -> dict[str, Any] | None:
    for condition in resource.get("status", {}).get("conditions", []):
        if condition.get("type") == condition_type:
            return condition
    return None


def _condition_is_true(resource: dict[str, Any], condition_type: str) -> bool:
    condition = _condition(resource, condition_type)
    return condition is not None and condition.get("status") == "True"


def _automatic_snapshot_is_ready(result: dict[str, Any]) -> bool:
    snapshot = result["snapshot"]
    snapshot_job = result["snapshot_job"]
    if snapshot is None or snapshot_job is None:
        return False

    if _condition_is_true(snapshot, "Failed") or _condition_is_true(
        snapshot_job, "Failed"
    ):
        raise AssertionError(
            "automatic snapshot failed before becoming Ready: "
            f"dgd_status={result['dgd_status']!r}; "
            f"snapshot_status={snapshot.get('status', {})!r}; "
            f"snapshot_job_status={snapshot_job.get('status', {})!r}"
        )
    return (
        result["dgd_status"].get("ready") is True
        and _condition_is_true(snapshot, "Ready")
        and _condition_is_true(snapshot_job, "Completed")
        and bool(snapshot.get("status", {}).get("boundSnapshotContentName"))
    )


def _is_snapshot_job_source(pod: Any) -> bool:
    labels = pod.raw.get("metadata", {}).get("labels", {})
    return SNAPSHOT_JOB_OWNER_LABEL in labels


def _runtime_decode_pods(
    deployment: ManagedDeployment, backend: CheckpointBackendConfig
) -> list[Any]:
    pods = deployment.get_pods([backend.decode_component]).get(
        backend.decode_component, []
    )
    return [pod for pod in pods if not _is_snapshot_job_source(pod)]


async def _wait_for_restored_decode_pod(
    deployment: ManagedDeployment,
    backend: CheckpointBackendConfig,
    snapshot_name: str,
    old_pod_names: set[str] | None = None,
) -> Any:
    def find_restored() -> Any:
        pods = _runtime_decode_pods(deployment, backend)
        last_seen: list[dict[str, Any]] = []
        for pod in pods:
            metadata = pod.raw.get("metadata", {})
            name = metadata.get("name", pod.name)
            if old_pod_names is not None and name in old_pod_names:
                continue
            annotations = metadata.get("annotations", {})
            last_seen.append(
                {
                    "name": name,
                    "snapshot": annotations.get(RESTORE_FROM_ANNOTATION),
                    "restored": _condition(pod.raw, RESTORED_CONDITION),
                    "phase": pod.raw.get("status", {}).get("phase"),
                    "node": pod.raw.get("spec", {}).get("nodeName"),
                }
            )
            if annotations.get(RESTORE_FROM_ANNOTATION) != snapshot_name:
                continue
            restored = _condition(pod.raw, RESTORED_CONDITION)
            if restored is None:
                continue
            if (
                restored.get("status") == "False"
                and restored.get("reason") in RESTORE_FAILURE_REASONS
            ):
                raise AssertionError(
                    f"restore failed for decode pod {name}: {last_seen[-1]}"
                )
            if restored.get("status") != "True":
                continue
            return pod
        return last_seen

    restored = await _wait_for(
        f"{backend.name} decode pod to restore from checkpoint",
        find_restored,
        lambda result: not isinstance(result, list),
        timeout_s=RESTORE_READY_TIMEOUT,
        interval_s=5,
    )
    logger.info("Restored decode pod: %s", restored.name)
    return restored


async def _scale_decode_component(
    deployment: ManagedDeployment,
    backend: CheckpointBackendConfig,
    replicas: int,
) -> None:
    if deployment._custom_api is None:
        raise RuntimeError("Kubernetes API not initialized")
    dgd = await _get_dgd(deployment)
    components = dgd["spec"]["components"]
    for component in components:
        if component.get("name") == backend.decode_component:
            component["replicas"] = replicas
            break
    else:
        raise AssertionError(f"component {backend.decode_component!r} not found")

    await deployment._custom_api.patch_namespaced_custom_object(
        group="nvidia.com",
        version=deployment.deployment_spec.api_version,
        namespace=deployment.namespace,
        plural=DGD_PLURAL,
        name=deployment.deployment_spec.name,
        body={"spec": {"components": components}},
        _content_type="application/merge-patch+json",
    )


async def _wait_for_decode_runtime_pod_count(
    deployment: ManagedDeployment,
    backend: CheckpointBackendConfig,
    expected: int,
) -> list[Any]:
    return await _wait_for(
        f"{expected} {backend.name} decode runtime pod(s)",
        lambda: _runtime_decode_pods(deployment, backend),
        lambda pods: len(pods) == expected,
        timeout_s=DECODE_SCALE_TIMEOUT,
        interval_s=2,
    )


def _assert_chat_response(response: requests.Response, expected_model: str) -> None:
    if response.status_code != 200:
        pytest.fail(
            f"Expected status 200, got {response.status_code}. "
            f"Response: {response.text[:500]}",
            pytrace=False,
        )
    data = response.json()
    if data.get("model") != expected_model:
        pytest.fail(
            f"Expected model {expected_model!r}, got response: {data}",
            pytrace=False,
        )
    choices = data.get("choices", [])
    if not choices:
        pytest.fail(
            f"Expected at least one chat choice, got response: {data}",
            pytrace=False,
        )
    message = choices[0].get("message", {})
    if message.get("role") != "assistant":
        pytest.fail(
            f"Expected assistant message, got response: {data}",
            pytrace=False,
        )
    if not message.get("content"):
        pytest.fail(
            f"Expected non-empty assistant content, got response: {data}",
            pytrace=False,
        )


def _assert_inference(base_url: str, endpoint: str, model: str) -> None:
    model_ready = wait_for_model_availability(
        url=base_url,
        endpoint=endpoint,
        model=model,
        logger=logger,
        max_attempts=MODEL_READY_MAX_ATTEMPTS,
        attempt_timeouts=MODEL_READY_ATTEMPT_TIMEOUTS,
    )
    if not model_ready:
        pytest.fail(f"model {model!r} did not become available", pytrace=False)

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
        "max_tokens": DEFAULT_MAX_TOKENS,
        "temperature": DEFAULT_TEMPERATURE,
        "stream": False,
    }
    response = send_request(
        f"{base_url}{endpoint}",
        payload,
        timeout=float(DEFAULT_REQUEST_TIMEOUT),
        method="POST",
    )
    _assert_chat_response(response, expected_model=model)


# The vLLM Immediate case runs a worker while its capture Job holds one GPU.
@pytest.mark.snapshot_restore
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_dgd_checkpoint_restore_deploy(
    namespace: str,
    image: str | None,
    skip_service_restart: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Verify a DGD worker can be checkpointed, restored, and still serve."""
    backend = _checkpoint_backend(request)
    if not image:
        pytest.fail(
            "--image is required for the checkpoint deploy test "
            f"(expected the CI-built {backend.name} checkpoint placeholder image)",
            pytrace=False,
        )
    frontend_image = request.config.getoption("--frontend-image")
    if not frontend_image:
        pytest.fail(
            "--frontend-image is required for the checkpoint deploy test "
            "(expected the CI-built frontend image)",
            pytrace=False,
        )

    suffix = str(int(time.time() * 1000))
    deployment_name = f"{backend.name}-checkpoint-{suffix}"
    deployment_spec = _new_checkpoint_spec(
        backend=backend,
        name=deployment_name,
        namespace=namespace,
        image=image,
        frontend_image=frontend_image,
        model_cache_pvc=request.config.getoption("--model-cache-pvc") or None,
        model_cache_mount=request.config.getoption("--model-cache-mount") or None,
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        readiness_timeout=(
            DEPLOYMENT_READY_TIMEOUT
            if backend.checkpoint_startup_policy is not None
            else IMMEDIATE_DEPLOYMENT_READY_TIMEOUT
        ),
    ) as deployment:
        frontend_pods = deployment.get_pods([backend.frontend_component]).get(
            backend.frontend_component, []
        )
        if not frontend_pods:
            pytest.fail(f"No frontend pods found for {deployment_name}", pytrace=False)
        port_forward = deployment.port_forward(frontend_pods[0], deployment_spec.port)
        if port_forward is None:
            pytest.fail("failed to establish frontend port-forward", pytrace=False)
        base_url = f"http://localhost:{port_forward.local_port}"

        old_pod_names: set[str] | None = None
        if backend.checkpoint_startup_policy is None:
            initial_pods = await _wait_for_decode_runtime_pod_count(
                deployment, backend, expected=1
            )
            old_pod_names = {pod.name for pod in initial_pods}
            logger.info("Validating inference on the initial Immediate worker")
            _assert_inference(base_url, deployment_spec.endpoint, backend.model)

        snapshot_name = await _wait_for_checkpoint_ready(deployment, backend)

        if old_pod_names is not None:
            logger.info("Scaling Immediate decode worker down after capture")
            await _scale_decode_component(deployment, backend, replicas=0)
            await _wait_for_decode_runtime_pod_count(deployment, backend, expected=0)
            logger.info("Scaling Immediate decode worker up to trigger restore")
            await _scale_decode_component(deployment, backend, replicas=1)

        await _wait_for_restored_decode_pod(
            deployment,
            backend=backend,
            snapshot_name=snapshot_name,
            old_pod_names=old_pod_names,
        )

        await deployment._wait_for_ready(timeout=RESTORED_DEPLOYMENT_READY_TIMEOUT)

        logger.info("Validating inference on the restored worker")
        _assert_inference(base_url, deployment_spec.endpoint, backend.model)
