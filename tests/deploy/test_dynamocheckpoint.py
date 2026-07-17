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

from tests.utils.client import send_request, wait_for_model_availability
from tests.utils.managed_deployment import (
    DeploymentSpec,
    ManagedDeployment,
    _get_workspace_dir,
)

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
CHECKPOINT_PLURAL = "dynamocheckpoints"

FRONTEND_COMPONENT = "Frontend"
TARGET_CONTAINER = "main"
CHECKPOINT_MODEL = "Qwen/Qwen3-0.6B"
CHECKPOINT_STORAGE_MOUNT_PATH = "/checkpoints"
TRTLLM_HF_HOME = f"{CHECKPOINT_STORAGE_MOUNT_PATH}/trtllm-hf-cache"

CHECKPOINT_ID_LABEL = "nvidia.com/snapshot-checkpoint-id"
CHECKPOINT_SOURCE_LABEL = "nvidia.com/snapshot-is-checkpoint-source"
RESTORE_TARGET_LABEL = "nvidia.com/snapshot-is-restore-target"
TARGET_CONTAINERS_ANNOTATION = "nvidia.com/snapshot-target-containers"

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
CHECKPOINT_READY_TIMEOUT = 300
RESTORE_READY_TIMEOUT = 300
DECODE_SCALE_TIMEOUT = 180
DGD_READY_TIMEOUT = 300
TEST_TIMEOUT = 1200


@dataclass(frozen=True)
class CheckpointBackendConfig:
    name: str
    manifest: tuple[str, ...]
    decode_component: str
    frontend_component: str
    target_container: str
    model: str
    command: tuple[str, ...]
    args: tuple[str, ...]
    env: tuple[tuple[str, str], ...] = ()
    extra_volumes: tuple[dict[str, Any], ...] = ()
    extra_volume_mounts: tuple[dict[str, Any], ...] = ()
    pod_spec_updates: dict[str, Any] | None = None
    container_resources: dict[str, Any] | None = None
    checkpoint_startup_policy: str | None = None


CHECKPOINT_BACKENDS = {
    "vllm": CheckpointBackendConfig(
        name="vllm",
        manifest=("examples", "backends", "vllm", "deploy", "v1beta1", "agg.yaml"),
        decode_component="VllmDecodeWorker",
        frontend_component=FRONTEND_COMPONENT,
        target_container=TARGET_CONTAINER,
        model=CHECKPOINT_MODEL,
        command=("python3", "-m", "dynamo.vllm.unified_main"),
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
            "v1beta1",
            "agg.yaml",
        ),
        decode_component="decode",
        frontend_component=FRONTEND_COMPONENT,
        target_container=TARGET_CONTAINER,
        model=CHECKPOINT_MODEL,
        command=("python3", "-m", "dynamo.sglang.unified_main"),
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
    ),
    "trtllm": CheckpointBackendConfig(
        name="trtllm",
        manifest=(
            "examples",
            "backends",
            "trtllm",
            "deploy",
            "v1beta1",
            "agg.yaml",
        ),
        decode_component="TRTLLMWorker",
        frontend_component=FRONTEND_COMPONENT,
        target_container=TARGET_CONTAINER,
        model=CHECKPOINT_MODEL,
        command=("python3", "-m", "dynamo.trtllm.unified_main"),
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
        # Keep the raw DGD PVC-free: the checkpoint operator mounts
        # snapshot-pvc at /checkpoints for checkpoint/restore pods, so HF_HOME
        # there preserves model files across restore without a model-cache PVC.
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
    container["command"] = list(backend.command)
    container["args"] = list(backend.args)
    if backend.container_resources:
        container["resources"] = copy.deepcopy(backend.container_resources)
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


async def _get_checkpoint(
    deployment: ManagedDeployment, checkpoint_name: str
) -> dict[str, Any]:
    if deployment._custom_api is None:
        raise RuntimeError("Kubernetes API not initialized")
    return await deployment._custom_api.get_namespaced_custom_object(
        group="nvidia.com",
        version="v1alpha1",
        namespace=deployment.namespace,
        plural=CHECKPOINT_PLURAL,
        name=checkpoint_name,
    )


async def _wait_for_checkpoint_ready(
    deployment: ManagedDeployment,
    backend: CheckpointBackendConfig,
) -> tuple[str, str]:
    async def fetch_status() -> dict[str, Any]:
        dgd = await _get_dgd(deployment)
        status = (
            dgd.get("status", {})
            .get("checkpoints", {})
            .get(backend.decode_component, {})
        )
        checkpoint_name = status.get("checkpointName")
        checkpoint = None
        if checkpoint_name:
            checkpoint = await _get_checkpoint(deployment, checkpoint_name)
        return {"dgd_status": status, "checkpoint": checkpoint}

    value = await _wait_for(
        f"{backend.name} DGD auto checkpoint to become Ready",
        fetch_status,
        _checkpoint_is_ready,
        timeout_s=CHECKPOINT_READY_TIMEOUT,
        interval_s=5,
    )
    checkpoint = value["checkpoint"]
    identity_hash = checkpoint["status"]["identityHash"]
    checkpoint_name = checkpoint["metadata"]["name"]
    logger.info("Checkpoint is Ready: %s (%s)", checkpoint_name, identity_hash)
    return checkpoint_name, identity_hash


def _checkpoint_is_ready(result: dict[str, Any]) -> bool:
    checkpoint = result["checkpoint"]
    if checkpoint is None:
        return False

    status = checkpoint.get("status", {})
    phase = status.get("phase")
    if phase == "Failed":
        raise AssertionError(
            "checkpoint failed before becoming Ready: "
            f"dgd_status={result['dgd_status']!r}; "
            f"checkpoint_status={status!r}"
        )
    return phase == "Ready" and bool(status.get("identityHash"))


def _runtime_decode_pods(
    deployment: ManagedDeployment, backend: CheckpointBackendConfig
) -> list[Any]:
    pods = deployment.get_pods([backend.decode_component]).get(
        backend.decode_component, []
    )
    return [
        pod
        for pod in pods
        if pod.raw.get("metadata", {}).get("labels", {}).get(CHECKPOINT_SOURCE_LABEL)
        != "true"
    ]


async def _scale_decode_component(
    deployment: ManagedDeployment, backend: CheckpointBackendConfig, replicas: int
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
    timeout_s: int,
) -> list[Any]:
    return await _wait_for(
        f"{expected} {backend.name} decode runtime pod(s)",
        lambda: _runtime_decode_pods(deployment, backend),
        lambda pods: len(pods) == expected,
        timeout_s=timeout_s,
        interval_s=2,
    )


async def _wait_for_restored_decode_pod(
    deployment: ManagedDeployment,
    backend: CheckpointBackendConfig,
    old_pod_names: set[str],
    checkpoint_hash: str,
) -> Any:
    restore_status_annotation = (
        f"nvidia.com/snapshot-restore-status.{backend.target_container}"
    )

    def find_restored() -> Any:
        pods = _runtime_decode_pods(deployment, backend)
        last_seen: list[dict[str, Any]] = []
        for pod in pods:
            metadata = pod.raw.get("metadata", {})
            name = metadata.get("name", pod.name)
            labels = metadata.get("labels", {})
            annotations = metadata.get("annotations", {})
            last_seen.append(
                {
                    "name": name,
                    "checkpoint": labels.get(CHECKPOINT_ID_LABEL),
                    "restore": annotations.get(restore_status_annotation),
                    "phase": pod.raw.get("status", {}).get("phase"),
                    "node": pod.raw.get("spec", {}).get("nodeName"),
                }
            )
            if name in old_pod_names:
                continue
            if labels.get(CHECKPOINT_ID_LABEL) != checkpoint_hash:
                continue
            if labels.get(RESTORE_TARGET_LABEL) != "true":
                continue
            if (
                annotations.get(TARGET_CONTAINERS_ANNOTATION)
                != backend.target_container
            ):
                continue
            if annotations.get(restore_status_annotation) == "failed":
                raise AssertionError(
                    f"restore failed for decode pod {name}: {last_seen[-1]}"
                )
            if annotations.get(restore_status_annotation) != "completed":
                continue
            return pod
        return last_seen

    restored = await _wait_for(
        f"replacement {backend.name} decode pod to restore from checkpoint",
        find_restored,
        lambda result: not isinstance(result, list),
        timeout_s=RESTORE_READY_TIMEOUT,
        interval_s=5,
    )
    logger.info("Restored decode pod: %s", restored.name)
    return restored


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
        max_attempts=30,
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


@pytest.mark.dynamocheckpoint
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.post_merge
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.timeout(TEST_TIMEOUT)
async def test_dgd_checkpoint_restore_deploy(
    namespace: str,
    image: str | None,
    skip_service_restart: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Verify a unified worker can be checkpointed, restored, and still serve."""
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
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
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

        logger.info("Validating inference before restore")
        _assert_inference(base_url, deployment_spec.endpoint, backend.model)

        _, checkpoint_hash = await _wait_for_checkpoint_ready(deployment, backend)

        old_decode_pods = await _wait_for_decode_runtime_pod_count(
            deployment,
            backend=backend,
            expected=1,
            timeout_s=DECODE_SCALE_TIMEOUT,
        )
        old_pod_names = {pod.name for pod in old_decode_pods}
        logger.info("Scaling decode down from pods: %s", sorted(old_pod_names))
        await _scale_decode_component(deployment, backend, replicas=0)
        await _wait_for_decode_runtime_pod_count(
            deployment,
            backend=backend,
            expected=0,
            timeout_s=DECODE_SCALE_TIMEOUT,
        )

        logger.info("Scaling decode back up to trigger restore")
        await _scale_decode_component(deployment, backend, replicas=1)
        await _wait_for_restored_decode_pod(
            deployment,
            backend=backend,
            old_pod_names=old_pod_names,
            checkpoint_hash=checkpoint_hash,
        )
        await deployment._wait_for_ready(timeout=DGD_READY_TIMEOUT)

        logger.info("Validating inference after restore")
        _assert_inference(base_url, deployment_spec.endpoint, backend.model)
