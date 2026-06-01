# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster DGD checkpoint/restore deploy test."""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Callable

import pytest
import requests

from tests.utils.client import send_request, wait_for_model_availability
from tests.utils.managed_deployment import (
    DeploymentSpec,
    ManagedDeployment,
    _get_workspace_dir,
)

logger = logging.getLogger(__name__)

DGD_PLURAL = "dynamographdeployments"
CHECKPOINT_PLURAL = "dynamocheckpoints"

DECODE_COMPONENT = "decode"
FRONTEND_COMPONENT = "Frontend"
TARGET_CONTAINER = "main"
MOCKER_MODEL = "nvidia/Llama-3.1-8B-Instruct-FP8"
DEFAULT_MOCKER_IMAGE = "nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.1.1"
MOCKER_PROFILE_DATA = (
    "/workspace/components/src/dynamo/planner/tests/data/profiling_results/"
    "H200_TP1P_TP1D"
)

CHECKPOINT_ID_LABEL = "nvidia.com/snapshot-checkpoint-id"
CHECKPOINT_SOURCE_LABEL = "nvidia.com/snapshot-is-checkpoint-source"
RESTORE_TARGET_LABEL = "nvidia.com/snapshot-is-restore-target"
TARGET_CONTAINERS_ANNOTATION = "nvidia.com/snapshot-target-containers"
RESTORE_STATUS_ANNOTATION = f"nvidia.com/snapshot-restore-status.{TARGET_CONTAINER}"

GPU_NODE_SELECTOR = {"nvidia.com/gpu.present": "true"}
GPU_TOLERATIONS = [
    {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"},
    {"key": "dedicated", "operator": "Exists", "effect": "NoSchedule"},
]

TEST_PROMPT = "Reply with one short sentence confirming this restored worker can serve."
DEFAULT_MAX_TOKENS = 24
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REQUEST_TIMEOUT = 120


def _checkpoint_job_command() -> str:
    return f"""set -eu
control_dir="${{DYN_SNAPSHOT_CONTROL_DIR:-/snapshot-control}}"
python3 -m dynamo.mocker \\
  --model-path {MOCKER_MODEL} \\
  --model-name {MOCKER_MODEL} \\
  --speedup-ratio 1.0 \\
  --planner-profile-data {MOCKER_PROFILE_DATA} &
child="$!"
sleep 30
if ! kill -0 "${{child}}" 2>/dev/null; then
  wait "${{child}}"
fi
touch "${{control_dir}}/ready-for-checkpoint"
while [ ! -f "${{control_dir}}/snapshot-complete" ]; do
  if ! kill -0 "${{child}}" 2>/dev/null; then
    wait "${{child}}"
  fi
  sleep 1
done
kill -TERM "${{child}}" 2>/dev/null || true
wait "${{child}}" 2>/dev/null || true
"""


def _component(spec: dict[str, Any], name: str) -> dict[str, Any]:
    for component in spec["spec"].get("components", []):
        if component.get("name") == name:
            return component
    raise AssertionError(f"component {name!r} not found in DGD spec")


def _new_mocker_checkpoint_spec(
    name: str, namespace: str, image: str
) -> DeploymentSpec:
    spec_path = (
        Path(_get_workspace_dir())
        / "examples"
        / "backends"
        / "mocker"
        / "deploy"
        / "v1beta1"
        / "agg.yaml"
    )
    deployment_spec = DeploymentSpec(str(spec_path))
    deployment_spec.name = name
    deployment_spec.namespace = namespace
    deployment_spec.set_image(image, FRONTEND_COMPONENT)
    deployment_spec.set_image(image, DECODE_COMPONENT)

    raw_spec = deployment_spec.spec()
    decode = _component(raw_spec, DECODE_COMPONENT)
    pod_spec = decode.setdefault("podTemplate", {}).setdefault("spec", {})
    pod_spec["nodeSelector"] = dict(GPU_NODE_SELECTOR)
    pod_spec["tolerations"] = list(GPU_TOLERATIONS)

    decode.setdefault("experimental", {})["checkpoint"] = {
        "mode": "Auto",
        "targetContainerName": TARGET_CONTAINER,
        "identity": {
            "model": MOCKER_MODEL,
            "backendFramework": "vllm",
            "tensorParallelSize": 1,
            "pipelineParallelSize": 1,
            "extraParameters": {"ciTest": name},
        },
        "job": {
            "podTemplate": {
                "spec": {
                    "nodeSelector": dict(GPU_NODE_SELECTOR),
                    "tolerations": list(GPU_TOLERATIONS),
                    "containers": [
                        {
                            "name": TARGET_CONTAINER,
                            "command": ["/bin/sh", "-c"],
                            "args": [_checkpoint_job_command()],
                        }
                    ],
                }
            }
        },
    }
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
    while time.monotonic() < deadline:
        last_value = fn()
        if hasattr(last_value, "__await__"):
            last_value = await last_value
        if predicate(last_value):
            return last_value
        await asyncio.sleep(interval_s)
    raise AssertionError(f"timed out waiting for {description}; last={last_value!r}")


async def _get_dgd(deployment: ManagedDeployment) -> dict[str, Any]:
    assert deployment._custom_api is not None, "Kubernetes API not initialized"
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
    assert deployment._custom_api is not None, "Kubernetes API not initialized"
    return await deployment._custom_api.get_namespaced_custom_object(
        group="nvidia.com",
        version="v1alpha1",
        namespace=deployment.namespace,
        plural=CHECKPOINT_PLURAL,
        name=checkpoint_name,
    )


async def _wait_for_checkpoint_ready(
    deployment: ManagedDeployment,
) -> tuple[str, str]:
    async def fetch_status() -> dict[str, Any]:
        dgd = await _get_dgd(deployment)
        status = dgd.get("status", {}).get("checkpoints", {}).get(DECODE_COMPONENT, {})
        checkpoint_name = status.get("checkpointName")
        checkpoint = None
        if checkpoint_name:
            checkpoint = await _get_checkpoint(deployment, checkpoint_name)
        return {"dgd_status": status, "checkpoint": checkpoint}

    value = await _wait_for(
        "DGD auto checkpoint to become Ready",
        fetch_status,
        lambda result: result["checkpoint"] is not None
        and result["checkpoint"].get("status", {}).get("phase") == "Ready"
        and bool(result["checkpoint"].get("status", {}).get("identityHash")),
        timeout_s=900,
        interval_s=5,
    )
    checkpoint = value["checkpoint"]
    identity_hash = checkpoint["status"]["identityHash"]
    checkpoint_name = checkpoint["metadata"]["name"]
    logger.info("Checkpoint is Ready: %s (%s)", checkpoint_name, identity_hash)
    return checkpoint_name, identity_hash


def _runtime_decode_pods(deployment: ManagedDeployment) -> list[Any]:
    pods = deployment.get_pods([DECODE_COMPONENT]).get(DECODE_COMPONENT, [])
    return [
        pod
        for pod in pods
        if pod.raw.get("metadata", {}).get("labels", {}).get(CHECKPOINT_SOURCE_LABEL)
        != "true"
    ]


async def _scale_decode_component(deployment: ManagedDeployment, replicas: int) -> None:
    assert deployment._custom_api is not None, "Kubernetes API not initialized"
    dgd = await _get_dgd(deployment)
    components = dgd["spec"]["components"]
    for component in components:
        if component.get("name") == DECODE_COMPONENT:
            component["replicas"] = replicas
            break
    else:
        raise AssertionError(f"component {DECODE_COMPONENT!r} not found")

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
    deployment: ManagedDeployment, expected: int, timeout_s: int
) -> list[Any]:
    return await _wait_for(
        f"{expected} decode runtime pod(s)",
        lambda: _runtime_decode_pods(deployment),
        lambda pods: len(pods) == expected,
        timeout_s=timeout_s,
        interval_s=2,
    )


async def _wait_for_restored_decode_pod(
    deployment: ManagedDeployment,
    old_pod_names: set[str],
    checkpoint_hash: str,
) -> Any:
    def find_restored() -> Any:
        pods = _runtime_decode_pods(deployment)
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
                    "restore": annotations.get(RESTORE_STATUS_ANNOTATION),
                    "phase": pod.raw.get("status", {}).get("phase"),
                }
            )
            if name in old_pod_names:
                continue
            if labels.get(CHECKPOINT_ID_LABEL) != checkpoint_hash:
                continue
            if labels.get(RESTORE_TARGET_LABEL) != "true":
                continue
            if annotations.get(TARGET_CONTAINERS_ANNOTATION) != TARGET_CONTAINER:
                continue
            if annotations.get(RESTORE_STATUS_ANNOTATION) != "completed":
                continue
            return pod
        return last_seen

    restored = await _wait_for(
        "replacement decode pod to restore from checkpoint",
        find_restored,
        lambda result: not isinstance(result, list),
        timeout_s=900,
        interval_s=5,
    )
    logger.info("Restored decode pod: %s", restored.name)
    return restored


def _assert_chat_response(response: requests.Response, expected_model: str) -> None:
    assert response.status_code == 200, (
        f"Expected status 200, got {response.status_code}. "
        f"Response: {response.text[:500]}"
    )
    data = response.json()
    assert data.get("model") == expected_model, data
    choices = data.get("choices", [])
    assert choices, data
    message = choices[0].get("message", {})
    assert message.get("role") == "assistant", data
    assert message.get("content"), data


def _assert_inference(base_url: str, endpoint: str, model: str) -> None:
    model_ready = wait_for_model_availability(
        url=base_url,
        endpoint=endpoint,
        model=model,
        logger=logger,
        max_attempts=30,
    )
    assert model_ready, f"model {model!r} did not become available"

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
@pytest.mark.none
@pytest.mark.timeout(1800)
async def test_dgd_checkpoint_restore_deploy(
    namespace: str,
    image: str | None,
    skip_service_restart: bool,
    request: pytest.FixtureRequest,
) -> None:
    """Verify a DGD worker can be checkpointed, restored, and still serve."""
    suffix = str(int(time.time() * 1000))
    deployment_name = f"mocker-checkpoint-{suffix}"
    deployment_spec = _new_mocker_checkpoint_spec(
        name=deployment_name,
        namespace=namespace,
        image=image or DEFAULT_MOCKER_IMAGE,
    )

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
    ) as deployment:
        frontend_pods = deployment.get_pods([FRONTEND_COMPONENT]).get(
            FRONTEND_COMPONENT, []
        )
        assert frontend_pods, f"No frontend pods found for {deployment_name}"
        port_forward = deployment.port_forward(frontend_pods[0], deployment_spec.port)
        assert port_forward is not None, "failed to establish frontend port-forward"
        base_url = f"http://localhost:{port_forward.local_port}"

        logger.info("Validating inference before restore")
        _assert_inference(base_url, deployment_spec.endpoint, MOCKER_MODEL)

        _, checkpoint_hash = await _wait_for_checkpoint_ready(deployment)

        old_decode_pods = await _wait_for_decode_runtime_pod_count(
            deployment, expected=1, timeout_s=300
        )
        old_pod_names = {pod.name for pod in old_decode_pods}
        logger.info("Scaling decode down from pods: %s", sorted(old_pod_names))
        await _scale_decode_component(deployment, replicas=0)
        await _wait_for_decode_runtime_pod_count(deployment, expected=0, timeout_s=300)

        logger.info("Scaling decode back up to trigger restore")
        await _scale_decode_component(deployment, replicas=1)
        await deployment._wait_for_ready(timeout=900)
        await _wait_for_restored_decode_pod(
            deployment,
            old_pod_names=old_pod_names,
            checkpoint_hash=checkpoint_hash,
        )

        logger.info("Validating inference after restore")
        _assert_inference(base_url, deployment_spec.endpoint, MOCKER_MODEL)
