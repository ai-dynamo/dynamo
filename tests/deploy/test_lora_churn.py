# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Nightly regression coverage for repeated vLLM LoRA registration."""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import requests
import yaml

from tests.deploy.dgd_utils import DeploymentSpec, ManagedDeployment, _get_workspace_dir
from tests.utils.client import send_request, wait_for_model_availability

logger = logging.getLogger(__name__)

BASE_MODEL = "Qwen/Qwen3-0.6B"
LORA_NAME = "codelion/Qwen3-0.6B-accuracy-recovery-lora"
LORA_SOURCE = f"hf://{LORA_NAME}"
CHURN_CYCLES = 30
CHATS_PER_CYCLE = 1
MAX_FD_GROWTH = 3
MAX_RSS_GROWTH_KIB = 64 * 1024
UNLOAD_TIMEOUT_SECONDS = 30
RSS_SETTLE_SECONDS = 60


def _memory_probe(pod: Any) -> dict[str, int]:
    """Read cgroup memory and PID 1 descriptor counts from a pod."""
    snippet = """
import json
import os
from pathlib import Path
memory_paths = (
    Path("/sys/fs/cgroup/memory.current"),
    Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"),
)
memory_bytes = next(
    (int(path.read_text()) for path in memory_paths if path.exists()),
    None,
)
if memory_bytes is None:
    raise RuntimeError("No cgroup memory counter is available")

targets = []
for descriptor in Path("/proc/1/fd").iterdir():
    try:
        targets.append(os.readlink(descriptor))
    except FileNotFoundError:
        pass
print(json.dumps({
    "fds": len(targets),
    "sockets": sum(target.startswith("socket:") for target in targets),
    "rss_kib": memory_bytes // 1024,
}))
"""
    result = pod.exec(["python3", "-c", snippet])
    return json.loads(result.stdout.decode())


def _snapshot(deployment: ManagedDeployment) -> dict[str, dict[str, int]]:
    """Capture resource measurements for the frontend and decode worker."""
    snapshots: dict[str, dict[str, int]] = {}
    for service_name in ("Frontend", "VllmDecodeWorker"):
        pods = deployment.get_pods([service_name]).get(service_name, [])
        assert len(pods) == 1, f"Expected one {service_name} pod, got {len(pods)}"
        snapshots[service_name] = _memory_probe(pods[0])
    return snapshots


def _dgd_manifest_path(tmp_path: Path) -> Path:
    """Write the DGD from the two-document LoRA example to an isolated file."""
    manifest = (
        Path(_get_workspace_dir())
        / "examples/backends/vllm/deploy/lora/agg_lora_hf.yaml"
    )
    documents = list(yaml.safe_load_all(manifest.read_text()))
    dgd = next(
        document
        for document in documents
        if document and document.get("kind") == "DynamoGraphDeployment"
    )
    dgd_path = tmp_path / "lora-churn-dgd.yaml"
    dgd_path.write_text(yaml.safe_dump(dgd, sort_keys=False))
    return dgd_path


def _adapter_present(base_url: str) -> bool:
    response = requests.get(f"{base_url}/v1/models", timeout=10)
    response.raise_for_status()
    return any(
        model.get("id") == LORA_NAME for model in response.json().get("data", [])
    )


def _adapter_removed(base_url: str) -> bool:
    loras = requests.get(f"{base_url}/v1/loras", timeout=10)
    loras.raise_for_status()
    return not _adapter_present(base_url) and loras.json().get("count") == 0


def _wait_for(predicate: Callable[[], bool], description: str) -> None:
    deadline = time.monotonic() + UNLOAD_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(1)
    pytest.fail(f"Timed out waiting for {description}")


def _load_lora(base_url: str) -> None:
    response = requests.post(
        f"{base_url}/v1/loras",
        json={"lora_name": LORA_NAME, "source": {"uri": LORA_SOURCE}},
        timeout=60,
    )
    assert response.ok, response.text
    _wait_for(lambda: _adapter_present(base_url), "LoRA to appear in /v1/models")


def _unload_lora(base_url: str) -> None:
    response = requests.delete(f"{base_url}/v1/loras/{LORA_NAME}", timeout=60)
    assert response.ok, response.text
    _wait_for(lambda: _adapter_removed(base_url), "LoRA to leave discovery")


def _assert_bounded_growth(
    baseline: dict[str, dict[str, int]],
    current: dict[str, dict[str, int]],
    cycle: int,
) -> None:
    for service_name, baseline_values in baseline.items():
        current_values = current[service_name]
        assert current_values["fds"] <= baseline_values["fds"] + MAX_FD_GROWTH, (
            f"{service_name} retained descriptors after cycle {cycle}: "
            f"baseline={baseline_values}, current={current_values}"
        )
        assert (
            current_values["sockets"] <= baseline_values["sockets"] + MAX_FD_GROWTH
        ), (
            f"{service_name} retained sockets after cycle {cycle}: "
            f"baseline={baseline_values}, current={current_values}"
        )
        assert (
            current_values["rss_kib"] <= baseline_values["rss_kib"] + MAX_RSS_GROWTH_KIB
        ), (
            f"{service_name} RSS grew after cycle {cycle}: "
            f"baseline={baseline_values}, current={current_values}"
        )


@pytest.mark.framework_agnostic
@pytest.mark.vllm
@pytest.mark.model(BASE_MODEL)
@pytest.mark.model(LORA_NAME)
@pytest.mark.profiled_vram_gib(4.0)
@pytest.mark.requested_vllm_kv_cache_bytes(941_712_000)
@pytest.mark.nightly
@pytest.mark.framework_only
@pytest.mark.core
@pytest.mark.e2e
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.gpu_1
@pytest.mark.timeout(1800)
async def test_lora_registration_churn_has_bounded_resources(
    image: str,
    namespace: str,
    skip_service_restart: bool,
    tmp_path: Path,
    request: pytest.FixtureRequest,
) -> None:
    """Repeated dynamic LoRA load/unload must not retain sockets or RSS."""
    frontend_image = request.config.getoption("--frontend-image")
    assert image, "--image is required for the vLLM decode worker"
    assert frontend_image, "--frontend-image is required for the frontend"
    assert namespace, "--namespace is required for the Kubernetes deployment"

    kv_cache_marker = request.node.get_closest_marker("requested_vllm_kv_cache_bytes")
    assert kv_cache_marker and kv_cache_marker.args, "vLLM KV cache budget is required"
    kv_cache_bytes = os.environ.get(
        "_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", str(kv_cache_marker.args[0])
    )

    deployment_spec = DeploymentSpec(str(_dgd_manifest_path(tmp_path)))
    deployment_spec.name = "vllm-lora-churn"
    deployment_spec.set_image(frontend_image, service_name="Frontend")
    deployment_spec.set_image(image, service_name="VllmDecodeWorker")
    deployment_spec.add_arg_to_service(
        "VllmDecodeWorker", "--kv-cache-memory-bytes", kv_cache_bytes
    )
    deployment_spec.add_arg_to_service(
        "VllmDecodeWorker", "--gpu-memory-utilization", "0.01"
    )
    model_cache_pvc = request.config.getoption("--model-cache-pvc")
    if model_cache_pvc:
        deployment_spec.mount_model_cache_pvc(model_cache_pvc)

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=deployment_spec,
        namespace=namespace,
        skip_service_restart=skip_service_restart,
        readiness_timeout=900,
    ) as deployment:
        frontend = deployment.get_pods(["Frontend"])["Frontend"]
        assert len(frontend) == 1, "Expected one frontend pod"
        port_forward = deployment.port_forward(frontend[0], deployment_spec.port)
        assert port_forward is not None, "Unable to port-forward the frontend"
        base_url = f"http://localhost:{port_forward.local_port}"
        assert wait_for_model_availability(
            url=base_url,
            endpoint=deployment_spec.endpoint,
            model=BASE_MODEL,
            logger=logger,
            max_attempts=30,
        ), "Base model did not become available"

        _load_lora(base_url)
        _unload_lora(base_url)
        baseline = _snapshot(deployment)
        logger.info("LoRA churn resource baseline: %s", baseline)

        for cycle in range(1, CHURN_CYCLES + 1):
            _load_lora(base_url)
            for _ in range(CHATS_PER_CYCLE):
                response = send_request(
                    f"{base_url}/v1/chat/completions",
                    {
                        "model": LORA_NAME,
                        "messages": [
                            {"role": "user", "content": "Reply with NVIDIA Dynamo."}
                        ],
                        "max_tokens": 16,
                        "temperature": 0,
                    },
                    timeout=60,
                )
                assert response.ok, response.text
                body = response.json()
                assert body.get("object") == "chat.completion", body
                assert body.get("choices"), body
            _unload_lora(base_url)
            measurements = _snapshot(deployment)
            logger.info("LoRA churn cycle %d: %s", cycle, measurements)
            _assert_bounded_growth(baseline, measurements, cycle)

        await asyncio.sleep(RSS_SETTLE_SECONDS)
        settled = _snapshot(deployment)
        logger.info("LoRA churn settled resources: %s", settled)
        _assert_bounded_growth(baseline, settled, CHURN_CYCLES)
