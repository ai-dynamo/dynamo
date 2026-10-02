# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live-cluster lifecycle checks for the shipped native-sidecar manifests."""

import asyncio
import logging
from pathlib import Path
from uuid import uuid4

import pytest

from tests.deploy.dgd_utils import (
    DEFAULT_REQUEST_TIMEOUT,
    DeploymentSpec,
    ManagedDeployment,
    _get_workspace_dir,
    validate_chat_response,
)
from tests.utils.client import wait_for_model_availability

MODEL = "Qwen/Qwen3-0.6B"
logger = logging.getLogger(__name__)


def _container(pod, name: str, *, is_engine: bool = False):
    statuses = (
        pod.status.init_container_statuses
        if is_engine
        else pod.status.container_statuses
    )
    return next(status for status in statuses if status.name == name)


def _ready(pod) -> bool:
    return any(
        condition.type == "Ready" and condition.status == "True"
        for condition in pod.status.conditions or []
    )


async def _wait_for_pod(deployment, name, predicate, timeout):
    async def poll():
        while True:
            pod = await deployment._core_api.read_namespaced_pod(
                name, deployment.namespace
            )
            if predicate(pod):
                return pod
            await asyncio.sleep(1)

    return await asyncio.wait_for(poll(), timeout)


@pytest.mark.framework_agnostic
@pytest.mark.k8s
@pytest.mark.deploy
@pytest.mark.sidecar
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.model(MODEL)
@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "backend,worker_name",
    [
        pytest.param("vllm", "worker", marks=pytest.mark.vllm),
        pytest.param("sglang", "SGLangWorker", marks=pytest.mark.sglang),
    ],
)
async def test_native_sidecar_engine_restart_and_deployment_cleanup(
    backend: str,
    worker_name: str,
    image: str | None,
    namespace: str,
    request: pytest.FixtureRequest,
) -> None:
    """Restart only the native engine, then verify recovery and owned-pod deletion."""
    frontend_image = request.config.getoption("--frontend-image")
    if not image or not frontend_image or not namespace:
        pytest.skip(
            "Requires --image (Rust sidecar), --frontend-image, and --namespace"
        )

    manifest = (
        Path(_get_workspace_dir()) / "lib" / "sidecar" / backend / "deploy/agg.yaml"
    )
    spec = DeploymentSpec(str(manifest))
    spec.name = f"{backend}-sidecar-lifecycle-{uuid4().hex[:8]}"
    spec.set_image(image, worker_name)
    spec.set_image(frontend_image, "Frontend")
    engine_name = f"{backend}-engine"

    async with ManagedDeployment(
        log_dir=request.node.name,
        deployment_spec=spec,
        namespace=namespace,
        skip_service_restart=True,
        readiness_timeout=600,
    ) as deployment:
        worker = deployment.get_pods([worker_name])[worker_name]
        assert len(worker) == 1
        worker = worker[0]
        original = await deployment._core_api.read_namespaced_pod(
            worker.name, namespace
        )
        engine_spec = next(
            container
            for container in original.spec.init_containers
            if container.name == engine_name
        )
        assert engine_spec.restart_policy == "Always"
        assert engine_spec.startup_probe is not None
        assert engine_spec.readiness_probe is not None
        original_engine = _container(original, engine_name, is_engine=True)
        original_main = _container(original, "main")
        assert original_engine.ready and original_main.ready and _ready(original)

        frontend = deployment.get_pods(["Frontend"])["Frontend"][0]
        port_forward = deployment.port_forward(frontend, spec.port)
        assert port_forward is not None

        def assert_inference():
            assert wait_for_model_availability(
                f"http://localhost:{port_forward.local_port}",
                spec.endpoint,
                MODEL,
                logger,
                max_attempts=6,
                attempt_timeouts=[10] * 6,
            )
            response = deployment.send_request_with_port_forward_retry(
                frontend,
                spec.port,
                spec.endpoint,
                {
                    "model": MODEL,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "max_tokens": 64,
                    "temperature": 0,
                    "stream": False,
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                float(DEFAULT_REQUEST_TIMEOUT),
                port_forward,
            )
            validate_chat_response(response, MODEL, min_content_length=1)

        assert_inference()
        signalled = worker.exec(
            ["python3", "-c", "import os, signal; os.kill(1, signal.SIGTERM)"],
            container=engine_name,
            check=False,
            timeout=30,
        )
        assert signalled.returncode in (0, 137, 143), signalled
        await _wait_for_pod(
            deployment,
            worker.name,
            lambda pod: (
                not _container(pod, engine_name, is_engine=True).ready
                and not _ready(pod)
            ),
            90,
        )
        recovered = await _wait_for_pod(
            deployment,
            worker.name,
            lambda pod: (
                _container(pod, engine_name, is_engine=True).restart_count
                > original_engine.restart_count
                and _ready(pod)
            ),
            600,
        )
        assert recovered.metadata.uid == original.metadata.uid
        assert (
            _container(recovered, engine_name, is_engine=True).container_id
            != original_engine.container_id
        )
        main = _container(recovered, "main")
        assert main.container_id == original_main.container_id
        assert main.restart_count == original_main.restart_count
        assert_inference()

    async def wait_for_cleanup():
        while await deployment.get_pod_names():
            await asyncio.sleep(1)

    await asyncio.wait_for(wait_for_cleanup(), 120)
