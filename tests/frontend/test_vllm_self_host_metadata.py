# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Local e2e: vLLM worker + frontend with `DYN_SELF_HOST_METADATA=true`."""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Generator, Tuple

import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import ServicePorts

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.post_merge,
    pytest.mark.model(TEST_MODEL),
]


class SelfHostVllmWorkerProcess(ManagedProcess):
    def __init__(
        self,
        request,
        *,
        frontend_port: int,
        system_port: int,
        worker_id: str = "vllm-worker-self-host",
    ):
        self.frontend_port = int(frontend_port)
        self.system_port = int(system_port)

        command = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            TEST_MODEL,
            "--max-model-len",
            "4096",
        ]

        env = os.environ.copy()
        env["DYN_LOG"] = "info"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_SELF_HOST_METADATA"] = "true"

        log_dir = f"{request.node.name}_{worker_id}"
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=[
                (f"http://localhost:{self.frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{self.system_port}/health", self.is_ready),
            ],
            timeout=500,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["VLLM::EngineCore"],
            straggler_commands=["-m dynamo.vllm"],
            log_dir=log_dir,
        )

    def is_ready(self, response) -> bool:
        try:
            status = (response.json() or {}).get("status")
        except ValueError:
            return False
        return status == "ready"


@pytest.fixture(scope="function")
def start_self_host_services(
    request,
    runtime_services_dynamic_ports,
    dynamo_dynamic_ports: ServicePorts,
    tmp_path,
) -> Generator[Tuple[ServicePorts, Path], None, None]:
    _ = runtime_services_dynamic_ports
    frontend_port = dynamo_dynamic_ports.frontend_port
    system_port = dynamo_dynamic_ports.system_ports[0]

    # Isolate the frontend's HOME so its MDC cache (and any HF cache
    # it might use as a fallback) lives under the test tmpdir. After
    # the test, we inspect this path to confirm the self-host http
    # path actually ran.
    frontend_home = tmp_path / "frontend-home"
    frontend_home.mkdir(parents=True, exist_ok=True)

    with DynamoFrontendProcess(
        request,
        frontend_port=frontend_port,
        extra_env={"HOME": str(frontend_home)},
        terminate_all_matching_process_names=False,
    ):
        with SelfHostVllmWorkerProcess(
            request,
            frontend_port=frontend_port,
            system_port=system_port,
        ):
            yield dynamo_dynamic_ports, frontend_home


@pytest.mark.timeout(300)
def test_self_host_metadata_env_var_registers_model(
    request,
    start_self_host_services: Tuple[ServicePorts, Path],
    predownload_models,
) -> None:
    ports, frontend_home = start_self_host_services
    base_url = f"http://localhost:{ports.frontend_port}"

    response = requests.get(f"{base_url}/v1/models", timeout=30)
    assert (
        response.status_code == 200
    ), f"GET /v1/models failed: {response.status_code} {response.text}"

    data = response.json()
    model_ids = [m.get("id") for m in data.get("data", [])]
    assert TEST_MODEL in model_ids, f"expected {TEST_MODEL} in {model_ids}"

    # Validate the frontend actually used the self-host http path.
    # `download_files` is the only writer to ~/.cache/dynamo/mdc/blobs/;
    # the legacy hub::from_hf fallback writes to the HF Hub cache,
    # which is a different directory. Blobs in the isolated HOME
    # prove download_files ran with the http URIs the worker
    # advertised in its MDC.
    blobs_dir = frontend_home / ".cache/dynamo/mdc/blobs"
    assert blobs_dir.exists(), (
        f"expected MDC blobs dir at {blobs_dir} — frontend did not exercise "
        f"the self-host http path"
    )
    blobs = list(blobs_dir.iterdir())
    assert len(blobs) > 0, (
        f"expected at least one blob in {blobs_dir} from the http fetch; " f"none found"
    )
    logger.info(
        "self-host http path verified: %d blob(s) under %s", len(blobs), blobs_dir
    )
