# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
End-to-end test for decode-only fallback metric with SGLang backend.

Verifies that when a prefill worker goes down and enforce_disagg is disabled,
requests fall back to decode-only mode and the decode_only_fallback_total
Prometheus counter increments accordingly.
"""

import logging
import os
import shutil

import pytest

from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import check_models_api
from tests.utils.port_utils import allocate_port, deallocate_port

from .utils import DynamoFrontendProcess, run_decode_fallback_test

logger = logging.getLogger(__name__)

pytestmark = [
    pytest.mark.fault_tolerance,
    pytest.mark.sglang,
    pytest.mark.gpu_1,
    pytest.mark.e2e,
    pytest.mark.model(FAULT_TOLERANCE_MODEL_NAME),
    pytest.mark.post_merge,
    pytest.mark.parametrize(
        "request_api",
        [
            pytest.param("chat"),
            pytest.param(
                "completion",
                marks=pytest.mark.skip(reason="Behavior unverified yet"),
            ),
        ],
    ),
    pytest.mark.parametrize(
        "stream",
        [
            pytest.param(True, id="stream"),
            pytest.param(
                False,
                id="unary",
                marks=pytest.mark.skip(reason="Behavior unverified yet"),
            ),
        ],
    ),
    pytest.mark.parametrize("request_plane", ["nats", "tcp"], indirect=True),
]


class DynamoWorkerProcess(ManagedProcess):
    """Process manager for Dynamo worker with SGLang backend.

    Supports disaggregated mode with separate prefill and decode workers.

    Args:
        request: pytest request fixture
        worker_id: Unique identifier for the worker (e.g., "prefill1", "decode1")
        frontend_port: Port where the frontend is running
        disagg_mode: "prefill" or "decode" for disaggregated mode
    """

    def __init__(
        self,
        request,
        worker_id: str,
        frontend_port: int,
        disagg_mode: str,
    ):
        self.worker_id = worker_id
        self.system_port = allocate_port(9100)
        self.disagg_mode = disagg_mode

        command = [
            "python3",
            "-m",
            "dynamo.sglang",
            "--model-path",
            FAULT_TOLERANCE_MODEL_NAME,
            "--served-model-name",
            FAULT_TOLERANCE_MODEL_NAME,
            "--trust-remote-code",
            "--page-size",
            "16",
            "--tp",
            "1",
            "--mem-fraction-static",
            "0.3",
            "--context-length",
            "8192",
            "--disaggregation-mode",
            disagg_mode,
            "--disaggregation-bootstrap-port",
            f"1234{worker_id[-1]}",
            "--host",
            "0.0.0.0",
            "--disaggregation-transfer-backend",
            "nixl",
        ]
        if disagg_mode == "prefill":
            command.extend(["--port", "40000"])

        env = os.environ.copy()
        env["DYN_REQUEST_PLANE"] = request.getfixturevalue("request_plane")
        env["DYN_LOG"] = "debug"
        env["DYN_HEALTH_CHECK_ENABLED"] = "false"
        env["DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS"] = '["generate"]'
        env["DYN_SYSTEM_PORT"] = str(self.system_port)
        env["DYN_HTTP_PORT"] = str(frontend_port)

        health_check_urls = [
            (f"http://localhost:{self.system_port}/health", self.is_ready)
        ]
        if disagg_mode == "decode":
            health_check_urls.append(
                (f"http://localhost:{frontend_port}/v1/models", check_models_api)
            )

        log_dir = f"{request.node.name}_{worker_id}"
        try:
            shutil.rmtree(log_dir)
            logger.info(f"Cleaned up existing log directory: {log_dir}")
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            health_check_urls=health_check_urls,
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=["SGLANG:EngineCore"],
            straggler_commands=["-m dynamo.sglang"],
            log_dir=log_dir,
            display_name=worker_id,
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            deallocate_port(self.system_port)
        except Exception as e:
            logging.warning(f"Failed to release SGLang worker port: {e}")
        return super().__exit__(exc_type, exc_val, exc_tb)

    def is_ready(self, response) -> bool:
        try:
            data = response.json()
            if data.get("status") == "ready":
                logger.info(f"{self.worker_id} status is ready")
                return True
            logger.warning(
                f"{self.worker_id} status is not ready: {data.get('status')}"
            )
        except ValueError:
            logger.warning(f"{self.worker_id} health response is not valid JSON")
        return False


@pytest.mark.timeout(350)
def test_decode_fallback_sglang(
    request,
    runtime_services_dynamic_ports,
    set_ucx_tls_no_mm,
    predownload_models,
    request_api,
    stream,
):
    """
    Test that the decode_only_fallback_total metric increments when
    the prefill worker goes down and requests fall back to decode-only.

    Setup: 1 prefill worker + 1 decode worker, no --enforce-disagg.
    """
    with DynamoFrontendProcess(request) as frontend:
        logger.info("Frontend started successfully")

        # Start decode worker first (registers model with frontend)
        with DynamoWorkerProcess(
            request, "worker0", frontend.frontend_port, disagg_mode="decode"
        ) as decode_worker:
            logger.info(f"Decode worker PID: {decode_worker.get_pid()}")

            with DynamoWorkerProcess(
                request, "worker1", frontend.frontend_port, disagg_mode="prefill"
            ) as prefill_worker:
                logger.info(f"Prefill worker PID: {prefill_worker.get_pid()}")

                run_decode_fallback_test(
                    frontend,
                    prefill_worker,
                    decode_worker,
                    use_chat=(request_api == "chat"),
                    stream=stream,
                )
