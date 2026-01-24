# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Shadow Engine Failover Test for vLLM.

Environment Variables:
    GPU_MEMORY_SERVICE_TEST_MODEL: Model to use (default: Qwen/Qwen3-0.6B)
    GPU_MEMORY_SERVICE_TP: Tensor parallelism (default: 1)
"""

import logging
import os
import shutil
import time

import pynvml
import pytest
import requests

from tests.utils.constants import QWEN
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess
from tests.utils.port_utils import allocate_port, deallocate_ports

logger = logging.getLogger(__name__)

MODEL = os.environ.get("GPU_MEMORY_SERVICE_TEST_MODEL", QWEN)
TP = int(os.environ.get("GPU_MEMORY_SERVICE_TP", "1"))
SOCKET_PATH = "/tmp/gpu_memory_service_{device}.sock"


def bytes_to_mb(b: int) -> float:
    return b / (1024 * 1024)


def get_gpu_memory_used(device: int = 0) -> int:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()


class GMSServerProcess(ManagedProcess):
    """GPU Memory Service server process."""

    def __init__(self, request, device: int):
        self.device = device
        self.socket_path = SOCKET_PATH.format(device=device)

        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)

        log_dir = f"/tmp/{request.node.name}_gms_{device}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=["python3", "-m", "gpu_memory_service", "--device", str(device)],
            env={**os.environ, "DYN_LOG": "debug"},
            timeout=60,
            display_output=True,
            terminate_existing=False,
            log_dir=log_dir,
            health_check_funcs=[self._socket_ready],
        )

    def _socket_ready(self, timeout: float = 30) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            if os.path.exists(self.socket_path):
                return True
            time.sleep(0.1)
        return False


class VLLMWithGMSProcess(ManagedProcess):
    """vLLM engine with GPU Memory Service."""

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        kv_event_port: int = 20080,
        nixl_port: int = 20096,
    ):
        self.engine_id = engine_id
        self.system_port = system_port

        log_dir = f"/tmp/{request.node.name}_{engine_id}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=[
                "python3",
                "-m",
                "dynamo.vllm",
                "--model",
                MODEL,
                "-tp",
                str(TP),
                "--load-format",
                "gms",
                "--enable-sleep-mode",
                "--gpu-memory-utilization",
                "0.8",
            ],
            env={
                **os.environ,
                "DYN_LOG": "debug",
                "DYN_SYSTEM_PORT": str(system_port),
                "DYN_VLLM_KV_EVENT_PORT": str(kv_event_port),
                "VLLM_NIXL_SIDE_CHANNEL_PORT": str(nixl_port),
            },
            health_check_urls=[
                (f"http://localhost:{system_port}/health", self._is_ready)
            ],
            timeout=300,
            display_output=True,
            terminate_existing=False,
            stragglers=[],
            log_dir=log_dir,
        )

    def _is_ready(self, response) -> bool:
        try:
            return response.json().get("status") == "ready"
        except ValueError:
            return False

    def sleep(self) -> dict:
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/sleep",
            json={"level": 1},
            timeout=30,
        )
        r.raise_for_status()
        logger.info(f"{self.engine_id} sleep: {r.json()}")
        return r.json()

    def wake(self) -> dict:
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/wake", json={}, timeout=30
        )
        r.raise_for_status()
        logger.info(f"{self.engine_id} wake: {r.json()}")
        return r.json()


def send_completion(port: int, prompt: str = "Hello") -> dict:
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 20},
        timeout=120,
    )
    r.raise_for_status()
    result = r.json()
    assert result.get("choices"), "No choices in response"
    return result


@pytest.fixture
def gms_failover_ports():
    ports = [allocate_port(p) for p in [8100, 8101, 8200, 20080, 20081, 20096, 20097]]
    yield {
        "shadow_system": ports[0],
        "primary_system": ports[1],
        "frontend": ports[2],
        "shadow_kv_event": ports[3],
        "primary_kv_event": ports[4],
        "shadow_nixl": ports[5],
        "primary_nixl": ports[6],
    }
    deallocate_ports(ports)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_2
@pytest.mark.fault_tolerance
@pytest.mark.nightly
@pytest.mark.model(MODEL)
@pytest.mark.timeout(600)
def test_gms_shadow_engine_failover(
    request, runtime_services, gms_failover_ports, predownload_models
):
    """
    Test shadow engine failover with GPU Memory Service.

    1. Start shadow engine and put it to sleep
    2. Start primary engine and serve inference
    3. Kill primary engine
    4. Wake shadow engine and verify it handles inference
    """
    ports = gms_failover_ports
    gms_servers = []

    try:
        # Start GMS servers
        for device in range(TP):
            gms = GMSServerProcess(request, device)
            gms.__enter__()
            gms_servers.append(gms)

        with DynamoFrontendProcess(request, frontend_port=ports["frontend"]):
            # Start shadow engine
            shadow = VLLMWithGMSProcess(
                request,
                "shadow",
                ports["shadow_system"],
                ports["shadow_kv_event"],
                ports["shadow_nixl"],
            )
            with shadow:
                time.sleep(5)

                # Verify shadow works
                send_completion(ports["frontend"])
                logger.info("Shadow inference OK")

                # Sleep shadow
                mem_before = get_gpu_memory_used()
                assert shadow.sleep()["status"] == "ok"
                time.sleep(2)

                mem_after_sleep = get_gpu_memory_used()
                logger.info(
                    f"Shadow sleep freed {bytes_to_mb(mem_before - mem_after_sleep):.0f} MB"
                )
                assert mem_after_sleep < mem_before

                # Start primary engine
                primary = VLLMWithGMSProcess(
                    request,
                    "primary",
                    ports["primary_system"],
                    ports["primary_kv_event"],
                    ports["primary_nixl"],
                )
                with primary:
                    time.sleep(5)
                    send_completion(ports["frontend"], "Primary test")
                    logger.info("Primary inference OK")

                # Primary is dead
                time.sleep(5)

                # Wake shadow
                assert shadow.wake()["status"] == "ok"
                time.sleep(5)

                # Verify shadow handles failover
                send_completion(ports["frontend"], "After failover")
                logger.info("Shadow handles failover OK")

                for i in range(3):
                    send_completion(ports["frontend"], f"Verify {i}")
                logger.info("All verification passed")

    finally:
        for gms in reversed(gms_servers):
            try:
                gms.__exit__(None, None, None)
            except Exception:
                pass

        for device in range(TP):
            path = SOCKET_PATH.format(device=device)
            if os.path.exists(path):
                os.unlink(path)
