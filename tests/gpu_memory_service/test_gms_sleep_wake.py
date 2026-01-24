# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
GPU Memory Service Basic Sleep/Wake Test for vLLM.

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

    def __init__(self, request, engine_id: str, system_port: int):
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
            env={**os.environ, "DYN_LOG": "debug", "DYN_SYSTEM_PORT": str(system_port)},
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
        return r.json()

    def wake(self) -> dict:
        r = requests.post(
            f"http://localhost:{self.system_port}/engine/wake", json={}, timeout=30
        )
        r.raise_for_status()
        return r.json()


def send_completion(port: int, prompt: str = "Hello") -> dict:
    r = requests.post(
        f"http://localhost:{port}/v1/completions",
        json={"model": MODEL, "prompt": prompt, "max_tokens": 20},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


@pytest.fixture
def gms_ports():
    ports = [allocate_port(p) for p in [8100, 8200]]
    yield {"system": ports[0], "frontend": ports[1]}
    deallocate_ports(ports)


@pytest.mark.vllm
@pytest.mark.e2e
@pytest.mark.gpu_1
@pytest.mark.fault_tolerance
@pytest.mark.model(MODEL)
@pytest.mark.timeout(300)
def test_gms_basic_sleep_wake(request, runtime_services, gms_ports, predownload_models):
    """Test basic sleep/wake with GPU Memory Service."""
    ports = gms_ports
    gms_servers = []

    try:
        # Start GMS servers
        for device in range(TP):
            gms = GMSServerProcess(request, device)
            gms.__enter__()
            gms_servers.append(gms)

        with DynamoFrontendProcess(request, frontend_port=ports["frontend"]):
            with VLLMWithGMSProcess(request, "engine", ports["system"]) as engine:
                time.sleep(5)

                # Initial inference
                assert send_completion(ports["frontend"])["choices"]

                mem_before = get_gpu_memory_used()
                logger.info(f"Memory before sleep: {bytes_to_mb(mem_before):.0f} MB")

                # Sleep
                assert engine.sleep()["status"] == "ok"
                time.sleep(2)

                mem_after_sleep = get_gpu_memory_used()
                logger.info(
                    f"Memory after sleep: {bytes_to_mb(mem_after_sleep):.0f} MB"
                )
                assert mem_after_sleep < mem_before, "Sleep should reduce memory"

                # Wake
                assert engine.wake()["status"] == "ok"
                time.sleep(2)

                # Inference after wake
                assert send_completion(ports["frontend"], "Goodbye")["choices"]

                logger.info(
                    f"Memory freed: {bytes_to_mb(mem_before - mem_after_sleep):.0f} MB"
                )

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
