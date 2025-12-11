#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in aggregated mode using Dynamo frontend/worker architecture.

This test suite validates KVBM determinism using the full Dynamo infrastructure:
- dynamo.frontend for HTTP API handling
- dynamo.vllm --connector kvbm for vLLM backend
- dynamo.trtllm with KVBM connector config for TRT-LLM backend

The expected results should be 100% match between requests with and without
KVBM onboarded KV blocks, when given the same inputs with fixed seed and temperature=0.

For standalone tests using `vllm serve` or `trtllm-serve` directly,
see test_determinism_agg_standalone.py.
"""

import importlib.util
import logging
import os
import shutil
import time
from typing import Any, Dict, Optional

import pytest
import requests

# Import testing utilities
from tests.utils.managed_process import ManagedProcess

from .common import DeterminismTester, ServerType
from .common import TestDeterminism as BaseTestDeterminism

# Test markers to align with repository conventions
pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
    pytest.mark.nightly,
]


# =============================================================================
# Dynamo Frontend/Worker Architecture Support
# =============================================================================


class DynamoFrontendProcess(ManagedProcess):
    """Process manager for Dynamo frontend in KVBM tests."""

    _logger = logging.getLogger(__name__)

    def __init__(self, request, port: int = 8000):
        command = [
            "python",
            "-m",
            "dynamo.frontend",
            "--http-port",
            str(port),
            "--router-mode",
            "round-robin",
        ]

        env = os.environ.copy()
        # Frontend doesn't use system metrics server
        env.pop("DYN_SYSTEM_PORT", None)

        log_dir = f"{request.node.name}_frontend"

        # Clean up any existing log directory from previous runs
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=True,
            health_check_ports=[port],
            log_dir=log_dir,
            timeout=120,
        )
        self.port = port
        self.base_url = f"http://localhost:{port}"


class DynamoVLLMWorkerProcess(ManagedProcess):
    """Process manager for Dynamo vLLM worker with KVBM connector."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        request,
        model: str,
        cpu_cache_gb: int = 1,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
    ):
        command = [
            "python",
            "-m",
            "dynamo.vllm",
            "--model",
            model,
            "--connector",
            "kvbm",
            "--enforce-eager",
            "--max-model-len",
            "8000",
        ]

        if gpu_cache_blocks is not None:
            command.extend(["--num-gpu-blocks-override", str(gpu_cache_blocks)])

        env = os.environ.copy()
        env["DYN_KVBM_CPU_CACHE_GB"] = str(cpu_cache_gb)
        env["RUST_BACKTRACE"] = "1"
        env["NATS_SERVER"] = "nats://localhost:4222"
        env["ETCD_ENDPOINTS"] = "http://localhost:2379"

        if cpu_cache_blocks is not None:
            env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

        log_dir = f"{request.node.name}_vllm_worker"
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=True,
            stragglers=["VLLM:EngineCore"],
            log_dir=log_dir,
            timeout=600,
        )


class DynamoTRTLLMWorkerProcess(ManagedProcess):
    """Process manager for Dynamo TRT-LLM worker with KVBM connector."""

    _logger = logging.getLogger(__name__)

    def __init__(
        self,
        request,
        model: str,
        cpu_cache_gb: int = 20,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
    ):
        # Generate KVBM config for TRT-LLM
        config_path = f"/tmp/kvbm_llm_api_config_{request.node.name}.yaml"
        self._generate_kvbm_config(config_path, gpu_cache_blocks)

        command = [
            "python",
            "-m",
            "dynamo.trtllm",
            "--model-path",
            model,
            "--served-model-name",
            model,
            "--extra-engine-args",
            config_path,
        ]

        env = os.environ.copy()
        env["DYN_KVBM_CPU_CACHE_GB"] = str(cpu_cache_gb)
        env["RUST_BACKTRACE"] = "1"
        env["NATS_SERVER"] = "nats://localhost:4222"
        env["ETCD_ENDPOINTS"] = "http://localhost:2379"

        if cpu_cache_blocks is not None:
            env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

        log_dir = f"{request.node.name}_trtllm_worker"
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        super().__init__(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=True,
            stragglers=["TRTLLM:EngineCore"],
            log_dir=log_dir,
            timeout=600,
        )

    def _generate_kvbm_config(
        self, config_path: str, gpu_cache_blocks: Optional[int] = None
    ):
        """Generate KVBM-enabled config YAML for TRT-LLM."""
        import yaml

        config: Dict[str, Any] = {
            # Disable CUDA graph (Connector API doesn't support it yet)
            "cuda_graph_config": None,
            "kv_cache_config": {
                "enable_partial_reuse": False,
                "free_gpu_memory_fraction": 0.10,
            },
            "kv_connector_config": {
                "connector_module": "kvbm.trtllm_integration.connector",
                "connector_scheduler_class": "DynamoKVBMConnectorLeader",
                "connector_worker_class": "DynamoKVBMConnectorWorker",
            },
        }

        if gpu_cache_blocks is not None:
            del config["kv_cache_config"]["free_gpu_memory_fraction"]
            config["kv_cache_config"]["max_tokens"] = int(gpu_cache_blocks) * 32

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


class DynamoKVBMServerManager:
    """Manages Dynamo frontend + worker lifecycle for KVBM determinism testing.

    This class manages both the frontend and worker processes together,
    using the Dynamo architecture for robust KVBM testing.
    """

    def __init__(
        self,
        request,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        cpu_cache_gb: int = 20,
        server_type: str = ServerType.vllm,
    ):
        self.request = request
        self.server_type = server_type
        self.port = port or int(os.environ.get("KVBM_SERVER_PORT", "8000"))
        self.base_url = base_url or f"http://localhost:{self.port}"
        self.cpu_cache_blocks = cpu_cache_blocks
        self.gpu_cache_blocks = gpu_cache_blocks
        self.cpu_cache_gb = cpu_cache_gb
        self.model = os.environ.get(
            "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )

        self.frontend_process: Optional[DynamoFrontendProcess] = None
        self.worker_process: Optional[ManagedProcess] = None

    def start_server(self, timeout: int = 600) -> bool:
        """Start Dynamo frontend and worker processes."""
        logger = logging.getLogger(__name__)
        logger.info(
            f"Starting Dynamo KVBM server (type={self.server_type}, model={self.model})"
        )

        try:
            # Start frontend first
            self.frontend_process = DynamoFrontendProcess(self.request, self.port)
            self.frontend_process.__enter__()
            logger.info(f"Frontend started on port {self.port}")

            # Start worker based on server type
            if self.server_type == ServerType.vllm:
                self.worker_process = DynamoVLLMWorkerProcess(
                    self.request,
                    self.model,
                    self.cpu_cache_gb,
                    self.cpu_cache_blocks,
                    self.gpu_cache_blocks,
                )
            elif self.server_type == ServerType.trtllm:
                self.worker_process = DynamoTRTLLMWorkerProcess(
                    self.request,
                    self.model,
                    self.cpu_cache_gb,
                    self.cpu_cache_blocks,
                    self.gpu_cache_blocks,
                )
            else:
                raise ValueError(f"Unsupported server type: {self.server_type}")

            self.worker_process.__enter__()
            logger.info(f"{self.server_type} worker started")

            # Wait for the server to be fully ready
            return self._wait_for_ready(timeout)

        except Exception as e:
            logger.error(f"Failed to start Dynamo KVBM server: {e}")
            self.stop_server()
            return False

    def _wait_for_ready(self, timeout: int = 600) -> bool:
        """Wait for the server to be fully ready to serve requests."""
        logger = logging.getLogger(__name__)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Check /v1/models endpoint
                response = requests.get(f"{self.base_url}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
                        # Try a test request
                        test_payload = {
                            "model": self.model,
                            "messages": [{"role": "user", "content": "test"}],
                            "max_completion_tokens": 1,
                            "temperature": 0,
                        }
                        response = requests.post(
                            f"{self.base_url}/v1/chat/completions",
                            json=test_payload,
                            timeout=30,
                        )
                        if response.status_code == 200:
                            logger.info("Dynamo KVBM server is ready")
                            return True
            except requests.exceptions.RequestException:
                pass

            time.sleep(5)
            elapsed = time.time() - start_time
            if int(elapsed) % 30 == 0:
                logger.info(
                    f"Still waiting for server... ({elapsed:.0f}s / {timeout}s)"
                )

        logger.error(f"Server did not become ready within {timeout}s")
        return False

    def stop_server(self):
        """Stop Dynamo frontend and worker processes."""
        logger = logging.getLogger(__name__)

        if self.worker_process:
            try:
                self.worker_process.__exit__(None, None, None)
                logger.info("Worker process stopped")
            except Exception as e:
                logger.warning(f"Error stopping worker: {e}")
            self.worker_process = None

        if self.frontend_process:
            try:
                self.frontend_process.__exit__(None, None, None)
                logger.info("Frontend process stopped")
            except Exception as e:
                logger.warning(f"Error stopping frontend: {e}")
            self.frontend_process = None

    def is_server_running(self) -> bool:
        """Check if the server is responding to requests."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False


class AggDeterminismTester(DeterminismTester):
    """Aggregated architecture specific determinism tester for Dynamo tests."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id, server_type)

    def reset_prefix_cache(self):
        """Reset the prefix cache."""
        print("Resetting prefix cache...")
        if self.server_type == ServerType.trtllm:
            # TRTLLM doesn't support reset_prefix_cache endpoint API
            # 300 shakespeare content could evict the 0.1 x 80G (~1700 blocks) on-device cache
            shakespeare_count = 300
            for seq_idx in range(1, shakespeare_count + 1):
                start_word = (seq_idx - 1) * self.word_count
                content = self.get_shakespeare_content(start_word)

                if content:
                    print(
                        f"Resetting Shakespeare sequence {seq_idx} (words {start_word}-{start_word + self.word_count - 1})..."
                    )
                    try:
                        self.make_request(content)
                    except Exception as e:
                        print(f"Resetting request failed: {e}")
        else:
            response = requests.post(
                f"{self.base_url}/reset_prefix_cache",
                timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
            )
            response.raise_for_status()
        print("Cache reset done")


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="function")
def dynamo_llm_server(request, runtime_services):
    """Start and stop a Dynamo-based LLM server for each test.

    This fixture uses Dynamo frontend + worker architecture for robust KVBM testing.

    To parametrize, use:
      @pytest.mark.parametrize("dynamo_llm_server", [{"cpu_blocks": 10000, "gpu_blocks": 2048}], indirect=True)
    """
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)
    cpu_cache_gb = getattr(request, "param", {}).get("cpu_cache_gb", 1)

    if importlib.util.find_spec("vllm") is not None:
        server_type = ServerType.vllm
    elif importlib.util.find_spec("tensorrt_llm") is not None:
        server_type = ServerType.trtllm
    else:
        raise Exception(
            "Neither the vllm nor the tensorrt_llm module is available in the current environment."
        )

    server_manager = DynamoKVBMServerManager(
        request=request,
        port=port,
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        cpu_cache_gb=cpu_cache_gb,
        server_type=server_type,
    )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "600"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start Dynamo {server_type} server (cpu_blocks={cpu_blocks}, gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def dynamo_tester(dynamo_llm_server):
    """Create determinism tester bound to the Dynamo server's base URL."""
    t = AggDeterminismTester(
        base_url=dynamo_llm_server.base_url,
        server_type=dynamo_llm_server.server_type,
    )
    t.download_shakespeare_text()
    return t


# =============================================================================
# Test Classes
# =============================================================================


class TestDeterminismAggDynamo(BaseTestDeterminism):
    """Test class for determinism validation using Dynamo frontend/worker architecture.

    These tests validate KVBM determinism using the full Dynamo infrastructure:
    - dynamo.frontend for HTTP API handling
    - dynamo.vllm --connector kvbm for vLLM backend
    - dynamo.trtllm with KVBM connector config for TRT-LLM backend
    """

    @pytest.mark.parametrize(
        "dynamo_llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    def test_determinism_dynamo_with_cache_reset(
        self, dynamo_tester, dynamo_llm_server, runtime_services
    ):
        """Test determinism with Dynamo frontend/worker architecture."""
        super().base_test_determinism_with_cache_reset(
            dynamo_tester, dynamo_llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "dynamo_llm_server",
        [
            {"cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000"))},
        ],
        indirect=True,
    )
    @pytest.mark.kvbm_v2
    def test_determinism_dynamo_with_cache_reset_v2(
        self, dynamo_tester, dynamo_llm_server, runtime_services, monkeypatch
    ):
        """Test determinism with Dynamo architecture and V2 transfer."""
        monkeypatch.setenv("DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL", "1")
        super().base_test_determinism_with_cache_reset(
            dynamo_tester, dynamo_llm_server, runtime_services
        )


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])
