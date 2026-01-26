#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Determinism test for KVBM in aggregated mode.

To make sure KVBM's accuracy, this test suite checks if the model produces
deterministic outputs when same requests are served 1) without KVBM onboarded KV
blocks and 2) with KVBM onboarded KV blocks, when given the same inputs with
fixed seed and temperature=0.

The expected results should be 100% match between the two cases. Compared to
disaggregated mode, aggregated mode has less randomness chances.
"""

import logging
import os
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO

import pytest
import requests

from .common import DeterminismTester, ServerType
from .common import TestDeterminism as BaseTestDeterminism
from .common import check_module_available

HAS_VLLM_BENCH = check_module_available("vllm")

# Test markers to align with repository conventions
# Todo: enable the rest when kvbm is built in the ci
pytestmark = [
    pytest.mark.e2e,
    pytest.mark.slow,
    pytest.mark.gpu_1,
    pytest.mark.nightly,
]


def _find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class LLMServerManager:
    """Manages LLM server lifecycle for determinism testing."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        log_dir: Optional[Path] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        self.server_type = server_type
        # Use provided port, env var, or find a free port to avoid conflicts
        if port is not None:
            self.port = port
        elif os.environ.get("KVBM_SERVER_PORT"):
            self.port = int(os.environ["KVBM_SERVER_PORT"])
        else:
            self.port = _find_free_port()
        self.base_url = base_url or f"http://localhost:{self.port}"
        self.process: Optional[subprocess.Popen] = None
        self.cpu_cache_blocks = cpu_cache_blocks
        self.gpu_cache_blocks = gpu_cache_blocks

        # Prepare logging
        self.log_dir = log_dir or Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = (
            f"cpu{cpu_cache_blocks or 'default'}_gpu{gpu_cache_blocks or 'default'}"
        )
        self.server_log_file = (
            self.log_dir / f"{self.server_type}_server_{config_str}_{timestamp}.log"
        )
        self.server_stdout_file: Optional[TextIO] = None
        self._tee_threads: List[threading.Thread] = []

        # Environment for the process
        self.env = os.environ.copy()
        self.env.update(
            {
                "RUST_BACKTRACE": "1",
                # DynamoConnector connection settings
                "NATS_SERVER": "nats://localhost:4222",
                "ETCD_ENDPOINTS": "http://localhost:2379",
                # Enable KVBM metrics for monitoring offload/onboard
                "DYN_KVBM_METRICS": "true",
                "DYN_KVBM_METRICS_PORT": "6880",
                # Enable vLLM batch invariant for deterministic batching
                "VLLM_BATCH_INVARIANT": "1",
                "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
            }
        )

        # CPU cache blocks override via env
        if cpu_cache_blocks is not None:
            self.env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

        if self.server_type == ServerType.vllm:
            self._set_up_vllm_config(gpu_cache_blocks)
        elif self.server_type == ServerType.trtllm:
            self._set_up_trtllm_config(gpu_cache_blocks)
        else:
            raise ValueError(
                f"{self.server_type} is not supported yet in the KVBM test suite"
            )

    def _set_up_vllm_config(self, gpu_cache_blocks):
        self.env["VLLM_SERVER_DEV_MODE"] = "1"

        # Construct serve command
        self.server_cmd = [
            "vllm",
            "serve",
            "--block-size",
            "16",
            "--port",
            str(self.port),
            "--kv-transfer-config",
            '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "kvbm.vllm_integration.connector"}',
            os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--max-model-len",
            "8000",  # required to fit on L4 GPU when using 8b model
        ]

        # GPU blocks override
        if gpu_cache_blocks is not None:
            self.server_cmd.extend(["--num-gpu-blocks-override", str(gpu_cache_blocks)])

    def _set_up_trtllm_config(self, gpu_cache_blocks):
        config_path = os.environ.get(
            "KVBM_TRTLLM_LLMAPI_CONFIG_PATH", "/tmp/kvbm_llm_api_config.yaml"
        )
        llm_api_config: Dict[str, Any] = {}
        llm_api_config[
            "cuda_graph_config"
        ] = None  # explicitly disable CUDA graph since Connector API doesn't support CUDA graph yet in TRTLLM
        llm_api_config["kv_cache_config"] = {
            "enable_partial_reuse": False,
            "free_gpu_memory_fraction": 0.10,  # Set a small GPU fraction so that we can evict/reset the on-device kv cache faster
        }
        llm_api_config["kv_connector_config"] = {
            "connector_module": "kvbm.trtllm_integration.connector",
            "connector_scheduler_class": "DynamoKVBMConnectorLeader",
            "connector_worker_class": "DynamoKVBMConnectorWorker",
        }

        # GPU blocks override
        if gpu_cache_blocks is not None:
            del llm_api_config["kv_cache_config"]["free_gpu_memory_fraction"]
            llm_api_config["kv_cache_config"]["max_tokens"] = (
                int(gpu_cache_blocks) * 32
            )  # TRTLLM defaults 32 tokens per block

        # Construct serve command
        self.server_cmd = [
            "trtllm-serve",
            os.environ.get("KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"),
            "--host",
            "localhost",
            "--port",
            str(self.port),
            "--backend",
            "pytorch",
            "--extra_llm_api_options",
            config_path,
        ]

        import yaml

        with open(config_path, "w") as f:
            yaml.dump(llm_api_config, f, default_flow_style=False, sort_keys=False)

    def _tee_output(self, pipe: Any, log_file: TextIO, prefix: str) -> None:
        """Read from pipe and write to both log file and stdout (tee)."""
        try:
            for line in iter(pipe.readline, ""):
                if not line:
                    break
                # Write to log file
                log_file.write(line)
                log_file.flush()
                # Write to stdout with prefix
                sys.stdout.write(f"[{prefix}] {line}")
                sys.stdout.flush()
        except (ValueError, OSError):
            pass  # Pipe closed
        finally:
            pipe.close()

    def start_server(self, timeout: int = 300) -> bool:
        """Start LLM server and wait for readiness."""
        if self.is_server_running():
            self.stop_server()
            time.sleep(2)

        # Open log file (combined stdout+stderr)
        self.server_stdout_file = open(self.server_log_file.with_suffix(".log"), "w")

        # Write header
        header = f"=== {self.server_type} Server Started at {datetime.now()} ===\nCommand: {' '.join(self.server_cmd)}\n"
        self.server_stdout_file.write(header)
        self.server_stdout_file.flush()
        print(f"[{self.server_type}] {header}", end="")

        # Launch with pipe, redirect stderr to stdout
        self.process = subprocess.Popen(
            self.server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Redirect stderr to stdout
            env=self.env,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,  # Line buffered
        )

        # Start tee thread for combined output
        self._tee_threads = [
            threading.Thread(
                target=self._tee_output,
                args=(self.process.stdout, self.server_stdout_file, self.server_type),
                daemon=True,
            ),
        ]
        for t in self._tee_threads:
            t.start()

        # Wait for health
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                return True
            if self.process.poll() is not None:
                # Process exited, wait for tee thread to finish
                for t in self._tee_threads:
                    t.join(timeout=2)
                self._close_log_files()
                return False
            time.sleep(5)

        # Timeout
        self.stop_server()
        return False

    def stop_server(self):
        """Stop LLM server and close logs."""
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                try:
                    self.process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    self.process.wait()
            except (ProcessLookupError, OSError):
                pass
            finally:
                self.process = None
        # Wait for tee threads to finish
        for t in self._tee_threads:
            t.join(timeout=2)
        self._tee_threads = []
        self._close_log_files()

    def _close_log_files(self):
        if self.server_stdout_file:
            self.server_stdout_file.write(
                f"\n=== Server Stopped at {datetime.now()} ===\n"
            )
            self.server_stdout_file.close()
            self.server_stdout_file = None

    def is_server_running(self) -> bool:
        try:
            # First check basic health
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                return False

            # Then check if the model endpoint is ready with a simple test request
            test_payload = {
                "model": os.environ.get(
                    "KVBM_MODEL_ID", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
                ),
                "messages": [{"role": "user", "content": "test"}],
                "max_completion_tokens": 1,
                "temperature": 0,
            }

            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                headers={"Content-Type": "application/json"},
                json=test_payload,
                timeout=10,
            )
            return response.status_code == 200

        except requests.exceptions.RequestException:
            return False


class AggDeterminismTester(DeterminismTester):
    """Aggregated architecture specific determinism tester."""

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


@pytest.fixture(scope="function")
def llm_server(request, runtime_services):
    """Start and stop a LLM server for each test with optional cache block overrides.

    To parametrize, use:
      @pytest.mark.parametrize("llm_server", [{"cpu_blocks": 10000, "gpu_blocks": 2048}], indirect=True)
    """
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)

    # Put logs in the per-test directory set up by tests/conftest.py
    log_dir = Path(request.node.name)

    if check_module_available("vllm"):
        server_type = ServerType.vllm
    elif check_module_available("tensorrt_llm"):
        server_type = ServerType.trtllm
    else:
        raise Exception(
            "Neither the vllm nor the tensorrt_llm module is available in the current environment."
        )

    server_manager = LLMServerManager(
        port=port,
        cpu_cache_blocks=cpu_blocks,
        gpu_cache_blocks=gpu_blocks,
        log_dir=log_dir,
        server_type=server_type,
    )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "300"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start {server_type} server (cpu_blocks={cpu_blocks}, gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def tester(llm_server):
    """Create determinism tester bound to the running server's base URL."""
    t = AggDeterminismTester(
        base_url=llm_server.base_url,
        server_type=llm_server.server_type,
    )
    t.download_shakespeare_text()
    return t


class TestDeterminismAgg(BaseTestDeterminism):
    """Test class for determinism validation."""

    @pytest.mark.parametrize(
        "llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "10000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "2048")),
            },
        ],
        indirect=True,
    )
    @pytest.mark.kvbm
    def test_determinism_agg_with_cache_reset(
        self, tester, llm_server, runtime_services
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        # Call the base class implementation
        super().base_test_determinism_with_cache_reset(
            tester, llm_server, runtime_services
        )

    @pytest.mark.parametrize(
        "llm_server",
        [
            {
                "cpu_blocks": int(os.environ.get("KVBM_CPU_BLOCKS", "30000")),
                "gpu_blocks": int(os.environ.get("KVBM_GPU_BLOCKS", "2048")),
            },
        ],
        indirect=True,
    )
    @pytest.mark.kvbm_concurrency
    @pytest.mark.skipif(
        not HAS_VLLM_BENCH, reason="requires vllm bench (vllm module not found)"
    )
    def test_concurrent_determinism_under_load(
        self, tester, llm_server, runtime_services
    ):
        """Test Spanish prompt determinism under high concurrency load.

        Reproduces the bug where Spanish responses become English or corrupted.
        """
        # Get the Spanish prompt path relative to this test file
        spanish_prompt_path = Path(
            os.path.join(os.path.dirname(__file__), "es_prompt.txt")
        ).absolute()

        # Call the base class implementation
        super().base_test_spanish_prompt_determinism_under_load(
            tester, llm_server, runtime_services, spanish_prompt_path
        )


if __name__ == "__main__":
    # Allow running as script
    pytest.main([__file__, "-v", "-s"])
