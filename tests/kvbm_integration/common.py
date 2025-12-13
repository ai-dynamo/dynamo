#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Common functionality for KVBM determinism tests.

This module contains shared classes and functions used by both
aggregated and disaggregated determinism tests.
"""

import importlib.util
import logging
import os
import re
import signal
import subprocess
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple, Union

import pytest
import requests


def check_logs_for_patterns(
    log_path: Path, patterns: List[str], process_name: str
) -> List[str]:
    """Check log file for specific patterns (errors, warnings, etc.)."""
    findings = []

    if not log_path.exists():
        return [f"{process_name} log file not found at {log_path}"]

    try:
        with open(log_path, "r") as f:
            content = f.read()

            for pattern in patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                if matches:
                    # Limit to first 3 matches and truncate each to 200 chars
                    for match in matches[:3]:
                        match_str = match if isinstance(match, str) else str(match)
                        findings.append(f"{process_name}: {match_str[:200]}")
    except Exception as e:
        findings.append(f"Error reading {process_name} log: {e}")

    return findings


class ApiTester:
    """Base class for making API requests to LLM endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
    ):
        self.base_url = (
            base_url or os.environ.get("DYNAMO_API_BASE_URL") or "http://localhost:8000"
        )
        self.model_id = model_id or os.environ.get("KVBM_MODEL_ID") or "Qwen/Qwen3-0.6B"

    def make_request(
        self,
        content: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Make API request and return completion text."""
        payload = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": content},
            ],
            "stream": False,
            "temperature": temperature,
            "seed": seed,
        }

        # Add max_tokens with appropriate key based on kwargs or defaults
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        elif "max_completion_tokens" in kwargs:
            payload["max_completion_tokens"] = kwargs.pop("max_completion_tokens")
        else:
            payload["max_completion_tokens"] = int(
                os.environ.get("KVBM_MAX_TOKENS", "48")
            )

        # Add any additional kwargs
        payload.update(kwargs)

        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=int(os.environ.get("KVBM_HTTP_TIMEOUT", "30")),
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def send_chat_request(
        self,
        messages: List[dict],
        max_tokens: int = 50,
        temperature: float = 0.0,
        seed: int = 42,
    ) -> dict:
        """Send a chat request and return full response JSON."""
        url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "seed": seed,
        }

        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()


class ServerType(str, Enum):
    vllm = "vllm"
    trtllm = "trtllm"


# =============================================================================
# Server Manager Base Class and Implementations
# =============================================================================


class BaseServerManager(ABC):
    """Abstract base class for LLM server lifecycle management.

    Provides common functionality for starting/stopping servers, health checks,
    and logging. Subclasses implement specific server architectures.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        log_dir: Optional[Path] = None,
        server_type: Optional[str] = ServerType.vllm,
        cpu_cache_gb: int = 1,
    ):
        self.server_type = server_type
        self.port = port or int(os.environ.get("KVBM_SERVER_PORT", "8000"))
        self.base_url = base_url or f"http://localhost:{self.port}"
        self.cpu_cache_blocks = cpu_cache_blocks
        self.gpu_cache_blocks = gpu_cache_blocks
        self.cpu_cache_gb = cpu_cache_gb
        self.model = os.environ.get("KVBM_MODEL_ID", "Qwen/Qwen3-0.6B")

        # Prepare logging
        self.log_dir = log_dir or Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.config_str = (
            f"cpu{cpu_cache_blocks or 'default'}_gpu{gpu_cache_blocks or 'default'}"
        )

        # Common environment setup
        self.env = os.environ.copy()
        self.env.update(
            {
                "RUST_BACKTRACE": "1",
                "NATS_SERVER": "nats://localhost:4222",
                "ETCD_ENDPOINTS": "http://localhost:2379",
            }
        )

        # CPU cache blocks override via env
        if cpu_cache_blocks is not None:
            self.env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(cpu_cache_blocks)

    @abstractmethod
    def start_server(self, timeout: int = 300) -> bool:
        """Start the server and wait for readiness."""
        pass

    @abstractmethod
    def stop_server(self):
        """Stop the server and clean up resources."""
        pass

    def is_server_running(self) -> bool:
        """Check if the server is responding to requests."""
        try:
            # First check basic health
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                return False

            # Then check if the model endpoint is ready with a simple test request
            test_payload = {
                "model": self.model,
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

    def _download_model(self):
        """Try to download the model using hf_transfer."""
        print("Attempting model download...")
        try:
            subprocess.run(
                f"pip install hf_transfer && HF_HUB_ENABLE_HF_TRANSFER=1 hf download {self.model}",
                check=True,
                shell=True,
            )
        except subprocess.CalledProcessError:
            print("Model download failed. Is this a locally stored model?")

    def _terminate_process(self, process: Optional[subprocess.Popen], name: str = ""):
        """Safely terminate a subprocess."""
        if process is None:
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait()
        except (ProcessLookupError, OSError):
            pass

    def _close_log_file(self, file: Optional[TextIO], message: str = ""):
        """Safely close a log file with optional closing message."""
        if file:
            if message:
                file.write(f"\n=== {message} at {datetime.now()} ===\n")
            file.close()

    @staticmethod
    def detect_server_type() -> ServerType:
        """Detect available server type based on installed modules."""
        if importlib.util.find_spec("vllm") is not None:
            return ServerType.vllm
        elif importlib.util.find_spec("tensorrt_llm") is not None:
            return ServerType.trtllm
        else:
            raise RuntimeError(
                "Neither the vllm nor the tensorrt_llm module is available "
                "in the current environment."
            )


class StandaloneServerManager(BaseServerManager):
    """Manages standalone LLM server lifecycle (vllm serve / trtllm-serve).

    This class manages `vllm serve` or `trtllm-serve` processes directly,
    without using the Dynamo frontend/worker architecture.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        log_dir: Optional[Path] = None,
        server_type: Optional[str] = ServerType.vllm,
        cpu_cache_gb: int = 1,
    ):
        super().__init__(
            base_url=base_url,
            port=port,
            cpu_cache_blocks=cpu_cache_blocks,
            gpu_cache_blocks=gpu_cache_blocks,
            log_dir=log_dir,
            server_type=server_type,
            cpu_cache_gb=cpu_cache_gb,
        )

        self.process: Optional[subprocess.Popen] = None
        self.server_log_file = (
            self.log_dir
            / f"{self.server_type}_server_{self.config_str}_{self.timestamp}.log"
        )
        self.server_stdout_file: Optional[TextIO] = None
        self.server_stderr_file: Optional[TextIO] = None
        self.server_cmd: List[str] = []

        if self.server_type == ServerType.vllm:
            self._set_up_vllm_config()
        elif self.server_type == ServerType.trtllm:
            self._set_up_trtllm_config()
        else:
            raise ValueError(
                f"{self.server_type} is not supported yet in the KVBM test suite"
            )

    def _set_up_vllm_config(self):
        """Configure vllm serve command."""
        self.env["VLLM_SERVER_DEV_MODE"] = "1"

        self.server_cmd = [
            "vllm",
            "serve",
            "--block-size",
            "16",
            "--port",
            str(self.port),
            "--kv-transfer-config",
            '{"kv_connector":"DynamoConnector","kv_role":"kv_both", "kv_connector_module_path": "kvbm.vllm_integration.connector"}',
            self.model,
            "--max-model-len",
            "8000",
        ]

        if self.gpu_cache_blocks is not None:
            self.server_cmd.extend(
                ["--num-gpu-blocks-override", str(self.gpu_cache_blocks)]
            )

    def _set_up_trtllm_config(self):
        """Configure trtllm-serve command."""
        config_path = os.environ.get(
            "KVBM_TRTLLM_LLMAPI_CONFIG_PATH", "/tmp/kvbm_llm_api_config.yaml"
        )
        llm_api_config: Dict[str, Any] = {
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

        if self.gpu_cache_blocks is not None:
            del llm_api_config["kv_cache_config"]["free_gpu_memory_fraction"]
            llm_api_config["kv_cache_config"]["max_tokens"] = (
                int(self.gpu_cache_blocks) * 32
            )

        self.server_cmd = [
            "trtllm-serve",
            self.model,
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

    def start_server(self, timeout: int = 300) -> bool:
        """Start LLM server and wait for readiness."""
        if self.is_server_running():
            self.stop_server()
            time.sleep(2)

        # Open log files
        self.server_stdout_file = open(
            self.server_log_file.with_suffix(".stdout.log"), "w"
        )
        self.server_stderr_file = open(
            self.server_log_file.with_suffix(".stderr.log"), "w"
        )
        if self.server_stdout_file is not None:
            self.server_stdout_file.write(
                f"=== {self.server_type} Server Started at {datetime.now()} ===\n"
                f"Command: {' '.join(self.server_cmd)}\n"
            )
            self.server_stdout_file.flush()

        self._download_model()

        # Launch
        self.process = subprocess.Popen(
            self.server_cmd,
            stdout=self.server_stdout_file,
            stderr=self.server_stderr_file,
            env=self.env,
            preexec_fn=os.setsid,
        )

        # Wait for health
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                return True
            if self.process.poll() is not None:
                self._close_log_files()
                return False
            time.sleep(5)

        # Timeout
        self.stop_server()
        return False

    def stop_server(self):
        """Stop LLM server and close logs."""
        self._terminate_process(self.process, "server")
        self.process = None
        self._close_log_files()

    def _close_log_files(self):
        """Close server log files."""
        self._close_log_file(self.server_stdout_file, "Server Stopped")
        self.server_stdout_file = None
        self._close_log_file(self.server_stderr_file)
        self.server_stderr_file = None


# Import ManagedProcess lazily to avoid circular imports
def _get_managed_process():
    """Lazily import ManagedProcess for Dynamo server management."""
    try:
        from tests.utils.managed_process import ManagedProcess

        return ManagedProcess
    except ImportError:
        return None


class DynamoFrontendProcess:
    """Process manager for Dynamo frontend in KVBM tests."""

    _logger = logging.getLogger(__name__)

    def __init__(self, request, port: int = 8000):
        import shutil

        ManagedProcess = _get_managed_process()
        if ManagedProcess is None:
            raise ImportError(
                "ManagedProcess not available - required for Dynamo server"
            )

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
        env.pop("DYN_SYSTEM_PORT", None)

        log_dir = f"{request.node.name}_frontend"
        try:
            shutil.rmtree(log_dir)
        except FileNotFoundError:
            pass

        # Store as instance attribute for delegation
        self._process = ManagedProcess(
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

    def __enter__(self):
        return self._process.__enter__()

    def __exit__(self, *args):
        return self._process.__exit__(*args)


class DynamoVLLMWorkerProcess:
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
        import shutil

        ManagedProcess = _get_managed_process()
        if ManagedProcess is None:
            raise ImportError(
                "ManagedProcess not available - required for Dynamo server"
            )

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

        self._process = ManagedProcess(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=False,  # Don't kill the frontend!
            stragglers=["VLLM:EngineCore"],
            log_dir=log_dir,
            timeout=600,
        )

    def __enter__(self):
        return self._process.__enter__()

    def __exit__(self, *args):
        return self._process.__exit__(*args)


class DynamoTRTLLMWorkerProcess:
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
        import shutil

        ManagedProcess = _get_managed_process()
        if ManagedProcess is None:
            raise ImportError(
                "ManagedProcess not available - required for Dynamo server"
            )

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

        self._process = ManagedProcess(
            command=command,
            env=env,
            display_output=True,
            terminate_existing=False,  # Don't kill the frontend!
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

    def __enter__(self):
        return self._process.__enter__()

    def __exit__(self, *args):
        return self._process.__exit__(*args)


class DynamoServerManager(BaseServerManager):
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
        log_dir: Optional[Path] = None,
        server_type: str = ServerType.vllm,
        cpu_cache_gb: int = 1,
    ):
        super().__init__(
            base_url=base_url,
            port=port,
            cpu_cache_blocks=cpu_cache_blocks,
            gpu_cache_blocks=gpu_cache_blocks,
            log_dir=log_dir,
            server_type=server_type,
            cpu_cache_gb=cpu_cache_gb,
        )
        self.request = request
        self.frontend_process: Optional[DynamoFrontendProcess] = None
        self.worker_process: Optional[
            Union["DynamoVLLMWorkerProcess", "DynamoTRTLLMWorkerProcess"]
        ] = None

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
                response = requests.get(f"{self.base_url}/v1/models", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    if data.get("data") and len(data["data"]) > 0:
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


class DisaggServerManager(BaseServerManager):
    """Manages disaggregated LLM server lifecycle (frontend + prefiller + decoder).

    This class manages a 3-process disaggregated architecture:
    - Dynamo frontend for routing
    - Prefiller worker with KVBM connector
    - Decoder worker with nixl connector
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        port: Optional[int] = None,
        cpu_cache_blocks: Optional[int] = None,
        gpu_cache_blocks: Optional[int] = None,
        log_dir: Optional[Path] = None,
        server_type: Optional[str] = ServerType.vllm,
        cpu_cache_gb: int = 1,
    ):
        super().__init__(
            base_url=base_url,
            port=port,
            cpu_cache_blocks=cpu_cache_blocks,
            gpu_cache_blocks=gpu_cache_blocks,
            log_dir=log_dir,
            server_type=server_type,
            cpu_cache_gb=cpu_cache_gb,
        )

        self.process_frontend: Optional[subprocess.Popen] = None
        self.process_prefiller: Optional[subprocess.Popen] = None
        self.process_decoder: Optional[subprocess.Popen] = None

        # Log files
        self.prefiller_log_file = (
            self.log_dir
            / f"{self.server_type}_prefiller_{self.config_str}_{self.timestamp}.log"
        )
        self.prefiller_stdout_file: Optional[TextIO] = None
        self.prefiller_stderr_file: Optional[TextIO] = None

        self.decoder_log_file = (
            self.log_dir / f"{self.server_type}_decoder_{self.timestamp}.log"
        )
        self.decoder_stdout_file: Optional[TextIO] = None
        self.decoder_stderr_file: Optional[TextIO] = None

        # Commands
        self.dynamo_frontend_cmd: List[str] = []
        self.prefiller_cmd: List[str] = []
        self.decoder_cmd: List[str] = []

        self._set_up_dynamo_config()

        if self.server_type == ServerType.vllm:
            self._set_up_vllm_config()
        else:
            raise ValueError(
                f"{self.server_type} is not supported yet in disaggregated mode"
            )

    def _set_up_dynamo_config(self, router_mode: str = "kv"):
        """Configure Dynamo frontend command."""
        self.dynamo_frontend_cmd = [
            "python3",
            "-m",
            "dynamo.frontend",
            "--router-mode",
            router_mode,
            "--http-port",
            str(self.port),
        ]

    def _set_up_vllm_config(self):
        """Configure vLLM prefiller and decoder commands."""
        self.env["VLLM_SERVER_DEV_MODE"] = "1"

        # Decoder command
        self.decoder_cmd = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            self.model,
            "--block-size",
            "16",
            "--max-model-len",
            "8000",
            "--connector",
            "nixl",
        ]

        # Prefiller command
        self.prefiller_cmd = [
            "python3",
            "-m",
            "dynamo.vllm",
            "--model",
            self.model,
            "--is-prefill-worker",
            "--block-size",
            "16",
            "--max-model-len",
            "8000",
            "--connector",
            "kvbm",
            "nixl",
        ]

        # GPU blocks override
        if self.gpu_cache_blocks is not None:
            self.decoder_cmd.extend(
                ["--num-gpu-blocks-override", str(self.gpu_cache_blocks)]
            )
            self.prefiller_cmd.extend(
                ["--num-gpu-blocks-override", str(self.gpu_cache_blocks)]
            )

    def start_server(self, timeout: int = 300) -> bool:
        """Start disaggregated server (frontend + prefiller + decoder)."""
        if self.is_server_running():
            self.stop_server()
            time.sleep(5)

        # Open log files
        self.prefiller_stdout_file = open(
            self.prefiller_log_file.with_suffix(".stdout.log"), "w"
        )
        self.prefiller_stderr_file = open(
            self.prefiller_log_file.with_suffix(".stderr.log"), "w"
        )
        if self.prefiller_stdout_file:
            self.prefiller_stdout_file.write(
                f"=== {self.server_type} Prefiller Started at {datetime.now()} ===\n"
                f"Command: {' '.join(self.prefiller_cmd)}\n"
            )
            self.prefiller_stdout_file.flush()

        self.decoder_stdout_file = open(
            self.decoder_log_file.with_suffix(".stdout.log"), "w"
        )
        self.decoder_stderr_file = open(
            self.decoder_log_file.with_suffix(".stderr.log"), "w"
        )
        if self.decoder_stdout_file:
            self.decoder_stdout_file.write(
                f"=== {self.server_type} Decoder Started at {datetime.now()} ===\n"
                f"Command: {' '.join(self.decoder_cmd)}\n"
            )
            self.decoder_stdout_file.flush()

        # Create separate environment configs for different processes
        decoder_env = self.env.copy()
        decoder_env["CUDA_VISIBLE_DEVICES"] = "0"

        prefiller_env = self.env.copy()
        prefiller_env["CUDA_VISIBLE_DEVICES"] = "1"

        # Launch frontend first
        self.process_frontend = subprocess.Popen(
            self.dynamo_frontend_cmd,
            env=self.env,
            preexec_fn=os.setsid,
        )
        print(f"Frontend process started with PID: {self.process_frontend.pid}")

        # Give frontend time to start up
        time.sleep(5)

        self._download_model()

        # Launch decoder
        self.process_decoder = subprocess.Popen(
            self.decoder_cmd,
            stdout=self.decoder_stdout_file,
            stderr=self.decoder_stderr_file,
            env=decoder_env,
            preexec_fn=os.setsid,
        )
        print(f"Decoder process started with PID: {self.process_decoder.pid}")

        # Launch prefiller
        self.process_prefiller = subprocess.Popen(
            self.prefiller_cmd,
            stdout=self.prefiller_stdout_file,
            stderr=self.prefiller_stderr_file,
            env=prefiller_env,
            preexec_fn=os.setsid,
        )
        print(f"Prefiller process started with PID: {self.process_prefiller.pid}")

        # Give time to start up
        print(
            "Sleeping for 30 seconds to wait for decoder and prefiller to start up..."
        )
        time.sleep(30)

        # Wait for health
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                if self.is_server_running():
                    return True
                if (
                    self.process_frontend.poll() is not None
                    or self.process_prefiller.poll() is not None
                    or self.process_decoder.poll() is not None
                ):
                    self.stop_server()
                    return False
            except Exception as e:
                print(f"Error checking server status: {e}")

            print(
                f"Waiting for server to start up: timeout: {timeout}, elapsed: {int(time.time() - start_time)}"
            )
            time.sleep(5)

        # Timeout
        self.stop_server()
        return False

    def stop_server(self):
        """Stop all disaggregated server processes."""
        self._terminate_process(self.process_frontend, "frontend")
        self.process_frontend = None

        self._terminate_process(self.process_prefiller, "prefiller")
        self.process_prefiller = None

        self._terminate_process(self.process_decoder, "decoder")
        self.process_decoder = None

        self._close_log_files()

    def _close_log_files(self):
        """Close all log files."""
        self._close_log_file(self.prefiller_stdout_file, "Prefiller Stopped")
        self.prefiller_stdout_file = None
        self._close_log_file(self.prefiller_stderr_file)
        self.prefiller_stderr_file = None

        self._close_log_file(self.decoder_stdout_file, "Decoder Stopped")
        self.decoder_stdout_file = None
        self._close_log_file(self.decoder_stderr_file)
        self.decoder_stderr_file = None


class DeterminismTester(ApiTester):
    """Test class for model determinism validation."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id)
        self.server_type = server_type

        self.shakespeare_file = Path("t8.shakespeare.txt")
        self.max_iterations = int(os.environ.get("KVBM_MAX_ITERATIONS", "10"))
        self.word_count = int(os.environ.get("KVBM_WORD_COUNT", "200"))

        # Test intervals
        self.control_interval = int(os.environ.get("KVBM_CONTROL_INTERVAL", "10"))
        self.shakespeare_interval = int(
            os.environ.get("KVBM_SHAKESPEARE_INTERVAL", "1")
        )
        self.random_interval = int(os.environ.get("KVBM_RANDOM_INTERVAL", "7"))

        # Response storage
        self.control_responses: Dict[int, List[str]] = defaultdict(list)
        self.shakespeare_responses: Dict[int, List[str]] = defaultdict(list)
        self.random_responses: Dict[int, List[str]] = defaultdict(list)

        # Control sequences
        self.control_sequences = [
            "Hello world",
            "The quick brown fox jumps over the lazy dog. This is a standard pangram that contains all letters of the alphabet.",
            "Find light in the beautiful sea, I choose to be happy, You and I, you and I, we are like a beautiful melody that never ends, dancing through the night with stars as our companions, whispering secrets to the wind as we journey through life together, hand in hand, heart to heart, forever and always.",
            "The advancement of technology has fundamentally transformed the way we live, work, and communicate in the modern world. From the invention of the printing press to the development of the internet, each technological breakthrough has opened new possibilities and created unprecedented opportunities for human progress. Today, artificial intelligence and machine learning are reshaping industries, healthcare, education, and countless other fields, promising to solve complex problems and improve the quality of life for people around the globe.",
            "In the heart of Eldoria, an ancient land of boundless magic and mysterious creatures, lies the long-forgotten city of Aeloria. Once a beacon of knowledge and power, Aeloria was buried beneath the shifting sands of time, lost to the world for centuries. You are an intrepid explorer, known for your unparalleled curiosity and courage, who has stumbled upon an ancient map hinting at ests that Aeloria holds a secret so profound that it has the potential to reshape the very fabric of reality. Your journey will take you through treacherous deserts, enchanted forests, and across perilous mountain ranges. Your Task: Character Background: Develop a detailed background for your character. Describe their motivations for seeking out Aeloria, their skills and weaknesses, and any personal connections to the ancient city or its legends. Are they driven by a quest for knowledge, a search for lost familt clue is hidden.",
            "The human brain is the most complex organ in the known universe, containing approximately 86 billion neurons, each connected to thousands of others through intricate networks of synapses. This biological supercomputer processes information at speeds that would make even the most advanced artificial intelligence systems seem primitive by comparison. Every thought, memory, emotion, and decision we make is the result of electrical and chemical signals traveling through this vast neural network. The brain's ability to learn, adapt, and create is unmatched by any machine we have ever built. It can recognize patterns in milliseconds, solve complex problems through intuition, and generate creative ideas that have never existed before. Yet despite our incredible advances in neuroscience, we still understand only a fraction of how this remarkable organ truly works. The mysteries of consciousness, memory formation, and the nature of human intelligence continue to challenge the brightest minds in science and philosophy.",
        ]

        # Random sequences
        self.random_sequences = [
            "Coffee is ready",
            "The cat sat on the mat while the dog slept peacefully in the corner, creating a perfect picture of domestic tranquility that warmed the heart of anyone who witnessed this simple moment of harmony between two natural enemies turned friends.",
            "Mathematics is the language of the universe, and numbers are its alphabet. Through the elegant dance of equations and the symphony of algorithms, we unlock the secrets of nature's most profound mysteries. From the simple beauty of prime numbers to the complex elegance of calculus, mathematics provides us with the tools to understand everything from the smallest subatomic particles to the vast expanse of galaxies stretching across the cosmic void.",
            "A journey of a thousand miles begins with a single step, as the ancient Chinese proverb wisely reminds us. This timeless wisdom speaks to the fundamental truth that every great achievement, every monumental discovery, and every life-changing transformation starts with that crucial moment of decision - the moment when we choose to take action instead of remaining in the comfort of inaction. Whether it's learning a new skill, starting a business, writing a novel, or embarking on a spiritual quest, the path to success is paved with countless small steps, each one building upon the last, until we find ourselves transformed by the journey itself.",
            "Technology evolves rapidly, but human nature remains constant through the ages. Despite the incredible advances in artificial intelligence, virtual reality, and biotechnology, the fundamental desires, fears, and aspirations that drive human behavior have remained remarkably consistent throughout history. We still seek connection, meaning, and purpose in our lives. We still fear the unknown and crave security. We still dream of a better future and work to create it for ourselves and our loved ones. This paradox - the ever-changing nature of our tools and the unchanging nature of our hearts - is perhaps the most fascinating aspect of the human condition, reminding us that while we may build increasingly sophisticated machines, we remain fundamentally human in our core essence.",
        ]

    def download_shakespeare_text(self):
        """Download Shakespeare text if not present."""
        if not self.shakespeare_file.exists():
            print("Downloading Shakespeare text...")
            import urllib.request

            url = os.environ.get(
                "KVBM_SHAKESPEARE_URL",
                "https://ocw.mit.edu/ans7870/6/6.006/s08/lecturenotes/files/t8.shakespeare.txt",
            )
            urllib.request.urlretrieve(url, self.shakespeare_file)

            # Remove double newlines
            with open(self.shakespeare_file, "r", encoding="utf-8") as f:
                content = f.read()
            content = content.replace("\n\n", "")
            with open(self.shakespeare_file, "w", encoding="utf-8") as f:
                f.write(content)

    # Inherited from ApiTester, but override to add top_p for determinism testing
    def make_request(
        self,
        content: str,
        max_tokens: Optional[int] = None,
        temperature: float = 0.0,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Make API request and return completion text with determinism settings."""
        # Use determinism-specific defaults
        if max_tokens is None:
            max_tokens = int(os.environ.get("KVBM_MAX_TOKENS", "48"))
        if seed == 42:  # Default seed, use env override
            seed = int(os.environ.get("KVBM_SEED", "42"))

        return super().make_request(
            content,
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
            top_p=0.0001,  # For determinism
            **kwargs,
        )

    def warmup_server(self):
        """Perform comprehensive server warmup with all test prompts."""
        print("=" * 70)
        print("PERFORMING COMPREHENSIVE SERVER WARMUP")
        print("=" * 70)
        print(
            "Sending all control, Shakespeare, and random prompts to warm up the server..."
        )

        # Warmup with all control sequences
        print("Warming up with control sequences...")
        for i, control_seq in enumerate(self.control_sequences):
            print(f"  Warmup control sequence {i + 1}: {control_seq[:50]}...")
            try:
                self.make_request(control_seq)
            except Exception as e:
                print(f"  Warning: Warmup request failed: {e}")

        # Warmup with Shakespeare sequences that will be used in testing
        print("Warming up with Shakespeare sequences...")
        shakespeare_count = self.max_iterations // self.shakespeare_interval
        for seq_idx in range(1, shakespeare_count + 1):
            start_word = (seq_idx - 1) * self.word_count
            content = self.get_shakespeare_content(start_word)

            if content:
                print(
                    f"  Warmup Shakespeare sequence {seq_idx} (words {start_word}-{start_word + self.word_count - 1})..."
                )
                try:
                    self.make_request(content)
                except Exception as e:
                    print(f"  Warning: Warmup request failed: {e}")

        # Warmup with all random sequences
        print("Warming up with random sequences...")
        for i, random_seq in enumerate(self.random_sequences):
            print(f"  Warmup random sequence {i + 1}: {random_seq[:50]}...")
            try:
                self.make_request(random_seq)
            except Exception as e:
                print(f"  Warning: Warmup request failed: {e}")

        print("Server warmup completed!")
        print("=" * 70)

    def get_shakespeare_content(self, start_word: int) -> str:
        """Get Shakespeare content starting from a specific word."""
        with open(self.shakespeare_file, "r", encoding="utf-8") as f:
            words = f.read().split()

        end_word = min(start_word + self.word_count, len(words))
        return " ".join(words[start_word:end_word])

    def download_ifeval_dataset(self) -> List[str]:
        """Download and extract all prompts from IFEval dataset."""
        try:
            from datasets import load_dataset

            print("Loading complete IFEval dataset...")
            dataset = load_dataset("google/IFEval", split="train")

            # Extract all prompts from the dataset
            prompts = []

            for example in dataset:
                # IFEval has 'prompt' field with the instruction
                if "prompt" in example:
                    prompt_text = example["prompt"].strip()
                    if prompt_text:  # Only skip empty prompts
                        prompts.append(prompt_text)

            print(f"Loaded {len(prompts)} prompts from complete IFEval dataset")
            return prompts

        except ImportError:
            print(
                "Warning: datasets library not available, falling back to default prompts"
            )
            return self.control_sequences + self.random_sequences
        except Exception as e:
            print(
                f"Warning: Failed to load IFEval dataset ({e}), falling back to default prompts"
            )
            return self.control_sequences + self.random_sequences

    def run_test_iterations(self):
        """Run the test iterations with comprehensive warmup."""
        # Perform initial warmup before testing
        self.warmup_server()

        for iteration in range(1, self.max_iterations + 1):
            print(f"Iteration {iteration}/{self.max_iterations}")

            # Control sequence test
            if iteration % self.control_interval == 0:
                control_idx = (iteration // self.control_interval - 1) % len(
                    self.control_sequences
                )
                control_content = self.control_sequences[control_idx]

                print(
                    f"  Running control sequence {control_idx + 1}: {control_content[:50]}..."
                )
                completion = self.make_request(control_content)
                self.control_responses[control_idx].append(completion)
                print(f"  Response: {completion}")

            # Shakespeare sequence test
            if iteration % self.shakespeare_interval == 0:
                start_word = (
                    iteration // self.shakespeare_interval - 1
                ) * self.word_count
                content = self.get_shakespeare_content(start_word)

                if content:
                    shakespeare_idx = iteration // self.shakespeare_interval - 1
                    print(
                        f"  Running Shakespeare sequence {shakespeare_idx + 1} (words {start_word}-{start_word + self.word_count - 1})..."
                    )
                    completion = self.make_request(content)
                    self.shakespeare_responses[shakespeare_idx].append(completion)
                    print(f"  Response: {completion}")

            # Random sequence test
            if iteration % self.random_interval == 0:
                random_idx = (iteration // self.random_interval - 1) % len(
                    self.random_sequences
                )
                random_content = self.random_sequences[random_idx]

                print(
                    f"  Running random sequence {random_idx + 1}: {random_content[:50]}..."
                )
                completion = self.make_request(random_content)
                self.random_responses[random_idx].append(completion)
                print(f"  Response: {completion}")

    def analyze_responses(
        self, responses: Dict[int, List[str]], sequence_type: str
    ) -> Tuple[int, int]:
        """Analyze responses for determinism."""
        passed = 0
        failed = 0

        print(f"\n=== {sequence_type.upper()} SEQUENCES ===")

        for idx, response_list in responses.items():
            if not response_list:
                continue

            print(f"\n{sequence_type} sequence {idx + 1}:")
            print(f"Total responses: {len(response_list)}")

            if len(response_list) == 1:
                print("Single response - cannot check determinism")
                continue

            reference = response_list[0]
            differences = 0

            print(f"Reference response: {reference}")

            for i, response in enumerate(response_list[1:], 2):
                if response == reference:
                    print(f"Response {i}: MATCHES reference")
                else:
                    print(f"Response {i}: DIFFERS from reference")
                    print(f"  Expected: {reference}")
                    print(f"  Got:      {response}")
                    differences += 1

            if differences == 0:
                print(" ALL RESPONSES IDENTICAL - DETERMINISTIC")
                passed += 1
            else:
                print(f" {differences} DIFFERENCES DETECTED - NON-DETERMINISTIC")
                failed += 1

        return passed, failed

    def test_concurrent_determinism(
        self, prompts: List[str], num_workers: int = 4, requests_per_prompt: int = 3
    ) -> bool:
        """Test determinism with concurrent requests to the same prompts."""
        print("\n=== CONCURRENT DETERMINISM TEST ===")
        print(f"Workers: {num_workers}, Requests per prompt: {requests_per_prompt}")

        # Prepare test data: each prompt will get multiple concurrent requests
        test_tasks = []
        for i, prompt in enumerate(prompts):
            for req_num in range(requests_per_prompt):
                test_tasks.append(
                    {
                        "prompt_idx": i,
                        "prompt": prompt,
                        "request_id": f"p{i}_r{req_num}",
                    }
                )

        print(f"Total concurrent requests: {len(test_tasks)}")

        # Storage for responses grouped by prompt
        concurrent_responses: Dict[int, List[Tuple[str, str]]] = defaultdict(list)

        def make_concurrent_request(task):
            """Worker function for concurrent requests."""
            try:
                response = self.make_request(task["prompt"])
                return {
                    "prompt_idx": task["prompt_idx"],
                    "request_id": task["request_id"],
                    "response": response,
                    "success": True,
                    "error": None,
                }
            except Exception as e:
                return {
                    "prompt_idx": task["prompt_idx"],
                    "request_id": task["request_id"],
                    "response": None,
                    "success": False,
                    "error": str(e),
                }

        # Execute concurrent requests
        print("Executing concurrent requests...")
        start_time = time.time()

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(make_concurrent_request, task): task
                for task in test_tasks
            }

            # Collect results
            completed = 0
            failed = 0
            for future in as_completed(future_to_task):
                result = future.result()
                completed += 1

                if result["success"]:
                    concurrent_responses[result["prompt_idx"]].append(
                        (result["request_id"], result["response"])
                    )
                    if completed % 10 == 0:
                        print(f"  Completed: {completed}/{len(test_tasks)}")
                else:
                    failed += 1
                    print(f"  Failed request {result['request_id']}: {result['error']}")

        elapsed = time.time() - start_time
        print(
            f"Completed {completed} requests in {elapsed:.2f}s ({completed/elapsed:.1f} req/s)"
        )
        print(f"Failed requests: {failed}")

        # Analyze concurrent determinism
        print("\n=== CONCURRENT DETERMINISM ANALYSIS ===")
        total_prompts_tested = 0
        deterministic_prompts = 0

        for prompt_idx, responses in concurrent_responses.items():
            if len(responses) < 2:
                print(
                    f"Prompt {prompt_idx}: Only {len(responses)} response(s), skipping"
                )
                continue

            total_prompts_tested += 1
            prompt_text = prompts[prompt_idx]
            print(f"\nPrompt {prompt_idx}: {prompt_text[:50]}...")
            print(f"Concurrent responses: {len(responses)}")

            # Extract just the response text
            response_texts = [resp[1] for resp in responses]
            request_ids = [resp[0] for resp in responses]

            # Check if all responses are identical
            reference_response = response_texts[0]
            mismatches = []

            for req_id, response_text in zip(request_ids[1:], response_texts[1:]):
                if response_text != reference_response:
                    mismatches.append((req_id, response_text))

            if not mismatches:
                print(
                    f"   DETERMINISTIC: All {len(responses)} concurrent responses identical"
                )
                print(f"     Response: {reference_response}")
                deterministic_prompts += 1
            else:
                print(f"    NON-DETERMINISTIC: {len(mismatches)} different responses")
                print(f"    Reference ({request_ids[0]}): {reference_response}")
                for req_id, diff_response in mismatches:
                    print(f"     Different ({req_id}): {diff_response}")

        # Final assessment
        success_rate = (
            deterministic_prompts / total_prompts_tested
            if total_prompts_tested > 0
            else 0
        )
        print("\n=== FINAL CONCURRENT DETERMINISM RESULT ===")
        print(f"Prompts tested: {total_prompts_tested}")
        print(f"Deterministic: {deterministic_prompts}")
        print(f"Non-deterministic: {total_prompts_tested - deterministic_prompts}")
        print(f"Success rate: {success_rate:.1%}")
        print(f"Concurrency level: {num_workers} workers")
        print(f"Request rate: {completed/elapsed:.1f} req/s")

        return success_rate == 1.0


# =============================================================================
# Specialized Determinism Testers
# =============================================================================


class AggDeterminismTester(DeterminismTester):
    """Aggregated architecture specific determinism tester.

    Used for both standalone (vllm serve) and Dynamo frontend/worker tests.
    """

    # Flag to use larger cache reset (for Dynamo mode with larger GPU cache)
    _use_large_reset: bool = False

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
            # Use Shakespeare content to evict the on-device cache
            shakespeare_count = 300 if self._use_large_reset else 10
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


class DisaggDeterminismTester(DeterminismTester):
    """Disaggregated architecture specific determinism tester.

    Used for prefiller/decoder separated architecture tests.
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model_id: Optional[str] = None,
        server_type: Optional[str] = ServerType.vllm,
    ):
        super().__init__(base_url, model_id, server_type)

    def reset_prefix_cache(self):
        """Reset the prefix cache by evicting with Shakespeare content."""
        print("Resetting prefix cache...")
        # 150 shakespeare requests (each ~17 blocks) could evict ~2550 blocks
        shakespeare_count = 150
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
        print("Cache reset done")


# =============================================================================
# Fixtures
# =============================================================================


def _create_server_fixture(request, server_class, runtime_services):
    """Factory function for creating server manager fixtures."""
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    cpu_blocks = getattr(request, "param", {}).get("cpu_blocks", None)
    gpu_blocks = getattr(request, "param", {}).get("gpu_blocks", None)
    port = getattr(request, "param", {}).get("port", None)
    cpu_cache_gb = getattr(request, "param", {}).get("cpu_cache_gb", 1)

    log_dir = Path(request.node.name)
    server_type = BaseServerManager.detect_server_type()

    # Handle DynamoServerManager which needs request parameter
    if server_class == DynamoServerManager:
        server_manager = server_class(
            request=request,
            port=port,
            cpu_cache_blocks=cpu_blocks,
            gpu_cache_blocks=gpu_blocks,
            log_dir=log_dir,
            server_type=server_type,
            cpu_cache_gb=cpu_cache_gb,
        )
    else:
        server_manager = server_class(
            port=port,
            cpu_cache_blocks=cpu_blocks,
            gpu_cache_blocks=gpu_blocks,
            log_dir=log_dir,
            server_type=server_type,
            cpu_cache_gb=cpu_cache_gb,
        )

    start_timeout = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "600"))
    if not server_manager.start_server(timeout=start_timeout):
        pytest.fail(
            f"Failed to start {server_type} server (cpu_blocks={cpu_blocks}, "
            f"gpu_blocks={gpu_blocks}, port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()


@pytest.fixture(scope="function")
def standalone_llm_server(request, runtime_services):
    """Start and stop a standalone LLM server (vllm serve / trtllm-serve) for each test.

    To parametrize, use:
      @pytest.mark.parametrize("standalone_llm_server", [{"cpu_blocks": 10000}], indirect=True)
    """
    yield from _create_server_fixture(
        request, StandaloneServerManager, runtime_services
    )


@pytest.fixture(scope="function")
def dynamo_llm_server(request, runtime_services):
    """Start and stop a Dynamo-based LLM server for each test.

    To parametrize, use:
      @pytest.mark.parametrize("dynamo_llm_server", [{"cpu_blocks": 10000}], indirect=True)
    """
    yield from _create_server_fixture(request, DynamoServerManager, runtime_services)


@pytest.fixture(scope="function")
def disagg_llm_server(request, runtime_services):
    """Start and stop a disaggregated LLM server for each test.

    To parametrize, use:
      @pytest.mark.parametrize("disagg_llm_server", [{"cpu_blocks": 10000, "gpu_blocks": 1000}], indirect=True)
    """
    yield from _create_server_fixture(request, DisaggServerManager, runtime_services)


@pytest.fixture(scope="function")
def standalone_tester(standalone_llm_server):
    """Create determinism tester bound to the standalone server's base URL."""
    t = AggDeterminismTester(
        base_url=standalone_llm_server.base_url,
        server_type=standalone_llm_server.server_type,
    )
    t.download_shakespeare_text()
    return t


@pytest.fixture(scope="function")
def dynamo_tester(dynamo_llm_server):
    """Create determinism tester bound to the Dynamo server's base URL."""
    t = AggDeterminismTester(
        base_url=dynamo_llm_server.base_url,
        server_type=dynamo_llm_server.server_type,
    )
    t._use_large_reset = True  # Use larger cache reset for Dynamo
    t.download_shakespeare_text()
    return t


@pytest.fixture(scope="function")
def disagg_tester(disagg_llm_server):
    """Create determinism tester bound to the disaggregated server's base URL."""
    t = DisaggDeterminismTester(
        base_url=disagg_llm_server.base_url,
        server_type=disagg_llm_server.server_type,
    )
    t.download_shakespeare_text()
    return t


# Legacy fixture aliases for backward compatibility
@pytest.fixture(scope="function")
def llm_server(request, runtime_services):
    """Legacy alias - uses standalone mode.

    To parametrize, use:
      @pytest.mark.parametrize("llm_server", [{"cpu_blocks": 10000}], indirect=True)
    """
    yield from _create_server_fixture(
        request, StandaloneServerManager, runtime_services
    )


@pytest.fixture(scope="function")
def tester(llm_server):
    """Legacy alias - Create determinism tester bound to the running server's base URL."""
    t = AggDeterminismTester(
        base_url=llm_server.base_url, server_type=llm_server.server_type
    )
    t.download_shakespeare_text()
    return t


class TestDeterminism:
    """Test class for determinism validation."""

    def base_test_determinism_with_cache_reset(
        self, tester, llm_server, runtime_services, success_rate_threshold=1.0
    ):
        """Test determinism across cache reset: run test with warmup, reset cache, run again without warmup."""
        print("\n" + "=" * 70)
        print("STARTING DETERMINISM TEST (WITH CACHE RESET)")
        print("=" * 70)

        # Phase 1: Run test with warmup
        print("\n=== PHASE 1: BEFORE CACHE RESET (WITH WARMUP) ===")
        tester.run_test_iterations()

        # Store Phase 1 results
        phase1_control = {k: v.copy() for k, v in tester.control_responses.items()}
        phase1_shakespeare = {
            k: v.copy() for k, v in tester.shakespeare_responses.items()
        }
        phase1_random = {k: v.copy() for k, v in tester.random_responses.items()}

        # Reset cache
        print("\n" + "=" * 50)
        print("RESETTING CACHE")
        print("=" * 50)
        tester.reset_prefix_cache()

        # Clear response storage for Phase 2 (they are defaultdict, so they'll auto-initialize)
        tester.control_responses.clear()
        tester.shakespeare_responses.clear()
        tester.random_responses.clear()

        # Phase 2: Run test without warmup
        print("\n=== PHASE 2: AFTER CACHE RESET (NO WARMUP) ===")
        # Temporarily disable warmup by modifying the method
        original_warmup = tester.warmup_server
        tester.warmup_server = lambda: print(
            "Skipping warmup (testing determinism across cache reset)"
        )

        try:
            tester.run_test_iterations()
        finally:
            # Restore original warmup method
            tester.warmup_server = original_warmup

        # Compare Phase 1 vs Phase 2 results
        print("\n" + "=" * 70)
        print("CROSS-CACHE-RESET DETERMINISM ANALYSIS")
        print("=" * 70)

        total_passed = 0
        total_failed = 0

        # Compare control sequences
        for seq_idx in phase1_control:
            if seq_idx in tester.control_responses:
                phase1_responses = phase1_control[seq_idx]
                phase2_responses = tester.control_responses[seq_idx]

                min_responses = min(len(phase1_responses), len(phase2_responses))
                for i in range(min_responses):
                    if phase1_responses[i] == phase2_responses[i]:
                        total_passed += 1
                        print(f"   Control {seq_idx}, response {i}: DETERMINISTIC")
                    else:
                        total_failed += 1
                        print(f"   Control {seq_idx}, response {i}: NON-DETERMINISTIC")
                        print(f"     Before: {phase1_responses[i]}")
                        print(f"     After:  {phase2_responses[i]}")

        # Compare Shakespeare sequences
        for seq_idx in phase1_shakespeare:
            if seq_idx in tester.shakespeare_responses:
                phase1_responses = phase1_shakespeare[seq_idx]
                phase2_responses = tester.shakespeare_responses[seq_idx]

                min_responses = min(len(phase1_responses), len(phase2_responses))
                for i in range(min_responses):
                    if phase1_responses[i] == phase2_responses[i]:
                        total_passed += 1
                        print(f"   Shakespeare {seq_idx}, response {i}: DETERMINISTIC")
                    else:
                        total_failed += 1
                        print(
                            f"   Shakespeare {seq_idx}, response {i}: NON-DETERMINISTIC"
                        )
                        print(f"     Before: {phase1_responses[i]}")
                        print(f"     After:  {phase2_responses[i]}")

        # Compare random sequences
        for seq_idx in phase1_random:
            if seq_idx in tester.random_responses:
                phase1_responses = phase1_random[seq_idx]
                phase2_responses = tester.random_responses[seq_idx]

                min_responses = min(len(phase1_responses), len(phase2_responses))
                for i in range(min_responses):
                    if phase1_responses[i] == phase2_responses[i]:
                        total_passed += 1
                        print(f"   Random {seq_idx}, response {i}: DETERMINISTIC")
                    else:
                        total_failed += 1
                        print(f"   Random {seq_idx}, response {i}: NON-DETERMINISTIC")
                        print(f"     Before: {phase1_responses[i]}")
                        print(f"     After:  {phase2_responses[i]}")

        # Final assessment
        print("\n" + "=" * 70)
        print("FINAL CROSS-CACHE-RESET DETERMINISM ASSESSMENT")
        print("=" * 70)
        print(f"Total comparisons: {total_passed + total_failed}")
        print(f"Passed (deterministic): {total_passed}")
        print(f"Failed (non-deterministic): {total_failed}")
        success_rate = (
            total_passed / (total_passed + total_failed)
            if total_passed + total_failed > 0
            else 0
        )
        print(f"Success rate: {success_rate:.1%}")
        print(
            "Test compared responses before cache reset (with warmup) vs after cache reset (no warmup)."
        )

        if total_passed + total_failed == 0:
            pytest.skip("No tests were completed - insufficient data")

        assert (
            success_rate >= success_rate_threshold
        ), f"Model is not deterministic across cache reset: {total_failed} comparisons failed, success rate {success_rate:.1%} lower than expected {success_rate_threshold*100}%"
