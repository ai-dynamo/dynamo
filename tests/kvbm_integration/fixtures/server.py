# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Layer B: vLLM server bring-up.

Decomposed out of `test_determinism_agg.py`'s previous inline `LLMServerManager`.
The v1/v2 seam lives in `build_kv_transfer_config`; phase 2 only invokes the
v1 branch from tests, v2 ships inert pending phase 4 (see ACTIVE_PLAN.md).

External-attach mode: when `KVBM_EXTERNAL_BASE_URL` is set, the `kvbm_server`
fixture skips spawning vllm and binds to the running server. Used by
`scripts/run_eval.sh` for layered local iteration.
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, TextIO

import pytest
import requests

from tests.utils.port_utils import allocate_port, deallocate_port
from tests.utils.test_output import resolve_test_output_path

from ..common import ServerType

KvbmVersion = Literal["v1", "v2"]


# ---------------------------------------------------------------------------
# Model config (moved from test_determinism_agg.py so other tests can reuse it)
# ---------------------------------------------------------------------------


@dataclass
class KvbmModelConfig:
    """Describes a model and the vLLM serving flags needed for KVBM testing."""

    model_id: str
    block_size: Optional[int] = None  # None = let vllm decide
    attention_backend: Optional[str] = None  # None = let vllm decide
    max_model_len: int = 8000
    # Set False for MLA models: VLLM_BATCH_INVARIANT=1 disables prefix caching
    # for TRITON_MLA in vLLM 0.17.1, defeating KV offload testing.
    batch_invariant: bool = True

    @property
    def short_name(self) -> str:
        return self.model_id.split("/")[-1]

    @property
    def use_mla(self) -> bool:
        """True when the model uses Multi-head Latent Attention (e.g. TRITON_MLA)."""
        return self.attention_backend is not None and "MLA" in self.attention_backend


# ---------------------------------------------------------------------------
# kv-transfer-config builder (the v1/v2 seam)
# ---------------------------------------------------------------------------


_VALID_ONBOARD_MODES = ("intra", "inter")


def build_kv_transfer_config(
    kvbm_version: KvbmVersion,
    model_config: KvbmModelConfig,
    onboard_mode: str = "intra",
    cpu_blocks: Optional[int] = None,
) -> Dict[str, Any]:
    """Build the vLLM ``--kv-transfer-config`` payload for the chosen KVBM version.

    Both v1 and v2 use the canonical ``kvbm.v{N}.vllm.connector`` façades
    introduced in phase 4 (1↔2 char path mirror).

    v2 (agg) deliberately omits the ``leader.nova`` discovery block —
    `lib/kvbm-config/src/messenger.rs:43` defaults `discovery: None` and
    `build_messenger()` short-circuits when absent, so single-process agg
    mode needs no external discovery service.

    `onboard_mode` controls `leader.onboard.mode` for v2 — `"intra"` matches
    the sandbox script default; `"inter"` is the alternative validated in
    phase 4. v1 ignores the flag.

    `cpu_blocks` (v2 only) sets ``cache.host.num_blocks`` on the v2 leader
    config — exact parity with v1's ``DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS``
    path. When ``None``, the ``cache.host`` block is omitted entirely; the v2
    leader will then fail to start (matches v1's sanity_check) unless a disk
    tier is configured through some other channel. Callers are expected to
    pass ``spec.cpu_blocks`` from ``KvbmServerSpec``.
    """
    if kvbm_version == "v1":
        return {
            "kv_connector": "DynamoConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "kvbm.v1.vllm.connector",
        }
    if kvbm_version == "v2":
        if onboard_mode not in _VALID_ONBOARD_MODES:
            raise ValueError(
                f"unknown onboard_mode: {onboard_mode!r} "
                f"(expected one of {_VALID_ONBOARD_MODES})"
            )
        leader: Dict[str, Any] = {
            "tokio": {"worker_threads": 2},
            "onboard": {"mode": onboard_mode},
        }
        if cpu_blocks is not None:
            leader["cache"] = {"host": {"num_blocks": int(cpu_blocks)}}
        return {
            "kv_connector": "DynamoConnector",
            "kv_role": "kv_both",
            "kv_connector_module_path": "kvbm.v2.vllm.connector",
            "kv_connector_extra_config": {
                "leader": leader,
                "worker": {
                    "nixl": {"backends": {"UCX": {}, "POSIX": {}}},
                    "tokio": {"worker_threads": 2},
                },
            },
        }
    raise ValueError(f"unknown kvbm_version: {kvbm_version!r} (expected 'v1' or 'v2')")


# ---------------------------------------------------------------------------
# Server spec (single parametrize point for the kvbm_server fixture)
# ---------------------------------------------------------------------------


@dataclass
class KvbmServerSpec:
    """All parameters needed to launch (or attach to) a vLLM+KVBM server."""

    kvbm_version: KvbmVersion
    model_config: KvbmModelConfig
    cpu_blocks: Optional[int] = None
    gpu_blocks: Optional[int] = None
    port: Optional[int] = None
    server_type: str = ServerType.vllm
    # v2 only: leader onboard mode. v1 ignores this field.
    # Phase 5 TODO: decide whether to enumerate intra+inter per model in
    # the spec list, or gate one behind a KVBM_ONBOARD_MODE env var like
    # KVBM_ENABLE_MLA from phase 3.
    onboard_mode: str = "intra"

    @property
    def id(self) -> str:
        """Pytest parametrize id."""
        base = f"{self.kvbm_version}-{self.model_config.short_name}"
        if self.kvbm_version == "v2":
            return f"{base}-{self.onboard_mode}"
        return base


# ---------------------------------------------------------------------------
# KvbmServerManager — extracted from LLMServerManager, now version-aware
# ---------------------------------------------------------------------------


class KvbmServerManager:
    """Manages a vllm/trtllm server lifecycle for KVBM determinism testing.

    Identical to the previous `LLMServerManager` in `test_determinism_agg.py`,
    with two changes:
      1. The kv-transfer-config is built by `build_kv_transfer_config(version, ...)`
         instead of being hardcoded.
      2. The constructor accepts a `KvbmServerSpec` to make parametrization clean.
    """

    def __init__(
        self,
        spec: KvbmServerSpec,
        log_dir: Optional[Path] = None,
    ):
        self.spec = spec
        self.server_type = spec.server_type
        self.model_config = spec.model_config
        self.kvbm_version: KvbmVersion = spec.kvbm_version
        self.cpu_cache_blocks = spec.cpu_blocks
        self.gpu_cache_blocks = spec.gpu_blocks

        # Use provided port, env var, or allocate a dynamic port to avoid conflicts
        if spec.port is not None:
            self.port = spec.port
            self.port_allocated = False
        elif os.environ.get("KVBM_SERVER_PORT"):
            self.port = int(os.environ["KVBM_SERVER_PORT"])
            self.port_allocated = False
        else:
            self.port = allocate_port(start_port=8000)
            self.port_allocated = True
        self.base_url = f"http://localhost:{self.port}"
        self.metrics_port = allocate_port(start_port=6880)
        self.metrics_port_allocated = True
        self.process: Optional[subprocess.Popen] = None

        # Prepare logging
        self.log_dir = log_dir or Path(".")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_str = (
            f"{self.kvbm_version}_cpu{self.cpu_cache_blocks or 'default'}"
            f"_gpu{self.gpu_cache_blocks or 'default'}"
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
                "DYN_KVBM_METRICS": "true",
                "DYN_KVBM_METRICS_PORT": str(self.metrics_port),
            }
        )

        # CPU cache blocks override via env
        if self.cpu_cache_blocks is not None:
            self.env["DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS"] = str(
                self.cpu_cache_blocks
            )

        if self.server_type == ServerType.vllm:
            self._set_up_vllm_config()
        elif self.server_type == ServerType.trtllm:
            self._set_up_trtllm_config()
        else:
            raise ValueError(
                f"{self.server_type} is not supported yet in the KVBM test suite"
            )

    def _set_up_vllm_config(self) -> None:
        self.env["VLLM_SERVER_DEV_MODE"] = "1"
        if self.model_config.batch_invariant:
            self.env["VLLM_BATCH_INVARIANT"] = "1"
        else:
            self.env.pop("VLLM_BATCH_INVARIANT", None)

        kv_transfer_config = build_kv_transfer_config(
            self.kvbm_version,
            self.model_config,
            onboard_mode=self.spec.onboard_mode,
            cpu_blocks=self.spec.cpu_blocks,
        )

        self.server_cmd = [
            "vllm",
            "serve",
            "--port",
            str(self.port),
            "--kv-transfer-config",
            json.dumps(kv_transfer_config),
            self.model_config.model_id,
            "--max-model-len",
            str(self.model_config.max_model_len),
        ]

        gpu_mem_util = os.environ.get("KVBM_GPU_MEMORY_UTILIZATION", "0.9")
        self.server_cmd.extend(["--gpu-memory-utilization", gpu_mem_util])

        if self.model_config.block_size is not None:
            self.server_cmd.extend(["--block-size", str(self.model_config.block_size)])

        if self.model_config.attention_backend is not None:
            self.server_cmd.extend(
                ["--attention-config.backend", self.model_config.attention_backend]
            )

        if self.gpu_cache_blocks is not None:
            self.server_cmd.extend(
                ["--num-gpu-blocks-override", str(self.gpu_cache_blocks)]
            )

    def _set_up_trtllm_config(self) -> None:
        if self.kvbm_version != "v1":
            raise ValueError(
                f"trtllm path only supports kvbm v1; got {self.kvbm_version!r}"
            )
        config_path = os.environ.get(
            "KVBM_TRTLLM_LLMAPI_CONFIG_PATH", "/tmp/kvbm_llm_api_config.yaml"
        )
        llm_api_config: Dict[str, Any] = {}
        # explicitly disable CUDA graph since Connector API doesn't support CUDA graph yet in TRTLLM
        llm_api_config["cuda_graph_config"] = None
        llm_api_config["kv_cache_config"] = {
            "enable_partial_reuse": False,
            # Set a small GPU fraction so we can evict/reset the on-device kv cache faster
            "free_gpu_memory_fraction": 0.10,
        }
        llm_api_config["kv_connector_config"] = {
            "connector_module": "kvbm.trtllm_integration.connector",
            "connector_scheduler_class": "DynamoKVBMConnectorLeader",
            "connector_worker_class": "DynamoKVBMConnectorWorker",
        }

        if self.gpu_cache_blocks is not None:
            del llm_api_config["kv_cache_config"]["free_gpu_memory_fraction"]
            # TRTLLM defaults 32 tokens per block
            llm_api_config["kv_cache_config"]["max_tokens"] = (
                int(self.gpu_cache_blocks) * 32
            )

        self.server_cmd = [
            "trtllm-serve",
            self.model_config.model_id,
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
                log_file.write(line)
                log_file.flush()
                sys.stdout.write(f"[{prefix}] {line}")
                sys.stdout.flush()
        except (ValueError, OSError):
            pass
        finally:
            pipe.close()

    def start_server(self, timeout: int = 300) -> bool:
        """Start LLM server and wait for readiness."""
        if self.is_server_running():
            self.stop_server()
            time.sleep(2)

        self.server_stdout_file = open(self.server_log_file.with_suffix(".log"), "w")

        header = (
            f"=== {self.server_type} Server Started at {datetime.now()} ===\n"
            f"Command: {' '.join(self.server_cmd)}\n"
        )
        self.server_stdout_file.write(header)
        self.server_stdout_file.flush()
        print(f"[{self.server_type}] {header}", end="")

        self.process = subprocess.Popen(
            self.server_cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=self.env,
            preexec_fn=os.setsid,
            text=True,
            bufsize=1,
        )

        self._tee_threads = [
            threading.Thread(
                target=self._tee_output,
                args=(self.process.stdout, self.server_stdout_file, self.server_type),
                daemon=True,
            ),
        ]
        for t in self._tee_threads:
            t.start()

        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_server_running():
                try:
                    requests.get(
                        f"http://localhost:{self.metrics_port}/metrics", timeout=5
                    )
                    return True
                except requests.exceptions.RequestException:
                    print(
                        f"Warning: server healthy but metrics port {self.metrics_port} not reachable yet"
                    )
            if self.process.poll() is not None:
                for t in self._tee_threads:
                    t.join(timeout=2)
                self._close_log_files()
                return False
            time.sleep(5)

        self.stop_server()
        return False

    def stop_server(self) -> None:
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
        for t in self._tee_threads:
            t.join(timeout=2)
        self._tee_threads = []
        self._close_log_files()

        if self.port_allocated:
            deallocate_port(self.port)
            self.port_allocated = False
        if self.metrics_port_allocated:
            deallocate_port(self.metrics_port)
            self.metrics_port_allocated = False

    def _close_log_files(self) -> None:
        if self.server_stdout_file:
            self.server_stdout_file.write(
                f"\n=== Server Stopped at {datetime.now()} ===\n"
            )
            self.server_stdout_file.close()
            self.server_stdout_file = None

    def is_server_running(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            if response.status_code != 200:
                return False

            test_payload = {
                "model": self.model_config.model_id,
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


# ---------------------------------------------------------------------------
# ServerHandle: the duck-typed object yielded by `kvbm_server`
# ---------------------------------------------------------------------------


@dataclass
class _ExternalServer:
    """Drop-in stand-in for `KvbmServerManager` when KVBM_EXTERNAL_BASE_URL is set.

    Exposes the same attributes the test bodies and `common.py` reach for
    (`base_url`, `metrics_port`, `model_config`, `server_type`). `stop_server()`
    is a no-op — the external process owns its own lifecycle.
    """

    base_url: str
    metrics_port: int
    model_config: KvbmModelConfig
    server_type: str = ServerType.vllm
    kvbm_version: KvbmVersion = "v1"
    cpu_cache_blocks: Optional[int] = None
    gpu_cache_blocks: Optional[int] = None

    def stop_server(self) -> None:
        return None

    def is_server_running(self) -> bool:
        try:
            return requests.get(f"{self.base_url}/health", timeout=5).status_code == 200
        except requests.exceptions.RequestException:
            return False


# ServerHandle = either KvbmServerManager (spawn mode) or _ExternalServer (attach mode).
# Both expose: base_url, metrics_port, model_config, server_type, stop_server(), is_server_running().
ServerHandle = Any


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------

_SERVER_START_TIMEOUT = int(os.environ.get("KVBM_SERVER_START_TIMEOUT", "600"))


@pytest.fixture(scope="function")
def kvbm_server_spec(request) -> KvbmServerSpec:
    """Indirect-parametrize entry point: provides the KvbmServerSpec for one test case."""
    spec = getattr(request, "param", None)
    if spec is None:
        raise RuntimeError(
            "kvbm_server_spec must be parametrized indirectly with a KvbmServerSpec instance"
        )
    if not isinstance(spec, KvbmServerSpec):
        raise TypeError(
            f"kvbm_server_spec param must be a KvbmServerSpec, got {type(spec).__name__}"
        )
    return spec


@pytest.fixture(scope="function")
def kvbm_server(request, kvbm_server_spec, kvbm_deps):
    """Spawn vllm+KVBM (or attach to a running one) and yield a server handle.

    External-attach mode: when ``KVBM_EXTERNAL_BASE_URL`` is set the fixture
    skips spawning and returns an `_ExternalServer` bound to the env-var
    base URL + metrics port. Used by `scripts/run_eval.sh`.
    """
    external_url = os.environ.get("KVBM_EXTERNAL_BASE_URL")
    if external_url:
        external_metrics = int(os.environ.get("KVBM_EXTERNAL_METRICS_PORT", "0"))
        if external_metrics == 0:
            raise RuntimeError(
                "KVBM_EXTERNAL_BASE_URL is set but KVBM_EXTERNAL_METRICS_PORT is not — "
                "both are required for external-attach mode"
            )
        handle = _ExternalServer(
            base_url=external_url,
            metrics_port=external_metrics,
            model_config=kvbm_server_spec.model_config,
            server_type=kvbm_server_spec.server_type,
            kvbm_version=kvbm_server_spec.kvbm_version,
            cpu_cache_blocks=kvbm_server_spec.cpu_blocks,
            gpu_cache_blocks=kvbm_server_spec.gpu_blocks,
        )
        if not handle.is_server_running():
            pytest.fail(
                f"KVBM_EXTERNAL_BASE_URL={external_url} is not reachable; "
                "is the server running?"
            )
        yield handle
        return

    # Spawn mode — kvbm_deps already brought up NATS+etcd for v1 (or no-op for v2).
    del kvbm_deps  # only used to enforce ordering; runtime_services env vars are set
    logger = logging.getLogger("pytest")
    logger.setLevel(logging.INFO)

    log_dir = Path(resolve_test_output_path(request.node.name))
    server_manager = KvbmServerManager(spec=kvbm_server_spec, log_dir=log_dir)

    if not server_manager.start_server(timeout=_SERVER_START_TIMEOUT):
        pytest.fail(
            f"Failed to start {kvbm_server_spec.server_type} server "
            f"(version={kvbm_server_spec.kvbm_version}, "
            f"model={kvbm_server_spec.model_config.short_name}, "
            f"cpu_blocks={kvbm_server_spec.cpu_blocks}, "
            f"gpu_blocks={kvbm_server_spec.gpu_blocks}, "
            f"port={server_manager.port})"
        )

    yield server_manager

    server_manager.stop_server()
