# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process orchestration for the cross-component GMS scenarios."""

from __future__ import annotations

import json
import logging
import os
import sys
import tempfile
from abc import ABC, abstractmethod
from contextlib import ExitStack

import requests

from tests.gpu_memory_service.common.gms import GMSServer
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DefaultPort
from tests.utils.engine_process import EngineProcess
from tests.utils.gpu_args import build_gpu_mem_args
from tests.utils.http_checks import check_health_ready
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)


def _replace_cli_option(args: list[str], option: str, value: str) -> list[str]:
    """Replace one two-token CLI option while retaining unrelated caps."""
    out = list(args)
    try:
        index = out.index(option)
    except ValueError:
        out.extend([option, value])
    else:
        if index + 1 >= len(out):
            raise ValueError(f"missing value for {option}")
        out[index + 1] = value
    return out


def _tp_size() -> int:
    """Return the local tensor-parallel size requested by an E2E scenario."""
    return max(1, int(os.environ.get("GMS_TEST_TP_SIZE", "1")))


def _tp_visible_devices() -> str:
    size = _tp_size()
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited:
        devices = [item.strip() for item in inherited.split(",") if item.strip()]
        if len(devices) < size:
            raise ValueError(
                f"GMS_TEST_TP_SIZE={size} exceeds CUDA_VISIBLE_DEVICES={inherited!r}"
            )
        return ",".join(devices[:size])
    return ",".join(map(str, range(size)))


class GMSProcessManager:
    """Start the shared GMS daemons and frontend for one test scenario."""

    def __init__(
        self,
        request,
        engine_cls,
        *,
        read_only_weights: bool = False,
        tags: tuple[str, ...] = ("weights", "kv_cache"),
        kv_directory: bool = False,
        migration_limit: int = 0,
    ):
        self._request = request
        self._engine_cls = engine_cls
        self._read_only_weights = read_only_weights
        self._tags = tags
        self._kv_directory = kv_directory
        self._migration_limit = max(0, int(migration_limit))
        self._directory_env: dict[str, str] = {}
        self.kv_directory_socket: str | None = None
        self.kv_directory_manifest: str | None = None
        self._stack: ExitStack | None = None
        self.frontend_port: int | None = None
        self.weights_gms = None
        self.kv_cache_gms = None
        self._engine_ids: set[str] = set()
        self.engines: dict[str, GMSEngineProcess] = {}

    def __enter__(self):
        stack = ExitStack()
        try:
            if self._kv_directory:
                shared_dir = stack.enter_context(
                    tempfile.TemporaryDirectory(prefix="gms-local-failover-")
                )
                self.kv_directory_socket = os.path.join(shared_dir, "directory.sock")
                self.kv_directory_manifest = f"local-{self._request.node.name}-v1"
                lease_dir = os.path.join(shared_dir, "leases")
                os.makedirs(lease_dir)
                self._directory_env = {
                    "FAILOVER_LOCK_PATH": os.path.join(shared_dir, "failover.lock"),
                    "GMS_KV_DIRECTORY_MODE": "authoritative",
                    "GMS_KV_DIRECTORY_SOCKET": self.kv_directory_socket,
                    "GMS_KV_DIRECTORY_MANIFEST": self.kv_directory_manifest,
                    "GMS_KV_DIRECTORY_DIAGNOSTICS": "1",
                    "GMS_KV_DIRECTORY_ASYNC_READ": "1",
                    "GMS_KV_DIRECTORY_ASYNC_PUBLISH": "1",
                    "GMS_KV_LEASES": "1",
                    "GMS_SGLANG_KV_LEASES": "1",
                    "GMS_KV_LEASE_SHM_DIR": lease_dir,
                    "GMS_SGLANG_SHARED_KV": "1",
                    "GMS_VLLM_SHARED_KV": "1",
                }
            if "weights" in self._tags:
                self.weights_gms = stack.enter_context(
                    GMSServer(device=0, tag="weights")
                )
                for device in range(1, _tp_size()):
                    stack.enter_context(GMSServer(device=device, tag="weights"))
            if "kv_cache" in self._tags:
                self.kv_cache_gms = stack.enter_context(
                    GMSServer(
                        device=0,
                        tag="kv_cache",
                        directory_socket_path=self.kv_directory_socket,
                    )
                )
                for device in range(1, _tp_size()):
                    stack.enter_context(
                        GMSServer(
                            device=device,
                            tag="kv_cache",
                        )
                    )
            frontend = stack.enter_context(
                DynamoFrontendProcess(
                    self._request,
                    frontend_port=0,
                    migration_limit=self._migration_limit,
                    display_name="frontend",
                )
            )
        except Exception:
            stack.close()
            raise

        self._stack = stack
        self.frontend_port = frontend.frontend_port
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stack = self._stack
        self._stack = None
        self.frontend_port = None
        self.weights_gms = None
        self.kv_cache_gms = None
        self._engine_ids.clear()
        self.kv_directory_socket = None
        self.kv_directory_manifest = None
        self._directory_env = {}
        self.engines.clear()
        if stack is None:
            return False
        return stack.__exit__(exc_type, exc_val, exc_tb)

    def create_engine(
        self,
        engine_id: str,
        *,
        read_only_weights: bool | None = None,
    ):
        if self._stack is None or self.frontend_port is None:
            raise RuntimeError(
                "GMSProcessManager must be entered before creating engines"
            )
        if engine_id in self._engine_ids:
            raise ValueError(f"engine {engine_id!r} already requested")

        if read_only_weights is None:
            read_only_weights = self._read_only_weights

        engine = self._engine_cls(
            self._request,
            self.frontend_port,
            engine_id=engine_id,
            read_only_weights=read_only_weights,
        )
        assert engine.env is not None
        engine.env.update(self._directory_env)
        engine.env["ENGINE_ID"] = engine_id
        self._engine_ids.add(engine_id)
        return engine

    def start_engine(
        self,
        engine_id: str,
        *,
        read_only_weights: bool | None = None,
        wait_until_ready: bool = True,
    ):
        if self._stack is None:
            raise RuntimeError(
                "GMSProcessManager must be entered before starting engines"
            )
        engine = self.create_engine(engine_id, read_only_weights=read_only_weights)
        health_checks = engine.health_check_urls
        if not wait_until_ready:
            engine.health_check_urls = []
        try:
            engine = self._stack.enter_context(engine)
        finally:
            engine.health_check_urls = health_checks
        self.engines[engine_id] = engine
        return engine


class GMSEngineProcess(EngineProcess, ABC):
    """Backend process wrapper with a common pause/resume surface."""

    pause_route: str
    resume_route: str

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        frontend_port: int,
        reserved_ports: list[int],
        *,
        read_only_weights: bool = False,
    ):
        self.engine_id = engine_id
        self.system_port = system_port
        self._reserved_ports = reserved_ports
        self.read_only_weights = read_only_weights

        super().__init__(
            command=self.command(),
            env={
                **os.environ,
                "DYN_LOG": "debug",
                "DYN_SYSTEM_PORT": str(system_port),
                **self.env_updates(),
            },
            health_check_urls=[
                (f"http://localhost:{system_port}/health", check_health_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=[],
            log_dir=f"{request.node.name}_{engine_id}",
            display_name=engine_id,
        )

    @abstractmethod
    def command(self) -> list[str]:
        raise NotImplementedError

    def env_updates(self) -> dict[str, str]:
        return {}

    def model_loader_extra_config(self) -> str | None:
        if not self.read_only_weights:
            return None
        return json.dumps({"gms_read_only": True})

    def wait_until_ready(self, timeout: float | None = None) -> None:
        """Wait for a previously started standby to become serving-ready."""
        self._check_urls(self.timeout if timeout is None else timeout)

    @abstractmethod
    def pause_payload(self) -> dict:
        raise NotImplementedError

    def resume_payload(self) -> dict:
        return {}

    def _request_engine(
        self,
        route: str,
        payload: dict,
        timeout: int,
        action: str,
    ) -> dict:
        response = requests.post(
            f"http://localhost:{self.system_port}/engine/control/{route}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info("%s %s: %s", self.engine_id, action, result)
        return result

    def pause(self) -> dict:
        return self._request_engine(
            self.pause_route,
            self.pause_payload(),
            30,
            "pause",
        )

    def resume(self, timeout: int = 30) -> dict:
        return self._request_engine(
            self.resume_route,
            self.resume_payload(),
            timeout,
            "resume",
        )

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            return super().__exit__(exc_type, exc_val, exc_tb)
        finally:
            deallocate_ports(self._reserved_ports)


class VLLMWithGMSProcess(GMSEngineProcess):
    pause_route = "sleep"
    resume_route = "wake_up"

    def __init__(
        self,
        request,
        frontend_port: int,
        *,
        engine_id: str,
        read_only_weights: bool = False,
    ):
        reserved_ports = allocate_ports(3, DefaultPort.SYSTEM1.value)
        self.kv_event_port = reserved_ports[1]
        self.nixl_port = reserved_ports[2]
        try:
            super().__init__(
                request,
                engine_id,
                reserved_ports[0],
                frontend_port,
                reserved_ports,
                read_only_weights=read_only_weights,
            )
        except Exception:
            deallocate_ports(reserved_ports)
            raise

    def env_updates(self) -> dict[str, str]:
        return {"VLLM_NIXL_SIDE_CHANNEL_PORT": str(self.nixl_port)}

    def command(self) -> list[str]:
        kv_events_cfg = json.dumps(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{self.kv_event_port}",
                "enable_kv_cache_events": True,
            }
        )
        gpu_mem_args = build_gpu_mem_args("build_vllm_gpu_mem_args") or [
            "--gpu-memory-utilization",
            "0.8",
        ]
        gpu_mem_util = os.environ.get("VLLM_GMS_GPU_MEM_UTIL")
        if gpu_mem_util is not None:
            gpu_mem_args = _replace_cli_option(
                gpu_mem_args, "--gpu-memory-utilization", gpu_mem_util
            )
        command = [
            sys.executable,
            "-m",
            "dynamo.vllm",
            "--model",
            FAULT_TOLERANCE_MODEL_NAME,
            "--load-format",
            "gms",
            "--enforce-eager",
            "--enable-sleep-mode",
            "--max-num-seqs",
            "1",
            "--kv-events-config",
            kv_events_cfg,
        ]
        command.extend(gpu_mem_args)
        extra_config = self.model_loader_extra_config()
        if extra_config is not None:
            command.extend(
                [
                    "--model-loader-extra-config",
                    extra_config,
                ]
            )
        return command

    def pause_payload(self) -> dict:
        return {"level": 2}


class TRTLLMWithGMSProcess(GMSEngineProcess):
    """TensorRT-LLM engine with GMS weights + pause/resume enabled."""

    pause_route = "release_memory_occupation"
    resume_route = "resume_memory_occupation"

    # Override via environment variables for CI or custom setups.
    TRTLLM_GMS_MODEL_NAME = os.environ.get(
        "TRTLLM_GMS_MODEL_NAME", FAULT_TOLERANCE_MODEL_NAME
    )
    TRTLLM_GMS_FREE_GPU_MEMORY_FRACTION = os.environ.get(
        "TRTLLM_GMS_FREE_GPU_MEMORY_FRACTION", "0.9"
    )
    TRTLLM_GMS_MAX_SEQ_LEN = os.environ.get("TRTLLM_GMS_MAX_SEQ_LEN", "256")
    TRTLLM_GMS_MAX_NUM_TOKENS = os.environ.get("TRTLLM_GMS_MAX_NUM_TOKENS", "256")
    TRTLLM_GMS_OVERRIDE_ENGINE_ARGS = os.environ.get(
        "TRTLLM_GMS_OVERRIDE_ENGINE_ARGS", ""
    )

    def __init__(
        self,
        request,
        frontend_port: int,
        *,
        engine_id: str,
        read_only_weights: bool = False,
        override_engine_args: str | None = None,
    ):
        reserved_ports = allocate_ports(1, DefaultPort.SYSTEM1.value)
        self._override_engine_args = override_engine_args
        try:
            super().__init__(
                request,
                engine_id,
                reserved_ports[0],
                frontend_port,
                reserved_ports,
                read_only_weights=read_only_weights,
            )
        except Exception:
            deallocate_ports(reserved_ports)
            raise

    def env_updates(self) -> dict[str, str]:
        env = {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
            "TLLM_WORKER_USE_SINGLE_PROCESS": "1",
            "MPI4PY_MPIABI": "openmpi",
            "OMPI_MCA_coll_ucc_enable": "0",
        }
        venv = os.environ.get("VIRTUAL_ENV")
        if venv:
            venv_lib = os.path.join(venv, "lib")
            existing = os.environ.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = f"{venv_lib}:{existing}" if existing else venv_lib
        return env

    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "dynamo.trtllm",
            "--model",
            self.TRTLLM_GMS_MODEL_NAME,
            "--gpus-per-node",
            "1",
            "--load-format",
            "gms",
            "--free-gpu-memory-fraction",
            self.TRTLLM_GMS_FREE_GPU_MEMORY_FRACTION,
            "--max-seq-len",
            self.TRTLLM_GMS_MAX_SEQ_LEN,
            "--max-num-tokens",
            self.TRTLLM_GMS_MAX_NUM_TOKENS,
        ]
        effective_override = self._override_engine_args
        if effective_override is None:
            effective_override = self.TRTLLM_GMS_OVERRIDE_ENGINE_ARGS
        if effective_override:
            command.extend(["--override-engine-args", effective_override])

        extra_config = self.model_loader_extra_config()
        if extra_config is not None:
            command.extend(["--model-loader-extra-config", extra_config])
        return command

    def pause_payload(self) -> dict:
        return {}


class SGLangWithGMSProcess(GMSEngineProcess):
    pause_route = "release_memory_occupation"
    resume_route = "resume_memory_occupation"

    def __init__(
        self,
        request,
        frontend_port: int,
        *,
        engine_id: str,
        read_only_weights: bool = False,
    ):
        reserved_ports = allocate_ports(2, DefaultPort.SYSTEM1.value)
        self.serve_port = reserved_ports[1]
        try:
            super().__init__(
                request,
                engine_id,
                reserved_ports[0],
                frontend_port,
                reserved_ports,
                read_only_weights=read_only_weights,
            )
        except Exception:
            deallocate_ports(reserved_ports)
            raise

    def command(self) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "dynamo.sglang",
            "--model-path",
            FAULT_TOLERANCE_MODEL_NAME,
            "--load-format",
            "gms",
            "--enable-memory-saver",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--mem-fraction-static",
            os.environ.get("SGLANG_GMS_MEM_FRACTION_STATIC", "0.8"),
            "--tp-size",
            str(_tp_size()),
            "--port",
            str(self.serve_port),
        ]
        extra_config = self.model_loader_extra_config()
        if extra_config is not None:
            command.extend(
                [
                    "--model-loader-extra-config",
                    extra_config,
                ]
            )
        return command

    def env_updates(self) -> dict[str, str]:
        return {
            "CUDA_VISIBLE_DEVICES": _tp_visible_devices(),
            "NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/g++",
        }

    def pause_payload(self) -> dict:
        return {}
