# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared process orchestration for the cross-component GMS scenarios."""

from __future__ import annotations

import json
import logging
import os
import sys
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack

import requests

from tests.gpu_memory_service.common.gms import GMSServer
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME, DefaultPort
from tests.utils.engine_process import EngineProcess
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)


def _http_ok(response) -> bool:
    """Readiness predicate: the endpoint answered with HTTP 200."""
    return response.status_code == 200


class GMSProcessManager:
    """Start the shared GMS daemons and frontend for one test scenario."""

    def __init__(
        self,
        request,
        engine_cls,
        *,
        read_only_weights: bool = False,
        tags: tuple[str, ...] = ("weights", "kv_cache"),
        tp: int = 1,
        model: str = FAULT_TOLERANCE_MODEL_NAME,
        shadow: bool = False,
        lock_path: str | None = None,
    ):
        self._request = request
        self._engine_cls = engine_cls
        self._read_only_weights = read_only_weights
        self._tags = tags
        self._tp = tp
        self._model = model
        self._shadow = shadow
        # Shadow engines coordinate active/standby through one shared flock.
        self._lock_path = lock_path
        self._stack: ExitStack | None = None
        self.frontend_port: int | None = None
        # Per-device GMS daemons. `weights_gms`/`kv_cache_gms` expose device 0
        # as the representative server for single-GPU assertions; the *_by_device
        # dicts hold every rank's server for TP>1.
        self.weights_gms = None
        self.kv_cache_gms = None
        self.weights_gms_by_device: dict[int, GMSServer] = {}
        self.kv_cache_gms_by_device: dict[int, GMSServer] = {}
        self._engine_ids: set[str] = set()
        self.engines: dict[str, GMSEngineProcess] = {}

    @property
    def lock_path(self) -> str | None:
        return self._lock_path

    def __enter__(self):
        stack = ExitStack()
        try:
            for device in range(self._tp):
                if "weights" in self._tags:
                    self.weights_gms_by_device[device] = stack.enter_context(
                        GMSServer(device=device, tag="weights")
                    )
                if "kv_cache" in self._tags:
                    self.kv_cache_gms_by_device[device] = stack.enter_context(
                        GMSServer(device=device, tag="kv_cache")
                    )
            self.weights_gms = self.weights_gms_by_device.get(0)
            self.kv_cache_gms = self.kv_cache_gms_by_device.get(0)
            frontend = stack.enter_context(
                DynamoFrontendProcess(
                    self._request,
                    frontend_port=0,
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
        self.weights_gms_by_device.clear()
        self.kv_cache_gms_by_device.clear()
        self._engine_ids.clear()
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
            tp=self._tp,
            model=self._model,
            shadow=self._shadow,
            lock_path=self._lock_path,
        )
        self._engine_ids.add(engine_id)
        return engine

    def start_engine(
        self,
        engine_id: str,
        *,
        read_only_weights: bool | None = None,
    ):
        if self._stack is None:
            raise RuntimeError(
                "GMSProcessManager must be entered before starting engines"
            )
        engine = self._stack.enter_context(
            self.create_engine(engine_id, read_only_weights=read_only_weights)
        )
        self.engines[engine_id] = engine
        return engine

    def start_engines_concurrently(
        self,
        engine_ids: list[str],
        *,
        read_only_weights: bool | None = None,
    ) -> dict[str, GMSEngineProcess]:
        """Start several engines at once and block until all are ready.

        Shadow engines race for the flock during startup, so they must come up
        concurrently (sequential start would let the first always win the lock).
        Each engine's context is still registered on the manager's ExitStack for
        teardown. Re-raises the first startup failure after cleaning up.
        """
        if self._stack is None:
            raise RuntimeError(
                "GMSProcessManager must be entered before starting engines"
            )
        created = {
            engine_id: self.create_engine(
                engine_id, read_only_weights=read_only_weights
            )
            for engine_id in engine_ids
        }
        # Enter (start + health-gate) each engine in its own thread, but DON'T
        # touch the shared ExitStack from those threads — ExitStack isn't
        # thread-safe. Register cleanup single-threaded after the join.
        with ThreadPoolExecutor(max_workers=len(created)) as executor:
            futures = {
                engine_id: executor.submit(engine.__enter__)
                for engine_id, engine in created.items()
            }
            started: dict[str, GMSEngineProcess] = {}
            errors = []
            for engine_id, future in futures.items():
                try:
                    started[engine_id] = future.result()
                except Exception as exc:  # noqa: BLE001 — surface after joining all
                    errors.append((engine_id, exc))
        for engine_id, engine in started.items():
            self._stack.push(engine)
            self.engines[engine_id] = engine
        if errors:
            engine_id, exc = errors[0]
            raise RuntimeError(f"engine {engine_id!r} failed to start: {exc}") from exc
        return {engine_id: self.engines[engine_id] for engine_id in engine_ids}


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
        tp: int = 1,
        model: str = FAULT_TOLERANCE_MODEL_NAME,
        shadow: bool = False,
        lock_path: str | None = None,
    ):
        self.engine_id = engine_id
        self.system_port = system_port
        self._reserved_ports = reserved_ports
        self.read_only_weights = read_only_weights
        self.tp = tp
        self.model = model
        self.shadow = shadow
        self.lock_path = lock_path

        super().__init__(
            command=self.command(),
            env={
                **os.environ,
                "DYN_LOG": "debug",
                "DYN_SYSTEM_PORT": str(system_port),
                **self.env_updates(),
            },
            # Shadow engines auto-pause into STANDBY and never register a
            # `generate` endpoint until they win the flock, so we can't gate on
            # the frontend discovering them. Readiness = the engine's own system
            # probe answering 200 (passes in both ACTIVE and STANDBY); the test
            # then asserts active/standby via GMS state and frontend inference.
            health_check_urls=(
                [(f"http://localhost:{system_port}/health", _http_ok)]
                if shadow
                else [
                    (f"http://localhost:{system_port}/health", self._is_ready),
                    (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                    (f"http://localhost:{frontend_port}/health", check_health_generate),
                ]
            ),
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

    @abstractmethod
    def pause_payload(self) -> dict:
        raise NotImplementedError

    def resume_payload(self) -> dict:
        return {}

    def _is_ready(self, response) -> bool:
        try:
            return response.json().get("status") == "ready"
        except ValueError:
            return False

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
        tp: int = 1,
        model: str = FAULT_TOLERANCE_MODEL_NAME,
        shadow: bool = False,
        lock_path: str | None = None,
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
                tp=tp,
                model=model,
                shadow=shadow,
                lock_path=lock_path,
            )
        except Exception:
            deallocate_ports(reserved_ports)
            raise

    def env_updates(self) -> dict[str, str]:
        env = {"VLLM_NIXL_SIDE_CHANNEL_PORT": str(self.nixl_port)}
        if self.shadow:
            # ENGINE_ID=0 loads + commits weights (RW); others import (RO).
            # The numeric suffix of "engine-N" drives RW/RO and the MX port offset.
            env["ENGINE_ID"] = self.engine_id.rsplit("-", 1)[-1]
            if self.lock_path is not None:
                env["FAILOVER_LOCK_PATH"] = self.lock_path
        return env

    def command(self) -> list[str]:
        kv_events_cfg = json.dumps(
            {
                "publisher": "zmq",
                "topic": "kv-events",
                "endpoint": f"tcp://*:{self.kv_event_port}",
                "enable_kv_cache_events": True,
            }
        )
        command = [
            sys.executable,
            "-m",
            "dynamo.vllm",
            "--model",
            self.model,
            "--tensor-parallel-size",
            str(self.tp),
            "--load-format",
            "gms",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.8",
            "--kv-events-config",
            kv_events_cfg,
        ]
        if self.shadow:
            # Shadow engines auto-pause and auto-wake via the flock; sleep mode
            # is implied by GMS shadow mode, so --enable-sleep-mode is omitted.
            command.append("--gms-shadow-mode")
        else:
            command.append("--enable-sleep-mode")
            command.extend(["--max-num-seqs", "1"])
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
        tp: int = 1,
        model: str = FAULT_TOLERANCE_MODEL_NAME,
        shadow: bool = False,
        lock_path: str | None = None,
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
                tp=tp,
                model=model,
                shadow=shadow,
                lock_path=lock_path,
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
        tp: int = 1,
        model: str = FAULT_TOLERANCE_MODEL_NAME,
        shadow: bool = False,
        lock_path: str | None = None,
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
                tp=tp,
                model=model,
                shadow=shadow,
                lock_path=lock_path,
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
            self.model,
            "--load-format",
            "gms",
            "--enable-memory-saver",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--mem-fraction-static",
            "0.8",
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
        return {"NVCC_PREPEND_FLAGS": "-ccbin /usr/bin/g++"}

    def pause_payload(self) -> dict:
        return {}
