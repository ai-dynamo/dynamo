# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from abc import ABC, abstractmethod
from contextlib import ExitStack

import pynvml
import requests
from gpu_memory_service.common.types import RequestedLockType

from tests.gpu_memory_service.common.gms import GMSServer
from tests.utils.constants import FAULT_TOLERANCE_MODEL_NAME
from tests.utils.engine_process import EngineProcess
from tests.utils.managed_process import DynamoFrontendProcess
from tests.utils.payloads import check_health_generate, check_models_api
from tests.utils.port_utils import allocate_ports, deallocate_ports

logger = logging.getLogger(__name__)


def _normalize_weights_lock_mode(weights_lock_mode: RequestedLockType | None):
    if weights_lock_mode is None:
        return None

    try:
        if isinstance(weights_lock_mode, RequestedLockType):
            weights_lock_mode = weights_lock_mode.value
        weights_lock_mode = RequestedLockType(str(weights_lock_mode).lower())
    except ValueError as exc:
        raise ValueError(
            "Engine weights_lock_mode must be RW, RO, RW_OR_RO, or None. "
            "Use RW_OR_RO or None for the default RW_OR_RO behavior."
        ) from exc

    if weights_lock_mode == RequestedLockType.RW_OR_RO:
        return None
    return weights_lock_mode


def get_gpu_memory_used(device: int = 0) -> int:
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device)
        return pynvml.nvmlDeviceGetMemoryInfo(handle).used
    finally:
        pynvml.nvmlShutdown()


class GMSProcessManager:
    def __init__(
        self,
        request,
        engine_cls,
        *,
        weights_lock_mode: RequestedLockType | None = None,
    ):
        self._request = request
        self._engine_cls = engine_cls
        self._weights_lock_mode = _normalize_weights_lock_mode(weights_lock_mode)
        self._stack: ExitStack | None = None
        self.frontend_port: int | None = None
        self.weights_gms = None
        self.kv_cache_gms = None
        self._engine_ids: set[str] = set()
        self.engines: dict[str, GMSEngineProcess] = {}

    def __enter__(self):
        stack = ExitStack()
        try:
            self.weights_gms = stack.enter_context(GMSServer(device=0, tag="weights"))
            self.kv_cache_gms = stack.enter_context(GMSServer(device=0, tag="kv_cache"))
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
        self._engine_ids.clear()
        self.engines.clear()
        if stack is None:
            return False
        return stack.__exit__(exc_type, exc_val, exc_tb)

    def create_engine(
        self,
        engine_id: str,
        *,
        weights_lock_mode: RequestedLockType | None = None,
    ):
        if self._stack is None or self.frontend_port is None:
            raise RuntimeError(
                "GMSProcessManager must be entered before creating engines"
            )
        if engine_id in self._engine_ids:
            raise ValueError(f"engine {engine_id!r} already requested")

        if weights_lock_mode is None:
            weights_lock_mode = self._weights_lock_mode
        else:
            weights_lock_mode = _normalize_weights_lock_mode(weights_lock_mode)

        engine = self._engine_cls(
            self._request,
            self.frontend_port,
            engine_id=engine_id,
            weights_lock_mode=weights_lock_mode,
        )
        self._engine_ids.add(engine_id)
        return engine

    def start_engine(
        self,
        engine_id: str,
        *,
        weights_lock_mode: RequestedLockType | None = None,
    ):
        if self._stack is None:
            raise RuntimeError(
                "GMSProcessManager must be entered before starting engines"
            )
        engine = self._stack.enter_context(
            self.create_engine(engine_id, weights_lock_mode=weights_lock_mode)
        )
        self.engines[engine_id] = engine
        return engine


class GMSEngineProcess(EngineProcess, ABC):
    quiesce_route: str
    resume_route: str

    def __init__(
        self,
        request,
        engine_id: str,
        system_port: int,
        frontend_port: int,
        reserved_ports: list[int],
        *,
        weights_lock_mode: RequestedLockType | None = None,
    ):
        self.engine_id = engine_id
        self.system_port = system_port
        self._reserved_ports = reserved_ports
        self.weights_lock_mode = _normalize_weights_lock_mode(weights_lock_mode)

        log_dir = f"{request.node.name}_{engine_id}"
        shutil.rmtree(log_dir, ignore_errors=True)

        super().__init__(
            command=self.command(),
            env={
                **os.environ,
                "DYN_LOG": "debug",
                "DYN_SYSTEM_PORT": str(system_port),
                **self.env_updates(),
            },
            health_check_urls=[
                (f"http://localhost:{system_port}/health", self._is_ready),
                (f"http://localhost:{frontend_port}/v1/models", check_models_api),
                (f"http://localhost:{frontend_port}/health", check_health_generate),
            ],
            timeout=300,
            display_output=True,
            terminate_all_matching_process_names=False,
            stragglers=[],
            log_dir=log_dir,
            display_name=engine_id,
        )

    @abstractmethod
    def command(self) -> list[str]:
        raise NotImplementedError

    def env_updates(self) -> dict[str, str]:
        return {}

    def model_loader_extra_config(self) -> str | None:
        if self.weights_lock_mode is None:
            return None
        return json.dumps({"gms_lock_mode": self.weights_lock_mode.value})

    @abstractmethod
    def quiesce_payload(self) -> dict:
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
            f"http://localhost:{self.system_port}/engine/{route}",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        logger.info("%s %s: %s", self.engine_id, action, result)
        return result

    def quiesce(self) -> dict:
        return self._request_engine(
            self.quiesce_route,
            self.quiesce_payload(),
            30,
            "quiesce",
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
    quiesce_route = "sleep"
    resume_route = "wake_up"

    def __init__(
        self,
        request,
        frontend_port: int,
        *,
        engine_id: str,
        weights_lock_mode: RequestedLockType | None = None,
    ):
        reserved_ports = allocate_ports(3)
        self.kv_event_port = reserved_ports[1]
        self.nixl_port = reserved_ports[2]
        try:
            super().__init__(
                request,
                engine_id,
                reserved_ports[0],
                frontend_port,
                reserved_ports,
                weights_lock_mode=weights_lock_mode,
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
            "--gpu-memory-utilization",
            "0.9",
            "--kv-events-config",
            kv_events_cfg,
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

    def quiesce_payload(self) -> dict:
        return {"level": 2}


class SGLangWithGMSProcess(GMSEngineProcess):
    quiesce_route = "release_memory_occupation"
    resume_route = "resume_memory_occupation"

    def __init__(
        self,
        request,
        frontend_port: int,
        *,
        engine_id: str,
        weights_lock_mode: RequestedLockType | None = None,
    ):
        reserved_ports = allocate_ports(2)
        self.serve_port = reserved_ports[1]
        try:
            super().__init__(
                request,
                engine_id,
                reserved_ports[0],
                frontend_port,
                reserved_ports,
                weights_lock_mode=weights_lock_mode,
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
            "--mem-fraction-static",
            "0.9",
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

    def quiesce_payload(self) -> dict:
        return {}
