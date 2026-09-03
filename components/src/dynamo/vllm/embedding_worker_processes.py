# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Embedding-only Dynamo process pool backed by one vLLM EngineCore.

vLLM supports multiple API processes as independent ``AsyncLLM`` clients of
one EngineCore. Embedding responses spend enough time in Python processing
that a single Dynamo endpoint process can become the bottleneck before the GPU.
This module applies the same supported vLLM multi-client topology to Dynamo's
embedding worker while keeping generation workers unchanged.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from vllm.config import VllmConfig
from vllm.usage.usage_lib import UsageContext
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.v1.engine.utils import get_engine_zmq_addresses, launch_core_engines
from vllm.v1.executor import Executor

logger = logging.getLogger(__name__)

_ROLE_ENV = "DYN_VLLM_EMBEDDING_PROCESS_ROLE"
_INDEX_ENV = "DYN_VLLM_EMBEDDING_PROCESS_INDEX"
_PARENT_PID_ENV = "DYN_VLLM_EMBEDDING_PARENT_PID"
_ENGINE_ADDRESSES_ENV = "DYN_VLLM_EMBEDDING_ENGINE_ADDRESSES"
_CHILD_ROLE = "child"
_RPC_BASE_PATH_ENV = "VLLM_RPC_BASE_PATH"


def is_embedding_process_child() -> bool:
    """Return whether this process is an internally launched endpoint child."""
    return os.environ.get(_ROLE_ENV) == _CHILD_ROLE


def start_embedding_parent_watchdog(poll_interval: float = 1.0) -> None:
    """Terminate an internal child if its owning Dynamo process disappears.

    The normal shutdown path explicitly terminates all children. This watchdog
    covers abrupt parent death, preventing orphaned runtime endpoints from
    remaining registered after the EngineCore owner has gone away.
    """
    if not is_embedding_process_child():
        return

    raw_parent_pid = os.environ.get(_PARENT_PID_ENV)
    if raw_parent_pid is None:
        raise RuntimeError(f"{_PARENT_PID_ENV} is required for an embedding child")
    parent_pid = int(raw_parent_pid)
    if os.getppid() != parent_pid:
        raise RuntimeError(
            f"embedding worker parent {parent_pid} exited before child startup"
        )

    def watch_parent() -> None:
        while True:
            if os.getppid() != parent_pid:
                logger.error(
                    "Embedding worker parent pid=%d exited; terminating child pid=%d",
                    parent_pid,
                    os.getpid(),
                )
                os.kill(os.getpid(), signal.SIGTERM)
                return
            time.sleep(poll_interval)

    threading.Thread(
        target=watch_parent,
        name="dynamo-embedding-parent-watchdog",
        daemon=True,
    ).start()


def _client_config(
    vllm_config: VllmConfig,
    *,
    process_count: int,
    process_index: int,
) -> VllmConfig:
    client_config = copy.deepcopy(vllm_config)
    client_config.parallel_config._api_process_count = process_count
    client_config.parallel_config._api_process_rank = process_index
    return client_config


def _attach_client(
    vllm_config: VllmConfig,
    *,
    process_count: int,
    process_index: int,
    input_address: str,
    output_address: str,
    usage_context: UsageContext,
    stat_loggers: list[Any],
    enable_log_requests: bool,
    disable_log_stats: bool,
) -> tuple[AsyncLLM, VllmConfig]:
    client_vllm_config = _client_config(
        vllm_config,
        process_count=process_count,
        process_index=process_index,
    )
    client = AsyncLLM.from_vllm_config(
        vllm_config=client_vllm_config,
        usage_context=usage_context,
        stat_loggers=stat_loggers,
        enable_log_requests=enable_log_requests,
        disable_log_stats=disable_log_stats,
        client_addresses={
            "input_address": input_address,
            "output_address": output_address,
        },
        client_count=process_count,
        client_index=process_index,
    )
    return client, client_vllm_config


class _PrecreatedStatLoggerFactory:
    """Return a logger that was created on the owning asyncio event loop."""

    def __init__(self, stat_logger: Any) -> None:
        self.stat_logger = stat_logger

    def __call__(self, _vllm_config: VllmConfig, _engine_index: int) -> Any:
        return self.stat_logger


def _precreate_stat_logger_factories(
    vllm_config: VllmConfig,
    stat_logger_factories: list[Any],
) -> list[_PrecreatedStatLoggerFactory]:
    # Dynamo's logger creates an asyncio task, so it must be instantiated on
    # this process's event-loop thread rather than in the bootstrap thread.
    return [
        _PrecreatedStatLoggerFactory(factory(vllm_config, 0))
        for factory in stat_logger_factories
    ]


def _decode_child_addresses(process_count: int) -> tuple[int, list[str], list[str]]:
    try:
        process_index = int(os.environ[_INDEX_ENV])
        payload = json.loads(os.environ[_ENGINE_ADDRESSES_ENV])
        payload_count = int(payload["process_count"])
        inputs = list(payload["inputs"])
        outputs = list(payload["outputs"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("invalid embedding child EngineCore bootstrap data") from exc

    if payload_count != process_count:
        raise RuntimeError(
            "embedding child process count does not match its parent: "
            f"config={process_count}, parent={payload_count}"
        )
    if len(inputs) != process_count or len(outputs) != process_count:
        raise RuntimeError(
            "embedding child received an incomplete EngineCore address set"
        )
    if not 0 <= process_index < process_count:
        raise RuntimeError(
            f"embedding child index {process_index} is outside [0, {process_count})"
        )
    return process_index, inputs, outputs


def _child_environment(
    *,
    process_count: int,
    process_index: int,
    addresses_json: str,
    parent_pid: int,
) -> dict[str, str]:
    env = os.environ.copy()
    env[_ROLE_ENV] = _CHILD_ROLE
    env[_INDEX_ENV] = str(process_index)
    env[_PARENT_PID_ENV] = str(parent_pid)
    env[_ENGINE_ADDRESSES_ENV] = addresses_json
    env["PYTHONUNBUFFERED"] = "1"

    # Each Dynamo runtime process needs its own system-status listener, but the
    # port must stay predictable: Prometheus scrapes a fixed containerPort and
    # Kubernetes liveness, readiness, and startup probes all target it, so an
    # ephemeral port leaves a child unscrapeable and unprobeable.
    #
    # Base plus index, which matches the operator's existing pattern for
    # multi-engine containers. The parent is index 0 and keeps the configured
    # port, and children start at index 1, so a pool of N occupies
    # DYN_SYSTEM_PORT .. DYN_SYSTEM_PORT + N - 1 with no collision.
    #
    # Only assign when the parent actually enabled the server. The default is
    # -1, meaning disabled, and hardcoding a port here used to start a listener
    # in every child even when the operator deliberately turned it off. An
    # explicit "0" means the caller asked for ephemeral ports, so pass that
    # through unchanged rather than overriding an intentional choice.
    parent_port = _configured_system_port()
    if parent_port is not None and parent_port > 0:
        env["DYN_SYSTEM_PORT"] = str(parent_port + process_index)
    return env


def _configured_system_port() -> int | None:
    """The parent's DYN_SYSTEM_PORT, or None when unset or unparseable."""
    raw = os.environ.get("DYN_SYSTEM_PORT")
    if raw is None or not raw.strip():
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning(
            "ignoring unparseable DYN_SYSTEM_PORT=%r; embedding worker "
            "children will inherit it unchanged",
            raw,
        )
        return None


def _terminate_processes(
    children: list[tuple[int, subprocess.Popen]], timeout: float
) -> None:
    for _index, child in children:
        if child.poll() is None:
            child.terminate()

    deadline = time.monotonic() + timeout
    for _index, child in children:
        if child.poll() is not None:
            continue
        try:
            child.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            child.kill()

    for _index, child in children:
        if child.poll() is None:
            try:
                child.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                logger.error(
                    "Embedding child pid=%d did not exit after SIGKILL", child.pid
                )


class EmbeddingWorkerProcessGroup:
    """Own child Dynamo endpoints, the shared EngineCore, and its IPC path."""

    def __init__(
        self,
        *,
        children: list[tuple[int, subprocess.Popen]],
        engine_manager: Any,
        rpc_directory: tempfile.TemporaryDirectory,
        previous_rpc_base_path: str | None,
        parent_failure_callback: Callable[[int, int], None] | None = None,
    ) -> None:
        self.children = children
        self.engine_manager = engine_manager
        self.rpc_directory = rpc_directory
        self.previous_rpc_base_path = previous_rpc_base_path
        self._parent_failure_callback = (
            parent_failure_callback or self._terminate_parent
        )
        self._stopping = threading.Event()
        self._cleanup_lock = threading.Lock()
        self._cleaned = False
        self._monitor_thread = threading.Thread(
            target=self._monitor_children,
            name="dynamo-embedding-child-monitor",
            daemon=True,
        )

    @staticmethod
    def _terminate_parent(process_index: int, returncode: int) -> None:
        logger.error(
            "Embedding endpoint child index=%d exited unexpectedly with status=%d; "
            "terminating the EngineCore owner",
            process_index,
            returncode,
        )
        os.kill(os.getpid(), signal.SIGTERM)

    def start_monitor(self) -> None:
        self._monitor_thread.start()

    def _monitor_children(self) -> None:
        while not self._stopping.wait(0.25):
            for process_index, child in self.children:
                returncode = child.poll()
                if returncode is not None:
                    if not self._stopping.is_set():
                        self._parent_failure_callback(process_index, returncode)
                    return

    def cleanup(self, timeout: float = 10.0) -> None:
        """Idempotently stop endpoints before stopping their EngineCore."""
        with self._cleanup_lock:
            if self._cleaned:
                return
            self._cleaned = True
            self._stopping.set()

            _terminate_processes(self.children, timeout)
            if self.engine_manager is not None:
                try:
                    self.engine_manager.shutdown(timeout=timeout)
                except Exception:
                    logger.exception("Failed to shut down shared embedding EngineCore")

            try:
                self.rpc_directory.cleanup()
            except Exception:
                logger.exception("Failed to remove shared embedding RPC directory")

            current_rpc_path = os.environ.get(_RPC_BASE_PATH_ENV)
            if current_rpc_path == self.rpc_directory.name:
                if self.previous_rpc_base_path is None:
                    os.environ.pop(_RPC_BASE_PATH_ENV, None)
                else:
                    os.environ[_RPC_BASE_PATH_ENV] = self.previous_rpc_base_path

        if (
            self._monitor_thread.is_alive()
            and threading.current_thread() is not self._monitor_thread
        ):
            self._monitor_thread.join(timeout=1.0)


class EmbeddingEngineCleanupResource:
    """Expose the existing ``cleanup()`` resource contract to WorkerFactory."""

    def __init__(
        self,
        process_group: EmbeddingWorkerProcessGroup,
        prometheus_temp_dir: Any,
    ) -> None:
        self.process_group = process_group
        self.prometheus_temp_dir = prometheus_temp_dir

    def cleanup(self) -> None:
        try:
            self.process_group.cleanup()
        finally:
            if self.prometheus_temp_dir is not None:
                self.prometheus_temp_dir.cleanup()


def _short_rpc_directory() -> tempfile.TemporaryDirectory:
    """Use a short path so vLLM IPC sockets stay below sockaddr_un limits."""
    base_dir = "/tmp" if os.path.isdir("/tmp") and os.access("/tmp", os.W_OK) else None
    rpc_directory = tempfile.TemporaryDirectory(prefix="dynamo-vllm-rpc-", dir=base_dir)
    # vLLM appends a slash and UUID. Linux sockaddr_un allows 107 path bytes.
    longest_socket_path = len("ipc://") + len(rpc_directory.name) + 1 + 36
    if longest_socket_path > 107:
        rpc_directory.cleanup()
        raise RuntimeError(
            "Unable to create a sufficiently short vLLM RPC path for "
            "--embedding-worker-processes"
        )
    return rpc_directory


def create_shared_embedding_engine_client(
    *,
    vllm_config: VllmConfig,
    process_count: int,
    usage_context: UsageContext,
    stat_loggers: list[Any],
    enable_log_requests: bool,
    disable_log_stats: bool,
) -> tuple[AsyncLLM, VllmConfig, EmbeddingWorkerProcessGroup | None]:
    """Create one client per Dynamo embedding process for one EngineCore."""
    if process_count <= 1:
        raise ValueError("shared embedding EngineCore requires at least 2 processes")

    if is_embedding_process_child():
        process_index, inputs, outputs = _decode_child_addresses(process_count)
        client, client_config = _attach_client(
            vllm_config,
            process_count=process_count,
            process_index=process_index,
            input_address=inputs[process_index],
            output_address=outputs[process_index],
            usage_context=usage_context,
            stat_loggers=stat_loggers,
            enable_log_requests=enable_log_requests,
            disable_log_stats=disable_log_stats,
        )
        logger.info(
            "Attached Dynamo embedding child %d/%d to shared EngineCore",
            process_index + 1,
            process_count,
        )
        return client, client_config, None

    role = os.environ.get(_ROLE_ENV)
    if role is not None:
        raise RuntimeError(f"unsupported internal embedding process role: {role!r}")

    previous_rpc_base_path = os.environ.get(_RPC_BASE_PATH_ENV)
    rpc_directory = _short_rpc_directory()
    os.environ[_RPC_BASE_PATH_ENV] = rpc_directory.name

    engine_manager = None
    process_group = None
    children: list[tuple[int, subprocess.Popen]] = []
    try:
        vllm_config.parallel_config._api_process_count = process_count
        vllm_config.parallel_config._api_process_rank = -1
        executor_class = Executor.get_class(vllm_config)
        addresses = get_engine_zmq_addresses(vllm_config, process_count)

        parent_logger_config = _client_config(
            vllm_config,
            process_count=process_count,
            process_index=0,
        )
        parent_stat_loggers = _precreate_stat_logger_factories(
            parent_logger_config, stat_loggers
        )
        with ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="dynamo-embedding-client"
        ) as client_pool:
            with launch_core_engines(
                vllm_config,
                executor_class,
                not disable_log_stats,
                addresses,
                process_count,
            ) as (engine_manager, coordinator, addresses, tensor_queue):
                if coordinator is not None or tensor_queue is not None:
                    raise RuntimeError(
                        "--embedding-worker-processes currently supports one "
                        "data-parallel EngineCore and text/token inputs only"
                    )

                addresses_json = json.dumps(
                    {
                        "process_count": process_count,
                        "inputs": list(addresses.inputs),
                        "outputs": list(addresses.outputs),
                    }
                )
                command = [sys.executable, "-m", "dynamo.vllm", *sys.argv[1:]]
                for process_index in range(1, process_count):
                    child = subprocess.Popen(
                        command,
                        env=_child_environment(
                            process_count=process_count,
                            process_index=process_index,
                            addresses_json=addresses_json,
                            parent_pid=os.getpid(),
                        ),
                    )
                    children.append((process_index, child))

                process_group = EmbeddingWorkerProcessGroup(
                    children=children,
                    engine_manager=engine_manager,
                    rpc_directory=rpc_directory,
                    previous_rpc_base_path=previous_rpc_base_path,
                )
                process_group.start_monitor()
                parent_client_future = client_pool.submit(
                    _attach_client,
                    vllm_config,
                    process_count=process_count,
                    process_index=0,
                    input_address=addresses.inputs[0],
                    output_address=addresses.outputs[0],
                    usage_context=usage_context,
                    stat_loggers=parent_stat_loggers,
                    enable_log_requests=enable_log_requests,
                    disable_log_stats=disable_log_stats,
                )

            parent_client, parent_config = parent_client_future.result()

        logger.info(
            "Started %d Dynamo embedding processes sharing one vLLM EngineCore",
            process_count,
        )
        return parent_client, parent_config, process_group
    except BaseException:
        if process_group is not None:
            process_group.cleanup()
        else:
            _terminate_processes(children, timeout=5.0)
            if engine_manager is not None:
                try:
                    engine_manager.shutdown(timeout=5.0)
                except Exception:
                    logger.exception(
                        "Failed to shut down EngineCore after embedding pool startup error"
                    )
            rpc_directory.cleanup()
            if previous_rpc_base_path is None:
                os.environ.pop(_RPC_BASE_PATH_ENV, None)
            else:
                os.environ[_RPC_BASE_PATH_ENV] = previous_rpc_base_path
        raise
