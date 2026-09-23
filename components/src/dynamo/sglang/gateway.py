# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process request gateway for the SGLang worker: the leader keeps the engine
and spawns N ``dynamo.sglang`` children that each serve requests through their own
SGLang ``TokenizerWorker``. Design notes: AGENTS.md, "Multi-process gateway".
"""

from __future__ import annotations

import asyncio
import ctypes
import logging
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import types
from typing import Awaitable, Callable, Optional

import sglang as sgl

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV
from dynamo.common.utils.graceful_shutdown import get_grace_period_seconds

ENV_PARENT_PID = "DYN_SGLANG_GATEWAY_PARENT_PID"
ENV_CHILD_INDEX = "DYN_SGLANG_GATEWAY_CHILD_INDEX"
ENV_SYSTEM_PORT = "DYN_SYSTEM_PORT"
ENV_SYSTEM_PORT_BASE = "DYN_SGLANG_GATEWAY_SYSTEM_PORT"
ENV_LOAD_TIME = "DYN_SGLANG_GATEWAY_LOAD_TIME_S"
# Runtime-data keys on every gateway child's model card: consumers that count
# workers or sum per-worker capacity can collapse the N instances of one engine.
GATEWAY_ENGINE_ID_KEY = "dynamo.sglang.gateway_engine"
GATEWAY_WORKERS_KEY = "dynamo.sglang.gateway_workers"
# Children drain like any worker (grace + drain + cleanup); give them that budget.
CHILD_DRAIN_AND_CLEANUP_SECS = 60.0

DIRECT_ENGINE_WORKER_FLAGS = (
    "image_diffusion_worker",
    "video_generation_worker",
    "rerank_worker",
    "embedding_worker",
    "multimodal_encode_worker",
    "multimodal_worker",
    "diffusion_worker",
)


class GatewayEngine:
    """What the request handlers need from an ``sgl.Engine``.

    A child never owns scheduler processes; it only holds a ``TokenizerWorker`` bound
    to the parent's router. ``Engine.async_generate`` only touches
    ``self.tokenizer_manager``, so it can be reused unchanged.
    """

    def __init__(self, tokenizer_manager, server_args, port_args, scheduler_info):
        self.tokenizer_manager = tokenizer_manager
        self.server_args = server_args
        self.port_args = port_args
        self._scheduler_init_result = types.SimpleNamespace(
            scheduler_infos=[scheduler_info]
        )

    async_generate = sgl.Engine.async_generate
    _resolve_routed_dp_rank = sgl.Engine._resolve_routed_dp_rank

    def shutdown(self):
        pass


def is_gateway_child() -> bool:
    return ENV_PARENT_PID in os.environ


def gateway_child_index() -> int:
    return int(os.environ.get(ENV_CHILD_INDEX, "0"))


def owns_engine_metrics() -> bool:
    """SGLang's schedulers push KV metrics to one PULL socket, so exactly one gateway
    process may consume them: child 0 (or the single worker when the mode is off)."""
    return gateway_child_index() == 0


def effective_gateway_workers(server_args, dynamo_args) -> int:
    """Gateway count implied by ``--gateway-workers`` and ``--tokenizer-worker-num``.

    The engine runs one SGLang tokenizer worker per gateway process. A tokenizer
    count above 1 that differs from ``--gateway-workers`` is an error rather than a
    silent choice; the default of 1 lets ``--gateway-workers N`` stand alone."""
    requested = getattr(dynamo_args, "gateway_workers", None)
    tokenizer_workers = getattr(server_args, "tokenizer_worker_num", 1) or 1
    if requested is None:
        return tokenizer_workers
    if tokenizer_workers > 1 and tokenizer_workers != requested:
        raise ValueError(
            f"--gateway-workers {requested} conflicts with --tokenizer-worker-num "
            f"{tokenizer_workers}: each gateway process is one SGLang tokenizer "
            "worker, so drop --tokenizer-worker-num or give both the same value"
        )
    return requested


def gateway_worker_count(server_args, dynamo_args) -> int:
    if (getattr(server_args, "node_rank", 0) or 0) != 0 or is_gateway_child():
        return 1
    return effective_gateway_workers(server_args, dynamo_args)


def validate_gateway_mode(server_args, dynamo_args, count: int) -> None:
    if count <= 1:
        return
    direct = [f for f in DIRECT_ENGINE_WORKER_FLAGS if getattr(dynamo_args, f, False)]
    if direct:
        raise ValueError(
            "gateway mode (--gateway-workers / --tokenizer-worker-num > 1) is only "
            "supported by the decode and prefill LLM workers, not with "
            f"--{direct[0].replace('_', '-')}"
        )
    if getattr(server_args, "enable_lora", False):
        raise ValueError(
            "gateway mode is not supported with --enable-lora: dynamic LoRA state "
            "lives in each gateway process"
        )
    if getattr(server_args, "enable_forward_pass_metrics", False):
        raise ValueError(
            "gateway mode is not supported with --enable-forward-pass-metrics: the "
            "schedulers stamp forward-pass metrics with the identity of the process "
            "that created the engine, which serves no requests in gateway mode"
        )
    if os.environ.get(SNAPSHOT_CONTROL_DIR_ENV):
        raise ValueError(
            "gateway mode is not supported in snapshot mode "
            f"({SNAPSHOT_CONTROL_DIR_ENV} is set): snapshot warmup needs a single "
            "tokenizer manager"
        )
    routes = getattr(dynamo_args, "engine_routes", None)
    if routes and getattr(sgl.Engine, "attach_tokenizer_worker", None) is None:
        from dynamo.sglang.engine_routes import parse_engine_route_descriptors

        engine_targets = [
            d.path
            for d in parse_engine_route_descriptors(routes)
            if d.target == "engine"
        ]
        if engine_targets:
            raise ValueError(
                "gateway mode with engine-target --engine-routes "
                f"({', '.join(engine_targets)}) needs an SGLang with "
                "Engine.attach_tokenizer_worker; on this SGLang the gateway children "
                "only expose the tokenizer manager"
            )


def reserve_system_port_for_children() -> None:
    """Leader side, before the runtime starts. The leader serves no requests, so the
    configured ``DYN_SYSTEM_PORT`` goes to child 0; the leader runs without a system
    status server and the other children bind a random port (0), because any fixed
    offset can collide with another worker group's configured port."""
    raw = os.environ.get(ENV_SYSTEM_PORT)
    try:
        port = int(raw) if raw is not None else -1
    except ValueError:
        return
    if port <= 0:
        return
    os.environ[ENV_SYSTEM_PORT_BASE] = str(port)
    os.environ[ENV_SYSTEM_PORT] = "-1"


def child_environment(index: int, load_time: Optional[float] = None) -> dict[str, str]:
    env = {
        **os.environ,
        ENV_PARENT_PID: str(os.getpid()),
        ENV_CHILD_INDEX: str(index),
    }
    base = env.pop(ENV_SYSTEM_PORT_BASE, None)
    if base is not None:
        env[ENV_SYSTEM_PORT] = base if index == 0 else "0"
    env.pop(ENV_LOAD_TIME, None)
    if index == 0 and load_time is not None:
        # The leader measured the load; the metrics owner publishes it once.
        env[ENV_LOAD_TIME] = repr(load_time)
    return env


def attached_engine_load_time() -> Optional[float]:
    raw = os.environ.get(ENV_LOAD_TIME)
    return float(raw) if raw else None


def follow_pause_broadcasts(
    tokenizer_manager, on_change: Callable[[], Awaitable[None]]
) -> bool:
    """A pause or resume issued through any sibling reaches this process only as the
    router's broadcast into its ``TokenizerWorker``; run ``on_change`` after each one so
    this child's discovery registration follows the shared state."""
    apply = getattr(tokenizer_manager, "_apply_pause_continue_broadcast", None)
    if apply is None:
        return False

    async def apply_and_follow(obj):
        await apply(obj)
        await on_change()

    tokenizer_manager._apply_pause_continue_broadcast = apply_and_follow
    return True


def gateway_engine_id() -> Optional[str]:
    pid = os.environ.get(ENV_PARENT_PID)
    if pid is None:
        return None
    return f"{socket.gethostname()}:{pid}"


def child_shutdown_timeout() -> float:
    return get_grace_period_seconds() + CHILD_DRAIN_AND_CLEANUP_SECS


def start_parent_watchdog(
    parent_pid: int,
    on_parent_death: Optional[Callable[[], None]] = None,
    poll_seconds: float = 2.0,
) -> threading.Thread:
    """The leader owns the schedulers; a child that outlives it would stay registered
    and fail every request. Ask the kernel for SIGTERM on parent death and poll as a
    fallback for the window before the flag is set and for non-Linux hosts."""
    if on_parent_death is None:

        def on_parent_death() -> None:
            logging.error(
                "gateway parent pid=%d is gone; shutting down child pid=%d",
                parent_pid,
                os.getpid(),
            )
            os.kill(os.getpid(), signal.SIGTERM)

    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass

    def poll() -> None:
        while True:
            try:
                os.kill(parent_pid, 0)
            except ProcessLookupError:
                on_parent_death()
                return
            except PermissionError:
                pass
            if os.getppid() != parent_pid:
                on_parent_death()
                return
            threading.Event().wait(poll_seconds)

    thread = threading.Thread(target=poll, name="gateway-parent-watchdog", daemon=True)
    thread.start()
    return thread


def metrics_fanout_endpoint() -> Optional[str]:
    """Where child 0 re-publishes the schedulers' KV metrics for its siblings."""
    pid = os.environ.get(ENV_PARENT_PID)
    if pid is None:
        return None
    return f"ipc://{tempfile.gettempdir()}/dynamo_sglang_gateway_metrics_{pid}"


def build_gateway_engine():
    """Child side: join the parent's engine through a ``TokenizerWorker``."""
    parent_pid = int(os.environ[ENV_PARENT_PID])
    start_parent_watchdog(parent_pid)
    attach = getattr(sgl.Engine, "attach_tokenizer_worker", None)
    if attach is not None:
        engine = attach(parent_pid)
        logging.info(
            "gateway child pid=%d attached via Engine.attach_tokenizer_worker",
            os.getpid(),
        )
        return engine

    from sglang.srt.managers.multi_tokenizer_mixin import (
        get_tokenizer_worker_class,
        read_from_shared_memory,
    )
    from sglang.srt.runtime_context import publish

    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{parent_pid}"
    )
    publish(server_args, role="tokenizer")
    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    tm = get_tokenizer_worker_class(server_args)(server_args, port_args)
    tm.max_req_input_len = scheduler_info["max_req_input_len"]
    tm.set_startup_time(scheduler_info["startup_time"])
    logging.info(
        "gateway child pid=%d registered TokenizerWorker ipc=%s",
        os.getpid(),
        port_args.tokenizer_ipc_name,
    )
    return GatewayEngine(tm, server_args, port_args, scheduler_info)


def _reap(proc: subprocess.Popen, timeout: Optional[float] = None) -> None:
    try:
        proc.wait(timeout=child_shutdown_timeout() if timeout is None else timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def serve_via_gateway_children(
    engine,
    count: int,
    shutdown_event: asyncio.Event,
    load_time: Optional[float] = None,
) -> None:
    from sglang.srt.managers.multi_tokenizer_mixin import write_data_for_multi_tokenizer

    shm = getattr(engine, "_multi_tokenizer_shm", None)
    owns_shm = shm is None
    if shm is None:
        scheduler_info = {
            **engine._scheduler_init_result.scheduler_infos[0],
            "startup_time": engine.tokenizer_manager.startup_time,
        }
        shm = write_data_for_multi_tokenizer(
            engine.port_args, engine.server_args, scheduler_info
        )
    argv = sys.argv[1:]
    procs: list[subprocess.Popen] = []
    try:
        for index in range(count):
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "dynamo.sglang", *argv],
                    env=child_environment(index, load_time),
                )
            )
        logging.info(
            "gateway parent pid=%d spawned %d children: %s",
            os.getpid(),
            count,
            [p.pid for p in procs],
        )
        while not shutdown_event.is_set():
            await asyncio.sleep(2)
            dead = [p for p in procs if p.poll() is not None]
            if dead and not shutdown_event.is_set():
                raise RuntimeError(
                    f"gateway child pid={dead[0].pid} exited rc={dead[0].returncode}"
                )
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()
        await asyncio.gather(*(asyncio.to_thread(_reap, p) for p in procs))
        if owns_shm:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
