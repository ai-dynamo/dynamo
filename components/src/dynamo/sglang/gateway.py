# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process request gateway for the SGLang worker: the leader keeps the engine
and spawns N ``dynamo.sglang`` children that each serve requests through their own
SGLang ``TokenizerWorker``. Design notes: AGENTS.md, "Multi-process gateway".
"""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import tempfile
import types
from typing import Optional

import sglang as sgl

from dynamo.common.snapshot.constants import SNAPSHOT_CONTROL_DIR_ENV

ENV_PARENT_PID = "DYN_SGLANG_GATEWAY_PARENT_PID"
ENV_CHILD_INDEX = "DYN_SGLANG_GATEWAY_CHILD_INDEX"
ENV_SYSTEM_PORT = "DYN_SYSTEM_PORT"
ENV_SYSTEM_PORT_BASE = "DYN_SGLANG_GATEWAY_SYSTEM_PORT"

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

    The engine runs exactly one SGLang tokenizer worker per gateway process, so two
    explicit, different values are an error rather than a silent choice."""
    requested = getattr(dynamo_args, "gateway_workers", None)
    tokenizer_workers = getattr(server_args, "tokenizer_worker_num", 1) or 1
    if requested is None:
        return tokenizer_workers
    if tokenizer_workers > 1 and tokenizer_workers != requested:
        raise ValueError(
            f"--gateway-workers {requested} conflicts with --tokenizer-worker-num "
            f"{tokenizer_workers}: each gateway process is one SGLang tokenizer "
            "worker, so set only one of the two flags or give them the same value"
        )
    return requested


def gateway_worker_count(server_args, dynamo_args) -> int:
    """Children this process must spawn; 1 on non-leader nodes and inside children."""
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


def reserve_system_port_for_children() -> None:
    """Leader side, before the runtime starts. The leader serves no requests, so it
    gives ``DYN_SYSTEM_PORT`` to the children: child 0 takes the configured port,
    child i takes port + i, and the leader runs without a system status server."""
    raw = os.environ.get(ENV_SYSTEM_PORT)
    try:
        port = int(raw) if raw is not None else -1
    except ValueError:
        return
    if port <= 0:
        return
    os.environ[ENV_SYSTEM_PORT_BASE] = str(port)
    os.environ[ENV_SYSTEM_PORT] = "-1"


def child_environment(index: int) -> dict[str, str]:
    env = {
        **os.environ,
        ENV_PARENT_PID: str(os.getpid()),
        ENV_CHILD_INDEX: str(index),
    }
    base = env.pop(ENV_SYSTEM_PORT_BASE, None)
    if base is not None:
        env[ENV_SYSTEM_PORT] = str(int(base) + index)
    return env


def metrics_fanout_endpoint() -> Optional[str]:
    """Where child 0 re-publishes the schedulers' KV metrics for its siblings."""
    pid = os.environ.get(ENV_PARENT_PID)
    if pid is None:
        return None
    return f"ipc://{tempfile.gettempdir()}/dynamo_sglang_gateway_metrics_{pid}"


def build_gateway_engine():
    """Child side: join the parent's engine through a ``TokenizerWorker``."""
    parent_pid = int(os.environ[ENV_PARENT_PID])
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


def _reap(proc: subprocess.Popen, timeout: float = 60) -> None:
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


async def serve_via_gateway_children(
    engine, count: int, shutdown_event: asyncio.Event
) -> None:
    """Parent side: publish the engine's launch data, spawn ``count`` children and
    keep the engine alive until shutdown or until a child dies."""
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
                    env=child_environment(index),
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
