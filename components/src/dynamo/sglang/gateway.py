# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process request gateway for the SGLang worker: the leader keeps the engine
and spawns N ``dynamo.sglang`` children that each serve requests through their own
SGLang ``TokenizerWorker``. Design notes: AGENTS.md, "Multi-process gateway".
"""

from __future__ import annotations

import asyncio
import copy
import logging
import os
import subprocess
import sys
import tempfile
import types

import sglang as sgl

ENV_PARENT_PID = "DYN_SGLANG_GATEWAY_PARENT_PID"
ENV_CHILD_INDEX = "DYN_SGLANG_GATEWAY_CHILD_INDEX"


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
    """SGLang's schedulers push KV metrics to one PULL socket and publish forward-pass
    metrics once, so exactly one gateway process may consume them: child 0 (or the
    single worker when the mode is off)."""
    return gateway_child_index() == 0


def gateway_worker_count(server_args, dynamo_args) -> int:
    """Number of gateway processes to run for this engine; 1 disables the mode."""
    if getattr(server_args, "tokenizer_worker_num", 1) <= 1:
        return 1
    if (getattr(server_args, "node_rank", 0) or 0) != 0:
        return 1
    if is_gateway_child():
        return 1
    n = getattr(dynamo_args, "gateway_workers", None)
    return n if n else server_args.tokenizer_worker_num


def _private_metrics_ipc(port_args):
    # Only one process may bind the PULL socket the schedulers push KV metrics to;
    # every other child gets an idle private one so its publisher can still start.
    port_args = copy.copy(port_args)
    port_args.metrics_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    return port_args


def _child_port_args(port_args):
    return port_args if owns_engine_metrics() else _private_metrics_ipc(port_args)


def build_gateway_engine():
    """Child side: join the parent's engine through a ``TokenizerWorker``."""
    parent_pid = int(os.environ[ENV_PARENT_PID])
    attach = getattr(sgl.Engine, "attach_tokenizer_worker", None)
    if attach is not None:
        engine = attach(parent_pid)
        engine.port_args = _child_port_args(engine.port_args)
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
    port_args = _child_port_args(port_args)
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
            env = {
                **os.environ,
                ENV_PARENT_PID: str(os.getpid()),
                ENV_CHILD_INDEX: str(index),
            }
            procs.append(
                subprocess.Popen(
                    [sys.executable, "-m", "dynamo.sglang", *argv], env=env
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
            if dead:
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
