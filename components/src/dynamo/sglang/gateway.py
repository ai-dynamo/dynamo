# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-process request gateway for dynamo.sglang (``--tokenizer-worker-num N``).

One Python interpreter fronting a 16-rank engine tops out near 50 req/s (GIL): SGLang's
TokenizerManager intake plus Dynamo's per-token relay. SGLang's own multi-tokenizer mode
already shards that gateway across processes for ``sglang serve``; this reuses it. The
parent owns the engine (schedulers, MultiTokenizerRouter, bootstrap server) and serves
nothing; N spawned children each build a TokenizerWorker from the shared-memory args and
run the normal init_decode/init_prefill path against a thin engine facade.
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


class GatewayEngine:
    """What the handlers need from an ``sgl.Engine``: tokenizer_manager, server_args,
    port_args, async_generate and the representative scheduler info."""

    def __init__(self, tokenizer_manager, server_args, port_args, scheduler_info):
        self.tokenizer_manager = tokenizer_manager
        self.server_args = server_args
        self.port_args = port_args
        self._scheduler_init_result = types.SimpleNamespace(
            scheduler_infos=[scheduler_info]
        )

    async_generate = sgl.Engine.async_generate  # only touches self.tokenizer_manager
    _resolve_routed_dp_rank = sgl.Engine._resolve_routed_dp_rank  # stateless helper

    def shutdown(self):
        pass  # the parent owns the engine processes


def is_gateway_child() -> bool:
    return ENV_PARENT_PID in os.environ


def wants_gateway(server_args) -> bool:
    return (
        getattr(server_args, "tokenizer_worker_num", 1) > 1
        and (getattr(server_args, "node_rank", 0) or 0) == 0
        and not is_gateway_child()
    )


def build_gateway_engine():
    # SGLang with Engine.attach_tokenizer_worker (sgl-project/sglang fix for #15157)
    # returns a real Engine bound to a TokenizerWorker; older trees get the facade.
    attach = getattr(sgl.Engine, "attach_tokenizer_worker", None)
    if attach is not None:
        engine = attach(int(os.environ[ENV_PARENT_PID]))
        # The Dynamo publisher binds a PULL socket on port_args.metrics_ipc_name; the
        # parent's schedulers publish there and the parent already owns that bind.
        engine.port_args = copy.copy(engine.port_args)
        engine.port_args.metrics_ipc_name = (
            f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        )
        logging.info(
            "gateway child pid=%d: attached via Engine.attach_tokenizer_worker",
            os.getpid(),
        )
        return engine
    from sglang.srt.managers.multi_tokenizer_mixin import (
        get_tokenizer_worker_class,
        read_from_shared_memory,
    )
    from sglang.srt.runtime_context import publish

    parent = int(os.environ[ENV_PARENT_PID])
    port_args, server_args, scheduler_info = read_from_shared_memory(
        f"multi_tokenizer_args_{parent}"
    )
    publish(server_args, role="tokenizer")
    port_args.tokenizer_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    # The publisher binds a PULL socket on metrics_ipc_name; the parent owns the real one.
    port_args.metrics_ipc_name = (
        f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    )
    tm = get_tokenizer_worker_class(server_args)(server_args, port_args)
    tm.max_req_input_len = scheduler_info["max_req_input_len"]
    tm.set_startup_time(scheduler_info["startup_time"])
    logging.info(
        "gateway child pid=%d: TokenizerWorker registered (ipc=%s)",
        os.getpid(),
        port_args.tokenizer_ipc_name,
    )
    return GatewayEngine(tm, server_args, port_args, scheduler_info)


async def serve_via_gateway_children(
    engine, server_args, shutdown_event: asyncio.Event
) -> None:
    """Parent side: publish engine args to shm, spawn N `python -m dynamo.sglang` children
    with the same argv, and babysit them until shutdown or a child dies."""
    from sglang.srt.managers.multi_tokenizer_mixin import write_data_for_multi_tokenizer

    n = server_args.tokenizer_worker_num
    scheduler_info = {
        **engine._scheduler_init_result.scheduler_infos[0],
        "startup_time": engine.tokenizer_manager.startup_time,
    }
    # An Engine that already published its args (patched SGLang) is left alone.
    shm = getattr(engine, "_multi_tokenizer_shm", None)
    owns_shm = shm is None
    if owns_shm:
        shm = write_data_for_multi_tokenizer(
            engine.port_args, engine.server_args, scheduler_info
        )
    env = {**os.environ, ENV_PARENT_PID: str(os.getpid())}
    argv = sys.argv[1:]
    procs = [
        subprocess.Popen([sys.executable, "-m", "dynamo.sglang", *argv], env=env)
        for _ in range(n)
    ]
    logging.info(
        "gateway parent pid=%d spawned %d children: %s",
        os.getpid(),
        n,
        [p.pid for p in procs],
    )
    try:
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
        for p in procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()
        if owns_shm:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass
