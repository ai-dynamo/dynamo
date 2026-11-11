# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import logging
import signal
from collections import defaultdict
from contextlib import contextmanager

import psutil  # type: ignore[import-untyped]

from dynamo.common.utils.worker_shutdown import WorkerShutdown


def coordinate_non_leader_shutdown(engine, shutdown: WorkerShutdown | None) -> None:
    """Keep follower schedulers alive while rank 0 drains requests and KV reads."""
    if shutdown is None or engine.server_args.node_rank == 0:
        return

    async def wait_for_schedulers() -> None:
        # Rank 0's ShutdownReq stops every rank after its drain. Do not run
        # block_until_scheduler_exits in an executor: cancellation cannot stop
        # its join thread, which would then hold interpreter shutdown open.
        get_pids = getattr(engine, "get_all_child_pids", None)
        pids = get_pids() if callable(get_pids) else None
        if not pids:
            logging.warning(
                "SGLang does not expose scheduler PIDs; keeping follower alive "
                "until the shutdown drain deadline"
            )
            await asyncio.Event().wait()
            return

        processes = []
        for pid in pids:
            try:
                processes.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
        while processes:
            alive = []
            for process in processes:
                try:
                    if (
                        process.is_running()
                        and process.status() != psutil.STATUS_ZOMBIE
                    ):
                        alive.append(process)
                except psutil.NoSuchProcess:
                    continue
            processes = alive
            if processes:
                await asyncio.sleep(0.05)

    # Only the serving leader knows when remote KV reads are done. Its
    # scheduler lifetime includes that wait; a second follower wait is redundant.
    shutdown.prefill = False
    shutdown.wait_for_children = wait_for_schedulers


@contextmanager
def defer_engine_signals(loop: asyncio.AbstractEventLoop, shutdown: WorkerShutdown):
    """Keep engine signal callbacks behind engine cleanup and runtime teardown."""
    signals = (signal.SIGTERM, signal.SIGINT)
    callbacks: dict[int, list] = defaultdict(list)
    previous = {}
    signum = None
    original_add = loop.add_signal_handler

    def on_signal(sig, _frame):
        nonlocal signum
        signum = sig
        shutdown.request_shutdown(sig)

    def capture(sig, callback, *args):
        if sig in signals:
            callbacks[sig].append((callback, args))
        else:
            original_add(sig, callback, *args)

    async def run_deferred():
        if signum is None:
            return
        for callback, args in callbacks.get(signum, []):
            result = callback(*args)
            if inspect.isawaitable(result):
                await result

    for sig in signals:
        previous[sig] = signal.signal(sig, on_signal)
    loop.add_signal_handler = capture  # type: ignore[method-assign, assignment]
    try:
        yield run_deferred
    finally:
        loop.add_signal_handler = original_add  # type: ignore[method-assign]
        for sig, handler in previous.items():
            signal.signal(sig, handler)
