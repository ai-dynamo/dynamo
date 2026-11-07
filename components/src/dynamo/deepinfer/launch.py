# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Starting the ranks of an engine: this node's children (and rank 0 in this
process), the other nodes' ranks on ``--control-port``, and the watch that
exits the process when one of them is lost.

Used by the Dynamo worker and by the controller process, so it must not import
``dynamo._core`` or ``dynamo.common.backend``.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional

from dynamo.deepinfer.controller import exit_process
from dynamo.deepinfer.options import visible_devices
from dynamo.deepinfer.protocol import (
    ControlServer,
    EngineMeta,
    LocalRankHandle,
    RankHandle,
    ZmqRankHandle,
    wait_all_ready,
)
from dynamo.deepinfer.rank import RankRunner

logger = logging.getLogger(__name__)


class ChildWatch:
    """Exit the process if a rank dies while another thread is blocked.

    Rank 0's constructor waits in DeepInfer's collectives for every rank; a
    child that died never joins them, and nothing else can interrupt the wait.
    It runs for the engine's whole life: rank 0's inline step blocked in a
    collective cannot notice a dead child (killed by a signal, it sends no
    error reply) or, with ranks on other nodes (``server``), a lost node.
    Ranks that were sent CLOSE may exit. ``grace_s`` lets a thread that reads
    the dead rank's error reply fail with it first (and stop this watch).
    In a controller process it also exits once the Dynamo worker that started
    it (``parent_pid``) is gone.
    """

    def __init__(
        self,
        children: list[ZmqRankHandle],
        on_fatal: Callable[[str], None] = exit_process,
        interval_s: float = 0.5,
        server: Optional[ControlServer] = None,
        grace_s: float = 0.0,
        parent_pid: Optional[int] = None,
    ) -> None:
        self._children = list(children)
        self._on_fatal = on_fatal
        self._interval_s = interval_s
        self._server = server
        self._grace_s = grace_s
        self._parent_pid = parent_pid
        self._stop = threading.Event()
        # Daemon: interpreter exit joins other threads before the finalizer
        # that would stop this one (_shutdown_ranks) runs.
        self._thread = threading.Thread(
            target=self._run, name="di-child-watch", daemon=True
        )

    def __enter__(self) -> ChildWatch:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            reason = None
            if self._server is not None:
                reason = self._server.watch(self._interval_s)
            elif self._stop.wait(self._interval_s):
                return
            for child in self._children:
                code = None if child.closing else child.proc.poll()
                if reason is None and code is not None:
                    reason = f"rank {child.rank} exited with code {code}"
            if (
                reason is None
                and self._parent_pid is not None
                and os.getppid() != self._parent_pid
            ):
                reason = "the Dynamo worker that started this controller exited"
            if reason is not None:
                if not self._stop.wait(self._grace_s):
                    self._on_fatal(reason)
                return


@dataclass
class Launch:
    """The ranks :func:`launch_ranks` started. :meth:`close` after the CLOSE
    iteration (``Controller.close``); idempotent."""

    handles: list[RankHandle]
    meta: EngineMeta
    ipc_dir: str
    server: Optional[ControlServer] = None
    watch: Optional[ChildWatch] = None

    def close(self) -> None:
        if self.watch is not None:
            self.watch.stop()
        if self.server is not None:
            self.server.close()
        shutil.rmtree(self.ipc_dir, ignore_errors=True)


def launch_ranks(
    args: Any,
    on_fatal: Callable[[str], None] = exit_process,
    *,
    raw_outputs: bool = False,
    parent_pid: Optional[int] = None,
) -> Launch:
    """Spawn this node's child ranks, then build rank 0 here; wait until every
    rank of the engine (other nodes' too) is READY.

    Children start first: every rank rendezvouses in DeepInfer's process-group
    init, so rank 0's constructor would block without them. ``on_fatal`` runs
    when a rank dies while another thread may be blocked in a collective.
    ``args`` holds the DeepInfer options (``DeepInferOptions``).
    ``raw_outputs`` and ``parent_pid``: for a controller process (``Controller``
    with ``raw_outputs``; exit with the Dynamo worker ``parent_pid``).
    """
    specs = args.rank_specs(engine_id=args.engine_id())
    local = args.local_ranks
    devices = visible_devices(len(local))
    ipc_dir = tempfile.mkdtemp(prefix="dynamo-deepinfer-")
    handles: list[RankHandle] = []
    server: Optional[ControlServer] = None
    watch: Optional[ChildWatch] = None
    first_child = 1 if args.rank0_in_process else 0
    try:
        if args.nnodes > 1:
            remote = [r for r in range(args.dp_size) if r not in local]
            server = ControlServer(
                args.control_port, remote, args.node_timeout_s, raw_outputs
            )
            logger.info(
                "DeepInfer controller listening on port %d for ranks %d..%d "
                "on %d other nodes",
                args.control_port,
                remote[0],
                remote[-1],
                args.nnodes - 1,
            )
            handles += [server.handle(r, specs[r].digest()) for r in remote]
        children: list[ZmqRankHandle] = []
        for rank in local[first_child:]:
            env = {"CUDA_VISIBLE_DEVICES": devices[rank - local[0]]}
            children.append(ZmqRankHandle.spawn(specs[rank], ipc_dir, env, raw_outputs))
        handles += children
        if children or server is not None or parent_pid is not None:
            # A dead child or a lost node would leave rank 0 blocked in a
            # collective, now and at any step later.
            watch = ChildWatch(
                children, on_fatal, server=server, grace_s=2.0, parent_pid=parent_pid
            )
            watch.start()
        if args.rank0_in_process:
            # Before DeepInfer (torch) initializes CUDA in this process.
            os.environ["CUDA_VISIBLE_DEVICES"] = devices[0]
            handles.append(LocalRankHandle(RankRunner(specs[local[0]])))
        handles.sort(key=lambda h: h.rank)
        metas = wait_all_ready(handles, args.startup_timeout_s)
    except BaseException:
        for handle in handles:
            handle.close(0.0)
        if watch is not None:
            watch.stop()
        if server is not None:
            server.close()
        shutil.rmtree(ipc_dir, ignore_errors=True)
        raise
    blocks = {m.num_gpu_blocks for m in metas}
    if len(blocks) > 1:
        logger.warning("DeepInfer ranks report different KV block counts: %s", blocks)
    meta = metas[0]
    meta.num_gpu_blocks = min(blocks)
    logger.info(
        "DeepInfer ranks ready: dp_size=%d nnodes=%d %s",
        len(handles),
        args.nnodes,
        meta,
    )
    return Launch(handles, meta, ipc_dir, server, watch)
