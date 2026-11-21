# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The DeepInfer ranks of one non-head node of an engine that spans nodes.

    python -m dynamo.deepinfer.node --model PATH --dp-size D --nnodes N \\
        --node-rank K --master-addr HEAD [--master-port P] [--control-port C] ...

Spawns this node's ``R = D / N`` ranks (global ranks ``K*R .. K*R+R-1``, one
GPU each, in ``CUDA_VISIBLE_DEVICES`` order) as ``rank_worker`` processes that
connect to the head's controller on ``tcp://HEAD:C``. The head runs
``python -m dynamo.deepinfer`` with the same DeepInfer flags and
``--node-rank 0``; this launcher accepts the head's command line and ignores
the Dynamo runtime flags.

Exits 0 once every rank closed (the head's CLOSE iteration). When a rank fails
it kills the others and exits 1, so the job scheduler notices. SIGTERM/SIGINT
give the ranks ``--shutdown-grace-s`` to finish the head's CLOSE, then kill
them (a second signal kills at once). Must not import ``dynamo._core`` or
``dynamo.common.backend``.
"""

from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence
from typing import Any, Optional

from dynamo.deepinfer.options import (
    DeepInferOptions,
    add_deepinfer_arguments,
    apply_env_defaults,
    route_address,
    visible_devices,
)
from dynamo.deepinfer.protocol import spawn_rank_process, tcp_endpoint

logger = logging.getLogger(__name__)

# The head reaches CLOSE up to ~30 s after SIGTERM by default (a 5 s grace,
# then a prefill engine drains held KV for up to 25 s), and the CLOSE itself
# takes a while; the node's ranks must still be there.
DEFAULT_SHUTDOWN_GRACE_S = 90.0


def parse_node_args(
    argv: Optional[Sequence[str]] = None,
) -> tuple[DeepInferOptions, float, list[str]]:
    """This node's options, the shutdown grace and the ignored (head-only) flags."""
    parser = argparse.ArgumentParser(
        description="DeepInfer ranks of a non-head node (--node-rank >= 1) of an "
        "engine that spans nodes; the head runs python -m dynamo.deepinfer",
        allow_abbrev=False,
    )
    add_deepinfer_arguments(parser)
    parser.add_argument(
        "--shutdown-grace-s",
        type=float,
        default=DEFAULT_SHUTDOWN_GRACE_S,
        help="After SIGTERM/SIGINT, time the ranks get to finish the head's "
        "CLOSE before they are killed. Must exceed the head's shutdown: its "
        "grace period, a prefill engine's KV drain and the CLOSE "
        "(DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS + DYN_PREFILL_DRAIN_TIMEOUT_S, "
        "within DYN_WORKER_GRACEFUL_SHUTDOWN_TIMEOUT); raise them together.",
    )
    ns, ignored = parser.parse_known_args(argv)
    try:
        opts = DeepInferOptions.from_namespace(ns)
    except ValueError as e:
        parser.error(str(e))
    if opts.nnodes < 2 or opts.node_rank < 1:
        parser.error(
            "needs --nnodes >= 2 and --node-rank 1..nnodes-1; the head "
            "(--node-rank 0) runs python -m dynamo.deepinfer"
        )
    return opts, ns.shutdown_grace_s, ignored


def spawn_ranks(opts: DeepInferOptions) -> list[subprocess.Popen]:
    """Start this node's ranks; each connects to the head's controller."""
    local = opts.local_ranks
    devices = visible_devices(len(local))
    advertise = opts.advertise_addr or route_address(
        opts.master_addr, opts.control_port
    )
    endpoint = tcp_endpoint(opts.master_addr, opts.control_port)
    args = [
        "--advertise-addr",
        advertise,
        "--node-timeout-s",
        str(opts.node_timeout_s),
        "--startup-timeout-s",
        str(opts.startup_timeout_s),
    ]
    logger.info(
        "Node %d/%d: ranks %d..%d on GPUs %s, controller %s, KV events on %s:%d+",
        opts.node_rank,
        opts.nnodes,
        local[0],
        local[-1],
        ",".join(devices),
        endpoint,
        advertise,
        opts.kv_events_port_base,
    )
    procs: list[subprocess.Popen] = []
    try:
        for device, spec in zip(devices, opts.rank_specs(opts.engine_id(), local)):
            env = {"CUDA_VISIBLE_DEVICES": device}
            procs.append(spawn_rank_process(spec, endpoint, env, args))
    except BaseException:
        _kill(procs)
        raise
    return procs


def _kill(procs: Sequence[subprocess.Popen]) -> None:
    # Ranks ignore SIGTERM (a group signal must not stop one mid-iteration).
    for proc in procs:
        if proc.poll() is None:
            proc.kill()
    for proc in procs:
        proc.wait()


class Supervisor:
    """Waits for this node's ranks; kills them all when one fails or after a
    signal's grace period."""

    def __init__(self, ranks: Sequence[int], grace_s: float) -> None:
        self.ranks = list(ranks)
        self.grace_s = grace_s
        self.procs: list[subprocess.Popen] = []
        self._signum: Optional[int] = None
        self._deadline: Optional[float] = None

    def on_signal(self, signum: int, frame: Any) -> None:
        if self._signum is None:
            logger.warning(
                "%s: ranks have %.0fs to close",
                signal.Signals(signum).name,
                self.grace_s,
            )
            self._signum = signum
            self._deadline = time.monotonic() + self.grace_s
        else:
            self._deadline = time.monotonic()

    def run(self, poll_s: float = 0.1) -> int:
        while True:
            codes = [proc.poll() for proc in self.procs]
            for rank, code in zip(self.ranks, codes):
                if code is not None and code != 0:
                    logger.error("Rank %d exited with code %d; stopping", rank, code)
                    _kill(self.procs)
                    return 1
            if all(code == 0 for code in codes):
                logger.info("Ranks %s closed", self.ranks)
                return 0
            if self._deadline is not None and time.monotonic() >= self._deadline:
                logger.error("Ranks did not close in time; killing them")
                _kill(self.procs)
                return 128 + (self._signum or signal.SIGTERM)
            time.sleep(poll_s)


def main(argv: Optional[Sequence[str]] = None) -> int:
    # DYN_LOG may carry per-target filters ("info,dynamo=debug"); use the default.
    level = os.environ.get("DYN_LOG", "info").split(",")[0].upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s [deepinfer node] %(name)s: %(message)s",
    )
    opts, grace_s, ignored = parse_node_args(argv)
    if ignored:
        logger.info("Ignoring flags for the head's Dynamo worker: %s", ignored)
    apply_env_defaults(opts)
    supervisor = Supervisor(opts.local_ranks, grace_s)
    signal.signal(signal.SIGTERM, supervisor.on_signal)
    signal.signal(signal.SIGINT, supervisor.on_signal)
    supervisor.procs = spawn_ranks(opts)
    return supervisor.run()


if __name__ == "__main__":
    sys.exit(main())
