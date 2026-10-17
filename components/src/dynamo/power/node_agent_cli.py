# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The node_agent's command line: flags -> :class:`NodeAgentSettings`.

Kept apart from :mod:`dynamo.power.node_agent` because that module imports
EPD and NVML at the top; this one imports only the standard library and
:mod:`dynamo.power.config`, so the CLI contract (defaults, the repeated and
comma-separated ``--metrics-url`` and ``--node-buffers`` forms, node-name
resolution, the ``--min-tgp-w`` refusal) is unit-testable on a machine
without either.
"""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import Mapping

from dynamo.power.config import (
    ALLOCATIONS,
    DEFAULT_ALLOCATION,
    DEFAULT_INTERVAL_S,
    DEFAULT_NODE_UBPC_BUFFER,
    DEFAULT_PERF_THRESHOLD,
    DEFAULT_WARMUP_POLICY,
    MODES,
    WARMUP_POLICIES,
    NodeAgentSettings,
    PowerControlConfigError,
    parse_fom,
    parse_gpu_indices,
    parse_metrics_urls,
    parse_node_buffers,
    parse_nodelist,
)

logger = logging.getLogger(__name__)

MIN_TGP_REFUSAL = (
    "--min-tgp-w is not enforced by EPD: its cap law "
    "(nupd._calculate_power_limits) has no lower floor and MIN_TGP is read only "
    "by the history-based ensemble, which this agent keeps off "
    "(NUPD_SKIP_HISTORY=1). Give the floor to the applier as --min-w-per-gpu "
    "instead; the launcher passes power_control.min_tgp_w there."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dynamo.power.node_agent",
        description="EPD node agent for one GPU group of a Dynamo power pool.",
    )
    parser.add_argument("--pool-name", required=True)
    parser.add_argument(
        "--rank",
        required=True,
        type=int,
        help="0 = the pool's controller, whatever the roster order",
    )
    parser.add_argument(
        "--nodelist", required=True, help="comma-separated pool roster, rank order"
    )
    parser.add_argument("--log-dir", required=True, help="EPD's LOG_DIR for this pool")
    parser.add_argument(
        "--gpu-indices",
        required=True,
        help="comma-separated GPU indices of this group (host NVML index == "
        "/dev/nvidiaN minor number)",
    )
    parser.add_argument("--mode", required=True, choices=MODES)
    parser.add_argument("--budget-w-per-gpu", required=True, type=int)
    parser.add_argument(
        "--fom",
        required=True,
        help="preset | prom:<metric>[:<k>=<v>,...][:scope=<frontend|pool_workers>]"
        ":<counter|gauge>:<sum|avg> | python:<module>:<Class>",
    )
    parser.add_argument("--interval-s", type=int, default=DEFAULT_INTERVAL_S)
    parser.add_argument("--perf-threshold", type=float, default=DEFAULT_PERF_THRESHOLD)
    parser.add_argument("--ubpc-buffer", type=float, default=DEFAULT_NODE_UBPC_BUFFER)
    parser.add_argument(
        "--allocation",
        choices=ALLOCATIONS,
        default=DEFAULT_ALLOCATION,
        help="NUPD's cap law: draw = caps proportional to each node's mean draw "
        "(EPD's historical law); need = nodes ranked by the cap they need, mean "
        "draw x their buffer, so a bursty prefill node keeps its burst headroom",
    )
    parser.add_argument(
        "--node-buffers",
        action="append",
        default=[],
        help="per-node UBPC buffers as node=buffer, comma-separated or repeated; "
        "a node without an entry keeps --ubpc-buffer",
    )
    parser.add_argument(
        "--min-tgp-w",
        type=int,
        default=None,
        help="refused: EPD does not enforce a floor; use the applier's "
        "--min-w-per-gpu",
    )
    parser.add_argument("--max-tgp-w", type=int, default=None)
    parser.add_argument(
        "--warmup-policy", choices=WARMUP_POLICIES, default=DEFAULT_WARMUP_POLICY
    )
    parser.add_argument(
        "--metrics-url",
        action="append",
        default=[],
        help="/metrics URL(s), comma-separated or repeated, optionally tagged "
        "frontend=<url> or worker=<url> (what a FoM scope selects); required on "
        "rank 0",
    )
    parser.add_argument("--gpus-per-node", type=int, default=None)
    parser.add_argument(
        "--pool-gpu-count",
        type=int,
        default=None,
        help="GPUs in the whole pool, for per-GPU FoM normalisation; the launcher "
        "passes it when nodes hold unequal shares of the pool (TEP6 tails)",
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> NodeAgentSettings:
    """Turn parsed flags into validated :class:`NodeAgentSettings`."""
    if args.min_tgp_w is not None:
        raise PowerControlConfigError(MIN_TGP_REFUSAL)
    return NodeAgentSettings(
        pool_name=args.pool_name,
        rank=args.rank,
        nodelist=parse_nodelist(args.nodelist),
        log_dir=os.path.abspath(args.log_dir),
        gpu_indices=parse_gpu_indices(args.gpu_indices),
        mode=args.mode,
        budget_w_per_gpu=args.budget_w_per_gpu,
        fom=parse_fom(args.fom),
        interval_s=args.interval_s,
        perf_threshold=args.perf_threshold,
        ubpc_buffer=args.ubpc_buffer,
        max_tgp_w=args.max_tgp_w,
        warmup_policy=args.warmup_policy,
        metrics_urls=parse_metrics_urls(args.metrics_url),
        gpus_per_node=args.gpus_per_node,
        pool_gpu_count_override=args.pool_gpu_count,
        allocation=args.allocation,
        node_buffers=parse_node_buffers(",".join(args.node_buffers)),
    )


def resolve_node_name(
    settings: NodeAgentSettings, environ: Mapping[str, str] | None = None
) -> str:
    """This node's name: ``SLURMD_NODENAME`` under SLURM, else the roster entry.

    EPD gates the controller on the rank it was given, not on the roster, so
    a node whose roster position disagrees with ``--rank`` only earns a
    warning: the process started with ``--rank 0`` decides.
    """
    env = os.environ if environ is None else environ
    node = env.get("SLURMD_NODENAME") or settings.nodelist[settings.rank]
    if node not in settings.nodelist:
        raise PowerControlConfigError(
            f"this node ({node}) is not in the pool roster {list(settings.nodelist)}"
        )
    if settings.nodelist[settings.rank] != node:
        logger.warning(
            "rank %d runs on %s but the roster puts %s at that rank; the node "
            "started with --rank 0 is the controller regardless of roster order",
            settings.rank,
            node,
            settings.nodelist[settings.rank],
        )
    return node


__all__ = [
    "MIN_TGP_REFUSAL",
    "build_parser",
    "resolve_node_name",
    "settings_from_args",
]
