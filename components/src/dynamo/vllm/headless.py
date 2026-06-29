# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless multi-node worker mode for the vLLM backend.

Shared by the legacy entry point (``dynamo.vllm.main``) and the unified
entry point (``dynamo.vllm.unified_main``). Secondary nodes in a multi-node
TP/PP (or ``mp`` data-parallel) deployment run vLLM workers only — no engine
core, no scheduler, no Dynamo endpoints — bypassing DistributedRuntime
entirely (no NATS/etcd).

``--headless`` is the multi-node mechanism for ``--data-parallel-backend mp``
(vLLM asserts ``nnodes > 1`` only with the ``mp`` backend). It is distinct
from — and mutually exclusive with — elastic-EP scaling, which requires the
Ray DP backend (``--data-parallel-backend ray``) with ``nnodes == 1``; there
the head node's engine schedules DP-worker Ray actors across the Ray cluster,
so secondary nodes run neither this headless mode nor the backend at all.
"""

from __future__ import annotations

import argparse
import logging

from .args import Config

logger = logging.getLogger(__name__)
GMS_VLLM_WORKER_CLS = "gpu_memory_service.integrations.vllm.worker.GMSWorker"


def _configure_gms_vllm_worker(engine_args) -> None:
    current = getattr(engine_args, "worker_cls", None)
    if current not in (None, "auto", GMS_VLLM_WORKER_CLS):
        logger.warning(
            "[GMS] Overriding user-provided vLLM worker_cls=%s with %s because "
            "--load-format=gms requires the GMS worker integration",
            current,
            GMS_VLLM_WORKER_CLS,
        )
    engine_args.worker_cls = GMS_VLLM_WORKER_CLS
    import gpu_memory_service.integrations.vllm.worker  # noqa: F401


def build_headless_namespace(config: Config) -> argparse.Namespace:
    """Build an argparse Namespace from engine_args for vLLM's run_headless().

    run_headless() expects the raw CLI namespace. We reconstruct it from
    the already-parsed AsyncEngineArgs so parse_args() doesn't need to
    leak transport details.
    """
    ns = argparse.Namespace(**vars(config.engine_args))
    # run_headless() reads api_server_count; default to 0 (no API server)
    if not hasattr(ns, "api_server_count"):
        ns.api_server_count = 0
    return ns


def run_dynamo_headless(config: Config) -> None:
    """Run in headless mode for multi-node TP/PP.

    Secondary nodes spawn vLLM workers only — no engine core, no scheduler,
    no Dynamo endpoints. Bypasses DistributedRuntime entirely (no NATS/etcd).
    """
    # Propagate worker_cls for custom load formats so headless workers use
    # the same model loader and patches as the leader node.
    if config.engine_args.load_format == "gms":
        _configure_gms_vllm_worker(config.engine_args)

        if config.gms_shadow_mode:
            from gpu_memory_service.integrations.vllm.utils import (
                configure_gms_lock_mode,
                configure_mx_ports,
            )

            configure_gms_lock_mode(config.engine_args)
            configure_mx_ports(config.engine_args)

    # ModelExpress uses vLLM's plugin path with --load-format=modelexpress.
    # Dynamo does not set a custom worker class here.

    # Detect failed secondary ranks before the NCCL collective timeout.
    _maybe_start_vllm_rank_liveness_client(config)

    # Keep the upstream CLI import local so tests that only exercise
    # build_headless_namespace() do not pull in vLLM's full CLI import graph.
    from vllm.entrypoints.cli.serve import run_headless

    args = build_headless_namespace(config)
    run_headless(args)


def _maybe_start_vllm_rank_liveness_client(config: Config) -> None:
    """Start the worker-side ZMQ rank-liveness channel for headless nodes."""
    from dynamo.common import rank_liveness as rl

    if not rl.liveness_enabled() or not config.gms_shadow_mode:
        return
    engine_args = config.engine_args
    node_rank = int(getattr(engine_args, "node_rank", 0) or 0)
    leader_host = getattr(engine_args, "master_addr", None)
    nnodes = int(getattr(engine_args, "nnodes", 1) or 1)
    if nnodes <= 1 or node_rank < 1 or not leader_host:
        return
    rl.RankLivenessClient(leader_host, node_rank).start()
