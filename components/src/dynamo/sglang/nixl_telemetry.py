# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give each co-located SGLang scheduler its own NIXL Prometheus exporter port.

SGLang runs one scheduler process per node-local rank, each building its own
NIXL agent, so the port must be set inside the rank's own process: NIXL reads
``NIXL_TELEMETRY_PROMETHEUS_PORT`` when the agent is constructed, ``spawn``
carries only ``os.environ`` into a child, and under ``--enable-dp-attention``
the schedulers are started by a data-parallel controller rather than by the
worker process. ``Engine.run_scheduler_process_func`` is SGLang's documented
override point for exactly this: it is forwarded through the data-parallel
controller and invoked in the scheduler process with that scheduler's own
arguments. The wrapper below fixes up the environment and then calls SGLang's
real entry point. See ``dynamo.common.utils.nixl_telemetry`` for the
derivation.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from dynamo.common.utils.nixl_telemetry import (
    NIXL_TELEMETRY_ENABLE_ENV,
    NIXL_TELEMETRY_PROMETHEUS_PORT_ENV,
    derive_nixl_prometheus_port,
    nixl_prometheus_base_port,
)

logger = logging.getLogger(__name__)


def _node_local_launch_shape(server_args: Any) -> tuple[int, int]:
    """Return SGLang's own split of the pipeline and tensor dimensions per node."""
    nnodes = getattr(server_args, "nnodes", 1) or 1
    tp_size = getattr(server_args, "tp_size", 1) or 1
    pp_size = getattr(server_args, "pp_size", 1) or 1

    pp_size_per_node = max(pp_size // nnodes, 1)
    nnodes_per_tp_group = max(nnodes // pp_size, 1)
    tp_size_per_node = max(tp_size // nnodes_per_tp_group, 1)
    if getattr(server_args, "is_ep_scale_joiner", False):
        # A scale joiner enumerates its whole tensor-parallel span on one node.
        tp_size_per_node = tp_size

    return pp_size_per_node, tp_size_per_node


def _node_local_rank_count(server_args: Any, *, dp_rank: int | None) -> int:
    """Return how many ranks this launch places on one node.

    This is the width of the port range the pod reserves, so it has to count
    the same schedulers ``_node_local_rank`` numbers: the launch places one on
    each node-local pipeline and tensor position, and one such group per
    data-parallel rank whenever ``dp_rank`` names a separate launch group.
    """
    pp_size_per_node, tp_size_per_node = _node_local_launch_shape(server_args)
    ranks = pp_size_per_node * tp_size_per_node
    if dp_rank is not None and not getattr(server_args, "enable_dp_attention", False):
        ranks *= max(getattr(server_args, "dp_size", 1) or 1, 1)

    return ranks


def _node_local_rank(
    server_args: Any,
    *,
    tp_rank: int,
    pp_rank: int,
    dp_rank: int | None,
) -> int:
    """Return the scheduler's index among the ranks sharing this node.

    SGLang places a scheduler on a device with ``gpu_id = base_gpu_id +
    dp_offset + (pp_rank % pp_size_per_node) * tp_size_per_node + (tp_rank %
    tp_size_per_node) * gpu_id_step``. That device index is not a port index:
    ``gpu_id_step`` spaces it out, ``base_gpu_id`` and the data-parallel offset
    shift it, and nothing bounds it by the number of ports the pod reserves.
    The rank arguments the same launch already computes give the position
    directly, and counting the pairs this node launches numbers them 0, 1, 2,
    ... with no gaps.

    ``dp_rank`` names a separate launch group only when SGLang's data-parallel
    controller starts one tensor-parallel group per data-parallel rank. Under
    ``--enable-dp-attention`` all ranks share one group and ``dp_rank`` is
    derived from ``tp_rank``, so folding it in there would hand two schedulers
    the same number.
    """
    pp_size_per_node, tp_size_per_node = _node_local_launch_shape(server_args)

    rank = (pp_rank % pp_size_per_node) * tp_size_per_node + (
        tp_rank % tp_size_per_node
    )
    if dp_rank is not None and not getattr(server_args, "enable_dp_attention", False):
        rank += dp_rank * pp_size_per_node * tp_size_per_node

    return rank


def _assign_nixl_prometheus_port(target: Any, args: tuple, kwargs: dict) -> None:
    """Rewrite this process's exporter port before the NIXL agent is built."""
    base_port = nixl_prometheus_base_port()
    if base_port is None:
        return

    # SGLang moves scheduler arguments between releases, and a rank this
    # process was not given is a dimension it does not participate in, which is
    # what the defaults below describe.
    bound = inspect.signature(target).bind(*args, **kwargs)
    bound.apply_defaults()
    arguments = bound.arguments
    tp_rank = arguments.get("tp_rank") or 0
    pp_rank = arguments.get("pp_rank") or 0
    dp_rank = arguments.get("dp_rank")

    # Number from the resolved configuration, not from the raw ``ServerArgs``
    # the scheduler is handed: resolution is where attention data parallelism is
    # turned on, so the raw values would fold ``dp_rank`` in a second time and
    # send later ranks past the reserved range. Imported here because the module
    # is imported to reach ``install_per_rank_nixl_prometheus_ports()``, which
    # must stay a no-op without SGLang when telemetry is off.
    from dynamo.sglang._compat import resolved_server_args

    server_args = resolved_server_args(arguments["server_args"])
    local_rank = _node_local_rank(
        server_args, tp_rank=tp_rank, pp_rank=pp_rank, dp_rank=dp_rank
    )
    # The reservation is as wide as this launch, not as wide as a pod may ever
    # reserve: a four-rank launch that fits below the top of the port range, or
    # beside another listener eight ports up, is one the maximum would refuse.
    colocated_ranks = _node_local_rank_count(server_args, dp_rank=dp_rank)
    port = derive_nixl_prometheus_port(base_port, local_rank, max_ranks=colocated_ranks)
    os.environ[NIXL_TELEMETRY_PROMETHEUS_PORT_ENV] = str(port)
    logger.info(
        "NIXL Prometheus exporter for tp_rank=%s pp_rank=%s dp_rank=%s is "
        "node-local rank %s of %s and listens on port %s (base %s)",
        tp_rank,
        pp_rank,
        dp_rank,
        local_rank,
        colocated_ranks,
        port,
        base_port,
    )


def run_scheduler_process_with_nixl_port(*args: Any, **kwargs: Any) -> Any:
    """SGLang scheduler entry point that first claims this rank's exporter port.

    Must stay a module-level function: ``spawn`` pickles the process target by
    module and qualified name.
    """
    from sglang.srt.managers.scheduler import run_scheduler_process

    _assign_nixl_prometheus_port(run_scheduler_process, args, kwargs)
    return run_scheduler_process(*args, **kwargs)


def install_per_rank_nixl_prometheus_ports() -> None:
    """Point SGLang's scheduler launches at the wrapper, when telemetry is on.

    A no-op when NIXL Prometheus telemetry is disabled, so a deployment that
    does not scrape NIXL keeps SGLang's own entry point. Raises ``RuntimeError``
    when telemetry is on but this SGLang offers no override point.
    """
    if nixl_prometheus_base_port() is None:
        return

    # Take the class from the module that defines it: ``sglang.Engine`` is a
    # lazy proxy, so an assignment through it would land on the proxy object and
    # leave every scheduler on SGLang's own entry point.
    from sglang.srt.entrypoints.engine import Engine

    if not hasattr(Engine, "run_scheduler_process_func"):
        raise RuntimeError(
            f"this SGLang has no Engine.run_scheduler_process_func override "
            f"point, so co-located ranks cannot be given distinct "
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV} values and all but one would "
            f"fail to bind their NIXL Prometheus exporter. Run a supported "
            f"SGLang version, or set {NIXL_TELEMETRY_ENABLE_ENV}=n to serve "
            f"without NIXL telemetry."
        )

    Engine.run_scheduler_process_func = staticmethod(
        run_scheduler_process_with_nixl_port
    )
