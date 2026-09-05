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

# SGLang reindexes CUDA_VISIBLE_DEVICES per child when this is set, collapsing every
# scheduler's gpu_id to 0 -- the only node-local rank index the process is handed.
_ONE_VISIBLE_DEVICE_ENV = "SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS"

_TRUTHY = frozenset({"1", "true", "yes", "y", "on"})


def _node_local_rank(server_args: Any, gpu_id: int) -> int:
    """Return the scheduler's index among the ranks sharing this node.

    ``gpu_id`` is the argument that stays distinct per co-located scheduler in
    every parallelism mode, which is what keeps two ranks off one port;
    ``tp_rank`` restarts at 0 in each data-parallel group. Pipeline stages are
    already numbered densely, because SGLang does not scale a stage's device
    shift by ``gpu_id_step``, so only the other modes divide the step out.
    """
    base_gpu_id = getattr(server_args, "base_gpu_id", 0) or 0
    gpu_id_step = getattr(server_args, "gpu_id_step", 1) or 1
    pp_size = getattr(server_args, "pp_size", 1) or 1
    offset = gpu_id - base_gpu_id
    return offset if pp_size > 1 else offset // gpu_id_step


def _assign_nixl_prometheus_port(target: Any, args: tuple, kwargs: dict) -> None:
    """Rewrite this process's exporter port before the NIXL agent is built."""
    base_port = nixl_prometheus_base_port()
    if base_port is None:
        return

    if os.environ.get(_ONE_VISIBLE_DEVICE_ENV, "").strip().lower() in _TRUTHY:
        raise ValueError(
            f"{_ONE_VISIBLE_DEVICE_ENV} hides each scheduler's device index, so "
            f"co-located ranks cannot be given distinct "
            f"{NIXL_TELEMETRY_PROMETHEUS_PORT_ENV} values and all but one would "
            f"fail to bind. Unset {_ONE_VISIBLE_DEVICE_ENV} or disable NIXL "
            f"Prometheus telemetry."
        )

    bound = inspect.signature(target).bind(*args, **kwargs)
    bound.apply_defaults()
    server_args = bound.arguments["server_args"]
    gpu_id = bound.arguments["gpu_id"]

    port = derive_nixl_prometheus_port(base_port, _node_local_rank(server_args, gpu_id))
    os.environ[NIXL_TELEMETRY_PROMETHEUS_PORT_ENV] = str(port)
    logger.info(
        "NIXL Prometheus exporter for gpu_id=%s listens on port %s (base %s)",
        gpu_id,
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
