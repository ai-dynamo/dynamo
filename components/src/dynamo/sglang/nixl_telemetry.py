# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Give each co-located SGLang scheduler its own NIXL Prometheus exporter port.

SGLang runs one scheduler process per node-local rank, and each of those
schedulers builds its own NIXL agent for KV transfer. They all inherit the one
``NIXL_TELEMETRY_PROMETHEUS_PORT`` the operator injects into the container, so
every rank after the first fails to bind and aborts, and the pod never reaches
Ready. See ``dynamo.common.utils.nixl_telemetry`` for why the answer is a
derived port rather than an ephemeral one.

The port has to be set inside the rank's own process, because NIXL reads the
variable when the agent is constructed. Two things rule out setting it from
the worker process before ``sgl.Engine(...)``:

* With the ``spawn`` start method only ``os.environ`` crosses into a child, so
  a patch applied in the worker process is not inherited -- it would look
  correct in a single-process test and do nothing in a deployment.
* Under ``--enable-dp-attention`` the worker process does not start the
  schedulers at all. It starts one data-parallel controller, and *that* process
  starts the schedulers, so a hook installed only in the worker process never
  runs anywhere near a rank.

SGLang supports exactly this case: ``Engine.run_scheduler_process_func`` is a
documented override point ("Some fields to allow people to override the server
args and launch processes for their private forks"), it is forwarded into the
data-parallel controller, and it is invoked in the scheduler process with that
scheduler's own arguments. Dynamo overrides it with a wrapper that fixes up the
environment and then calls SGLang's real entry point.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any

from dynamo.common.utils.nixl_telemetry import (
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

    ``gpu_id`` is the only argument that is unique per co-located scheduler in
    every SGLang parallelism mode. Tensor-parallel ranks restart from 0 in each
    data-parallel group, and pipeline ranks repeat across tensor-parallel
    groups, so neither is unique on its own; SGLang folds all of them into
    ``gpu_id`` precisely because it must name a distinct device per scheduler.
    Undoing ``base_gpu_id`` and ``gpu_id_step`` turns that device number back
    into a dense 0-based index.
    """
    base_gpu_id = getattr(server_args, "base_gpu_id", 0) or 0
    gpu_id_step = getattr(server_args, "gpu_id_step", 1) or 1
    return (gpu_id - base_gpu_id) // gpu_id_step


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
    does not scrape NIXL keeps SGLang's own entry point.
    """
    if nixl_prometheus_base_port() is None:
        return

    import sglang as sgl

    if not hasattr(sgl.Engine, "run_scheduler_process_func"):
        # Log rather than raise: without the override the deployment behaves as it
        # did before, and a metrics feature should not become a startup failure.
        logger.error(
            "sglang.Engine has no run_scheduler_process_func override point, so "
            "co-located ranks keep one shared %s and all but one will fail to "
            "bind their NIXL Prometheus exporter.",
            NIXL_TELEMETRY_PROMETHEUS_PORT_ENV,
        )
        return

    sgl.Engine.run_scheduler_process_func = staticmethod(
        run_scheduler_process_with_nixl_port
    )
