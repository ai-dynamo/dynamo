# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Disaggregated-serving helpers for the TokenSpeed Dynamo backend.

These helpers compute the bootstrap host/port the Dynamo runtime needs to
advertise via ``ModelRuntimeConfig.set_disaggregated_endpoint`` and validate
that prefill / decode workers are configured compatibly. TokenSpeed's KV
transport is Mooncake (not NIXL); the engine itself owns the actual KV
movement once both workers agree on ``{bootstrap_host, bootstrap_port,
bootstrap_room}``.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Tuple

from dynamo.common.constants import DisaggregationMode

# Env override for the bootstrap host the prefill worker advertises. Useful
# in container/k8s setups where ``socket.gethostname()`` returns a name the
# decode worker can't resolve.
BOOTSTRAP_HOST_ENV = "DYN_TOKENSPEED_BOOTSTRAP_HOST"


def resolve_bootstrap_host(server_args: Any) -> str:
    """Pick the host string to advertise for KV bootstrap.

    Precedence: ``DYN_TOKENSPEED_BOOTSTRAP_HOST`` env var, then
    ``server_args.host`` (TokenSpeed's ``--host``), then the local hostname.
    """
    env_host = os.environ.get(BOOTSTRAP_HOST_ENV)
    if env_host:
        return env_host
    host = getattr(server_args, "host", None)
    if host:
        return str(host)
    return socket.gethostname()


def runtime_disaggregated_endpoint(server_args: Any) -> Tuple[str, int]:
    """Return ``(host, port)`` for ``set_disaggregated_endpoint``.

    Raises ``ValueError`` if TokenSpeed's bootstrap port isn't set — this is
    a configuration bug, not a runtime condition; failing fast is correct.
    """
    port = getattr(server_args, "disaggregation_bootstrap_port", None)
    if port is None:
        raise ValueError(
            "TokenSpeed prefill worker requires --disaggregation-bootstrap-port"
        )
    return resolve_bootstrap_host(server_args), int(port)


def validate_disagg_compatibility(
    mode: DisaggregationMode, server_args: Any
) -> None:
    """Cheap pre-flight checks before the engine is started.

    The user can extend this — see TODO below — to enforce additional
    compatibility constraints they care about (e.g. block size matching
    across workers, model parallelism caps, attention backend).
    """
    if mode == DisaggregationMode.AGGREGATED:
        return

    # MVP constraint: TokenSpeed upstream KV transfer doesn't yet plumb
    # data_parallel_rank into ``Engine.async_generate`` (codex review,
    # tokenspeed/runtime/entrypoints/engine.py:248-312). Hard-fail rather
    # than silently mis-route requests.
    dp_size = getattr(server_args, "dp_size", 1) or 1
    if dp_size > 1:
        raise ValueError(
            "TokenSpeed disaggregated mode does not yet support DP > 1; "
            f"got dp_size={dp_size}"
        )

    block_size = getattr(server_args, "block_size", None)
    if block_size is None:
        raise ValueError(
            "TokenSpeed disaggregated mode requires --block-size to be set "
            "explicitly so prefill and decode workers agree on KV page size"
        )

    # TODO(yna): validate any additional cross-worker invariants the
    # operator cares about (TP size, attention backend, dtype). The shape
    # of "compatible" is deployment-specific — leaving this as a learning-
    # mode contribution point rather than guessing.
