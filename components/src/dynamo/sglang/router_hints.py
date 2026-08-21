# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Router-hint capability advertisement for the SGLang backend.

Mirrors ``dynamo/vllm/router_hints.py``: a worker must advertise the
``router_hint`` runtime capability, its topology role, and a per-DP-rank map of
KV source control endpoints, or the router treats it as hint-incapable and
silently emits no hints.

The only difference from the vLLM version is where the endpoints come from.
vLLM reads ``kv_transfer_config.kv_connector_extra_config.secondary_tiers[]``;
SGLang has no secondary tiers, so the equivalent values live in the KVCR
HiCache storage backend's extra-config JSON
(``--hicache-storage-backend-extra-config``), whose ``control_advertise_host``
and ``control_port`` fields describe the ZMQ peer control channel the KVCR store
binds.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Optional

from dynamo.common.constants import (
    ROUTER_HINT_RUNTIME_CAPABILITY_KEY,
    ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY,
)

# The HiCache storage backend that speaks the router-hint protocol.
_KVCR_BACKEND_NAME = "kvcr"

# Hosts that identify a bind-any wildcard rather than a reachable peer address.
_UNROUTABLE_HOSTS = frozenset({"0.0.0.0", "::"})

# SGLang's disaggregation_mode spelled as the router's worker_type vocabulary.
# SGLang says "null" for a non-disaggregated engine; the router says
# "aggregated". The router only matches a hint source to a target when both
# report the same role, so this mapping has to agree with the vLLM backend's
# ``_router_hint_worker_type``.
_WORKER_TYPE_BY_DISAGGREGATION_MODE = {
    "null": "aggregated",
    "prefill": "prefill",
    "decode": "decode",
}


def _router_hint_worker_type(server_args: Any) -> Optional[str]:
    """Normalize SGLang's disaggregation mode into router-hint runtime metadata."""
    mode = getattr(server_args, "disaggregation_mode", None) or "null"
    return _WORKER_TYPE_BY_DISAGGREGATION_MODE.get(mode)


def _dp_port_stride(server_args: Any) -> int:
    """How many KVCR control ports one attention-DP rank of this engine owns.

    Unlike vLLM, where the KV secondary tier lives in the one scheduler process
    per DP rank, SGLang builds a HiCache storage backend in *every* attention
    rank's scheduler process. All of them read the same configured base port and
    offset it by their own rank coordinate, so a DP rank occupies a whole block
    of ports rather than a single one, and the next DP rank starts after that
    block.

    The block size is the number of schedulers per DP rank, which SGLang spells
    as ``attn_cp_size * attn_tp_size`` and which reduces to ``tp_size //
    dp_size`` (``attn_tp_size = tp_size // dp_size // attn_cp_size``). Without
    attention DP there is a single DP group and the stride is unused, but 1 is
    also the honest answer: the whole engine is one block.

    Must stay in step with ``_rank_port_offset`` in SGLang's
    ``mem_cache/storage/kvcr/kvcr_store.py``; a mismatch does not fail, it makes
    peers dial the wrong rank and silently fetch the wrong attention shard.
    """
    dp_size = getattr(server_args, "dp_size", 1) or 1
    if not getattr(server_args, "enable_dp_attention", False) or dp_size <= 1:
        return 1
    tp_size = getattr(server_args, "tp_size", 1) or 1
    return max(tp_size // dp_size, 1)


def _source_control_endpoint(
    extra_config: dict[str, Any], port_offset: int = 0
) -> Optional[str]:
    """Peer-reachable ZMQ control endpoint, or None if not advertisable.

    ``port_offset`` locates one DP rank's port block relative to the configured
    base port (see :func:`_dp_port_stride`). The endpoint names that block's
    first port, i.e. the DP rank's first attention rank; the consuming backend
    adds its own within-DP-group offset, since the router has no TP concept and
    cannot resolve that half itself.

    Only ``control_advertise_host`` names the address peers dial;
    ``control_host`` is the bind address and is legitimately a wildcard, so it
    is not a fallback. An ephemeral port (0) cannot be advertised either -- the
    bound port is only known inside the scheduler process, so registration has
    nothing to publish. The offset port is range-checked too, since a base port
    near the top of the range can carry the last DP rank past 65535.
    """
    try:
        control_port = int(extra_config.get("control_port")) + port_offset
    except (TypeError, ValueError):
        return None
    if not 0 < control_port <= 65535:
        return None
    host = extra_config.get("control_advertise_host")
    if not isinstance(host, str) or not host or host in _UNROUTABLE_HOSTS:
        return None
    return f"tcp://{host}:{control_port}"


def _source_control_endpoints(
    extra_config: dict[str, Any], dp_bounds: tuple[int, int], dp_port_stride: int
) -> Optional[dict[str, str]]:
    """Per-global-DP-rank endpoint map, or None if any rank is unresolvable.

    The router keys hint sources by ``(worker_id, dp_rank)``, so the map is
    keyed by *global* rank while the port offset follows the *local* rank -- on
    a multinode engine this node only owns its own slice of the global range,
    but each node numbers its ports from the same base. It is all-or-nothing: a
    partial map would let the router select a rank no peer can dial.
    """
    dp_start, dp_end = dp_bounds
    endpoints: dict[str, str] = {}
    for local_dp_rank in range(dp_end - dp_start):
        endpoint = _source_control_endpoint(
            extra_config, local_dp_rank * dp_port_stride
        )
        if endpoint is None:
            return None
        endpoints[str(dp_start + local_dp_rank)] = endpoint
    return endpoints


def enable_router_hint_support(
    runtime_config: Any,
    server_args: Any,
    extra_config: dict[str, Any],
    dp_bounds: tuple[int, int],
) -> None:
    """Advertise router-hint capability when this worker runs the KVCR backend.

    No-op unless the KVCR HiCache storage backend is selected and configured to
    consume hints -- a worker without a remote-capable KV source has nothing to
    serve a peer from. Raises when the backend is configured for hints but its
    control endpoints are not advertisable, since that combination would
    register a worker the router selects as a source but no peer can dial.

    All three runtime keys are set together: the router requires all of them, so
    publishing a subset produces a worker it silently never hints to.
    """
    if getattr(server_args, "hicache_storage_backend", None) != _KVCR_BACKEND_NAME:
        return
    if not extra_config.get("enable_remote_hint"):
        return

    worker_type = _router_hint_worker_type(server_args)
    if worker_type is None:
        return

    endpoints = _source_control_endpoints(
        extra_config, dp_bounds, _dp_port_stride(server_args)
    )
    if endpoints is None:
        raise ValueError(
            "router_hint support requires advertisable source control endpoints "
            "for all managed DP ranks; set control_advertise_host to a "
            "peer-reachable address and a control_port that keeps every rank "
            "within 1..65535 in --hicache-storage-backend-extra-config"
        )

    # set_engine_specific expects JSON text; the Rust side accepts a truthy
    # string for the capability flag but parses the other two as JSON values.
    runtime_config.set_engine_specific(
        ROUTER_HINT_RUNTIME_CAPABILITY_KEY, json.dumps(True)
    )
    runtime_config.set_engine_specific(
        ROUTER_HINT_WORKER_TYPE_RUNTIME_KEY, json.dumps(worker_type)
    )
    runtime_config.set_engine_specific(
        ROUTER_HINT_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
        json.dumps(endpoints),
    )
    logging.info(
        "Advertised router_hint capability (worker_type=%s) with source control "
        "endpoints %s",
        worker_type,
        endpoints,
    )
