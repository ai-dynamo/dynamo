# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang capability publication for Dynamo KV hints.

Mirrors ``dynamo/vllm/kv_hints.py``: a worker advertises the ``router_hint``
capability, its topology role, and a per-DP-rank map of KV source control
endpoints, or the router treats it as hint-incapable and emits no hints.

SGLang has no vLLM-style secondary tiers. The equivalent settings live in the
KVCR backend's extra-config JSON (``--hicache-storage-backend-extra-config``),
whose ``control_advertise_host`` and ``control_port`` describe the ZMQ peer
control channel each scheduler rank binds. Two SGLang modes run KVCR:

* the direct external linker (``--enable-unified-cache-external-linker``
  ``--unified-cache-external-linker-backend kvcr``), and
* the HiCache storage backend (``--hicache-storage-backend kvcr``).

Both bind ``control_port + engine_global_attention_rank``, so one DP rank owns
a block of ``tp_size // dp_size`` ports and the router is handed the block's
first port for every DP rank this node runs.
"""

from __future__ import annotations

import ipaddress
import json
import logging
from typing import Any, Mapping, Optional

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)

logger = logging.getLogger(__name__)

KVCR_BACKEND_NAME = "kvcr"

# SGLang's disaggregation_mode in the router's worker_type vocabulary. The
# router only pairs a hint source with a target of the same role, so this must
# agree with the vLLM backend's mapping.
_WORKER_TYPE_BY_DISAGGREGATION_MODE = {
    None: "aggregated",
    "null": "aggregated",
    "prefill": "prefill",
    "decode": "decode",
}


def kvcr_mode(server_args: Any) -> Optional[str]:
    """``"linker"``, ``"hicache"``, or None when this worker does not run KVCR."""
    if getattr(server_args, "enable_unified_cache_external_linker", False) and (
        getattr(server_args, "unified_cache_external_linker_backend", None)
        == KVCR_BACKEND_NAME
    ):
        return "linker"
    if getattr(server_args, "hicache_storage_backend", None) == KVCR_BACKEND_NAME:
        return "hicache"
    return None


def parse_kvcr_extra_config(raw_extra_config: Any) -> dict[str, Any]:
    """The KVCR extra-config mapping from a dict, JSON text, or ``@file``."""
    if raw_extra_config is None:
        return {}
    if isinstance(raw_extra_config, Mapping):
        return dict(raw_extra_config)
    if not isinstance(raw_extra_config, str):
        raise ValueError("hicache_storage_backend_extra_config must be JSON text")
    text = raw_extra_config.strip()
    if not text:
        return {}
    if text.startswith("@"):
        with open(text[1:], "r", encoding="utf-8") as handle:
            text = handle.read()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("hicache_storage_backend_extra_config must be a JSON object")
    return parsed


def kv_hint_worker_type(server_args: Any) -> Optional[str]:
    mode = getattr(server_args, "disaggregation_mode", None)
    return _WORKER_TYPE_BY_DISAGGREGATION_MODE.get(mode)


def dp_port_stride(server_args: Any) -> int:
    """Control ports one attention-DP rank of this engine owns.

    Every attention rank's scheduler binds ``base + engine_global_rank``, and
    SGLang lays ranks out DP-major, so DP rank ``d`` starts at
    ``base + d * (tp_size // dp_size)``. Without attention DP the engine is one
    block. Must agree with the SGLang KVCR backends' port offset; a mismatch
    makes peers dial the wrong rank and silently fetch another shard.
    """
    dp_size = getattr(server_args, "dp_size", 1) or 1
    if not getattr(server_args, "enable_dp_attention", False) or dp_size <= 1:
        return 1
    tp_size = getattr(server_args, "tp_size", 1) or 1
    return max(tp_size // dp_size, 1)


def _advertise_host(host: object) -> Optional[str]:
    if not isinstance(host, str) or not host:
        return None
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host
    if address.is_unspecified:
        return None
    return f"[{address.compressed}]" if address.version == 6 else address.compressed


def _source_control_endpoint(
    extra_config: Mapping[str, Any], port_offset: int
) -> Optional[str]:
    port = extra_config.get("control_port")
    if isinstance(port, bool) or not isinstance(port, (int, str)):
        return None
    try:
        control_port = int(port) + port_offset
    except ValueError:
        return None
    if not 0 < control_port <= 65535:
        return None
    host = _advertise_host(extra_config.get("control_advertise_host"))
    if host is None:
        return None
    return f"tcp://{host}:{control_port}"


def source_control_endpoints(
    extra_config: Mapping[str, Any],
    dp_bounds: tuple[int, int],
    stride: int,
) -> Optional[dict[str, str]]:
    """Endpoint per global DP rank this node runs, or None if any is undialable.

    All-or-nothing: a partial map would let the router pick a rank no peer can
    reach. Keys are global DP ranks (the router's ``(worker_id, dp_rank)``);
    the port offset follows the node-local rank because every node numbers its
    ports from the same base.
    """
    dp_start, dp_end = dp_bounds
    endpoints: dict[str, str] = {}
    for local_dp_rank in range(dp_end - dp_start):
        endpoint = _source_control_endpoint(extra_config, local_dp_rank * stride)
        if endpoint is None:
            return None
        endpoints[str(dp_start + local_dp_rank)] = endpoint
    return endpoints


def publish_kv_hint_capabilities(
    runtime_config: Any,
    server_args: Any,
    extra_config: Mapping[str, Any],
    dp_bounds: tuple[int, int],
) -> bool:
    """Advertise this worker as a KV hint source when KVCR remote hints are on.

    No-op unless a KVCR mode is selected and ``enable_remote_hint`` is set: a
    worker without a remote-capable KV source has nothing to serve. Raises when
    hints are enabled but the endpoints are not advertisable, since that would
    register a source no peer can dial. All three keys are published together;
    the router requires all of them. Returns whether anything was published.
    """
    mode = kvcr_mode(server_args)
    if mode is None or not extra_config.get("enable_remote_hint"):
        return False
    worker_type = kv_hint_worker_type(server_args)
    if worker_type is None:
        return False
    endpoints = source_control_endpoints(
        extra_config, dp_bounds, dp_port_stride(server_args)
    )
    if endpoints is None:
        raise ValueError(
            "KVCR router hints require advertisable source control endpoints for "
            "every managed DP rank: set control_advertise_host to a peer-reachable "
            "address and a control_port that keeps every rank within 1..65535 in "
            "--hicache-storage-backend-extra-config"
        )
    # set_engine_specific expects JSON text. Publish the capability flag last so
    # a partially observed registration is never capability-only.
    runtime_config.set_engine_specific(
        KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, json.dumps(endpoints)
    )
    runtime_config.set_engine_specific(
        KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, json.dumps(worker_type)
    )
    runtime_config.set_engine_specific(
        KV_HINT_TRANSFER_CAPABILITY_KEY, json.dumps(True)
    )
    logger.info(
        "Advertised KV hint capability (mode=%s worker_type=%s) with source control "
        "endpoints %s",
        mode,
        worker_type,
        endpoints,
    )
    return True
