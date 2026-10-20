# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KVCR transfer capability and in-process request-hint forwarding."""

from __future__ import annotations

import ipaddress
import json
from collections.abc import Mapping
from typing import Any

from dynamo.common.constants import (
    KV_HINT_TRANSFER_CAPABILITY_KEY,
    KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY,
    KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY,
)
from dynamo.sglang._compat import (
    filter_supported_async_generate_kwargs,
    resolved_server_args,
)


def kv_hint_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Dynamo's private kv_hint envelope maps to SGLang's kv_hints input.

    The router creates the envelope after worker selection; do not pull a
    client-supplied hint out of extra_args or nvext. Older engines can ignore
    this advisory input, but must not advertise transfer capability.
    """
    hint = request.get("kv_hint")
    if not isinstance(hint, Mapping):
        return {}
    return filter_supported_async_generate_kwargs(engine, {"kv_hints": dict(hint)})


def kvcr_mode(server_args: Any) -> str | None:
    """The direct linker or the HiCache storage backend; None without KVCR.

    SGLang rejects enabling both, and both read the KVCR options from
    --hicache-storage-backend-extra-config.
    """
    if (
        getattr(server_args, "enable_unified_cache_external_linker", False)
        and getattr(server_args, "unified_cache_external_linker_backend", None)
        == "kvcr"
    ):
        return "linker"
    if (
        getattr(server_args, "enable_hierarchical_cache", False)
        and getattr(server_args, "hicache_storage_backend", None) == "kvcr"
    ):
        return "hicache"
    return None


def kvcr_hint_endpoints(
    *,
    extra_config: Mapping[str, Any],
    dp_size: int,
    tp_size: int,
    attn_cp_size: int,
    pp_size: int,
    mode: str = "hicache",
) -> dict[str, str]:
    """One base endpoint per DP replica, matching SGLang's rank-port layout."""
    if mode not in ("hicache", "linker"):
        raise ValueError(f"Unknown KVCR mode: {mode}")
    # HiCache does not consume advertise_host: accepting it here would publish
    # an address different from the storage backend's actual bound endpoint.
    host = extra_config.get("control_advertise_host", "127.0.0.1")
    port = extra_config.get("control_port", 32100)
    if mode == "linker":
        # Linker revisions have different defaults (19500 vs OS-assigned 0)
        # and option names. Require explicit coordinates rather than guess.
        host = extra_config.get("advertise_host")
        alternate = extra_config.get("control_advertise_host")
        if host is not None and alternate is not None and host != alternate:
            raise ValueError("Conflicting KVCR linker advertise hosts")
        if host is None:
            host = alternate
        port = extra_config.get("control_port")
    if not isinstance(host, str) or not host:
        raise ValueError("KVCR requires an advertisable control host")
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
        if any(char.isspace() or char in "/@?#:[]" for char in host):
            raise ValueError("KVCR requires a host, not a URL or malformed address")
    if address is not None:
        if address.is_unspecified:
            raise ValueError("KVCR cannot advertise an unspecified control host")
        host = f"[{address.compressed}]" if address.version == 6 else address.compressed
    stride = tp_size * attn_cp_size * pp_size
    if (
        type(port) is not int
        or min(dp_size, tp_size, attn_cp_size, pp_size) < 1
        or not 0 < port <= port + dp_size * stride - 1 <= 65535
    ):
        raise ValueError("KVCR control-port range is invalid")
    return {
        str(rank): f"tcp://{host}:{port + rank * stride}" for rank in range(dp_size)
    }


def kvcr_hint_sources(
    server_args: Any, worker_type: Any
) -> tuple[str, dict[str, str]] | None:
    """The role and per-DP-rank source-control endpoints this worker advertises.

    None when the worker runs no supported single-host KVCR linker or HiCache
    path with remote hints enabled.
    """
    server_args = resolved_server_args(server_args)
    mode = kvcr_mode(server_args)
    if (
        mode is None
        or getattr(server_args, "nnodes", 1) != 1
        or getattr(server_args, "pp_size", 1) != 1
    ):
        return None
    extra = server_args.hicache_storage_backend_extra_config
    if isinstance(extra, str):
        if extra.startswith("@"):
            # The SGLang controller resolves file configs. Avoid advertising a
            # guessed endpoint until registration can read its resolved result.
            raise ValueError("KVCR hint publication requires inline JSON configuration")
        extra = json.loads(extra)
    if extra is None:
        extra = {}
    if not isinstance(extra, Mapping):
        raise ValueError("KVCR HiCache extra configuration must be an object")
    # Legacy linker revisions have no switch; newer ones can disable hints.
    if extra.get("enable_remote_hint", True) is not True:
        return None
    if (
        mode == "linker"
        and extra.get("advertise_host") is None
        and extra.get("enable_remote_hint", False) is not True
    ):
        # The newer control_advertise_host dialect defaults remote hints off.
        # The legacy advertise_host runtime has no switch and enables peers.
        return None
    if mode == "linker" and (
        "control_port" not in extra
        or (
            extra.get("advertise_host") is None
            and extra.get("control_advertise_host") is None
        )
    ):
        # Local-only/default linkers can still serve, but cannot advertise a
        # guessed source-control endpoint to another worker.
        return None
    # WorkerType is a PyO3 enum, not enum.Enum: its wire role is __str__,
    # and accessing .value aborts registration after the model has loaded.
    role = str(worker_type)
    if role == "agg":
        role = "aggregated"
    if role not in ("aggregated", "prefill", "decode"):
        return None
    # DP attention partitions TP into independent attention/cache groups.
    dp_size = server_args.attn_dp_size
    attn_cp_size = server_args.attn_cp_size
    tp_size = server_args.tp_size // (dp_size * attn_cp_size)
    endpoints = kvcr_hint_endpoints(
        extra_config=extra,
        dp_size=dp_size,
        tp_size=tp_size,
        attn_cp_size=attn_cp_size,
        pp_size=server_args.pp_size,
        mode=mode,
    )
    return role, endpoints


def publish_kvcr_hint_capabilities(
    runtime_config, engine, server_args, worker_type
) -> None:
    """Opt in only for the supported single-host KVCR linker or HiCache path."""
    if engine is None or not filter_supported_async_generate_kwargs(
        engine, {"kv_hints": None}
    ):
        return
    sources = kvcr_hint_sources(server_args, worker_type)
    if sources is None:
        return
    role, endpoints = sources
    for key, value in (
        (KV_HINT_TRANSFER_WORKER_TYPE_RUNTIME_KEY, role),
        (KV_HINT_TRANSFER_SOURCE_CONTROL_ENDPOINTS_RUNTIME_KEY, endpoints),
        (KV_HINT_TRANSFER_CAPABILITY_KEY, True),
    ):
        runtime_config.set_engine_specific(key, json.dumps(value))
