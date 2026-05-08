# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Compatibility shim for SGLang internal APIs.

SGLang is pre-1.0 and routinely moves, renames, or introduces APIs between
releases. This module is the single place where we handle those differences
so the rest of the component can import from here without version-specific
try/except blocks.

Policy: support current SGLang release + 1 version back (N and N-1). Each
fallback branch must document which version it covers and when it can be
removed. When the old version falls outside the support window, delete the
fallback and any associated polyfills.
"""

import inspect
import ipaddress
import logging
import socket
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Top-level sglang exports: Engine, ServerArgs
#
# Some SGLang dev builds (including 0.5.x snapshots) do not re-export these
# from sglang/__init__.py, while Dynamo historically uses `import sglang as sgl`
# followed by `sgl.Engine(...)` throughout this backend.
# ---------------------------------------------------------------------------
def ensure_sglang_top_level_exports() -> None:
    """Restore top-level SGLang exports omitted by some install flavors."""
    import sglang as sgl

    if not hasattr(sgl, "Engine"):
        from sglang.srt.entrypoints.engine import Engine

        sgl.Engine = Engine

    if not hasattr(sgl, "ServerArgs"):
        from sglang.srt.server_args import ServerArgs

        sgl.ServerArgs = ServerArgs


ensure_sglang_top_level_exports()


@lru_cache(maxsize=32)
def _get_async_generate_supported_kwarg_names(
    async_generate: Any,
) -> frozenset[str] | None:
    """Return supported async_generate keyword names, or None for **kwargs."""
    try:
        signature = inspect.signature(async_generate)
    except (TypeError, ValueError):
        logger.debug(
            "Could not inspect SGLang Engine.async_generate signature; "
            "dropping optional compatibility kwargs"
        )
        return frozenset()

    names: set[str] = set()
    for name, param in signature.parameters.items():
        if param.kind == inspect.Parameter.VAR_KEYWORD:
            return None
        if param.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(name)

    return frozenset(names)


def filter_supported_async_generate_kwargs(
    engine: Any, kwargs: dict[str, Any]
) -> dict[str, Any]:
    """Return only async_generate kwargs accepted by this SGLang engine.

    SGLang occasionally adds optional Engine.async_generate kwargs before every
    supported install flavor has them. Keep the compatibility boundary narrow:
    callers decide which kwargs are optional, and this helper only drops those
    optional kwargs when the installed engine cannot accept them.
    """
    async_generate = engine.async_generate
    signature_source = getattr(async_generate, "__func__", async_generate)

    try:
        supported_kwarg_names = _get_async_generate_supported_kwarg_names(
            signature_source
        )
    except TypeError:
        supported_kwarg_names = _get_async_generate_supported_kwarg_names.__wrapped__(
            signature_source
        )

    if supported_kwarg_names is None:
        return kwargs

    return {key: value for key, value in kwargs.items() if key in supported_kwarg_names}


# ---------------------------------------------------------------------------
# Network utilities: NetworkAddress, get_local_ip_auto, get_zmq_socket
#
# 0.5.10+: sglang.srt.utils.network (canonical)
# 0.5.9:   sglang.srt.utils (get_local_ip_auto, get_zmq_socket only;
#           NetworkAddress did not exist)
# ---------------------------------------------------------------------------
try:
    from sglang.srt.utils.network import (  # noqa: F401
        NetworkAddress,
        get_local_ip_auto,
        get_zmq_socket,
    )
except ImportError:
    # Fallback for sglang 0.5.9. Remove when min supported version is 0.5.10+
    from sglang.srt.utils import (  # type: ignore[no-redef]  # noqa: F401
        get_local_ip_auto,
        get_zmq_socket,
    )

    logger.info(
        "sglang.srt.utils.network not found (sglang 0.5.9); "
        "using compatibility shim for NetworkAddress"
    )

    class NetworkAddress:  # type: ignore[no-redef]
        """Minimal polyfill for sglang.srt.utils.network.NetworkAddress."""

        def __init__(self, host: str, port: int) -> None:
            self.host = host
            self.port = port

        @property
        def is_ipv6(self) -> bool:
            try:
                ipaddress.IPv6Address(self.host)
                return True
            except ValueError:
                return False

        @classmethod
        def parse(cls, addr: str) -> "NetworkAddress":
            """Parse 'host:port', '[IPv6]:port', or bare host."""
            addr = addr.strip()
            if addr.startswith("["):
                end = addr.find("]")
                host = addr[1:end] if end != -1 else addr.strip("[]")
                rest = addr[end + 1 :] if end != -1 else ""
                if rest.startswith(":") and rest[1:].isdigit():
                    return cls(host, int(rest[1:]))
                return cls(host, 0)
            if addr.count(":") == 1:
                host_part, port_part = addr.rsplit(":", 1)
                if port_part.isdigit():
                    return cls(host_part, int(port_part))
            return cls(addr, 0)

        def resolved(self) -> "NetworkAddress":
            """DNS-resolve the host, preserving port."""
            try:
                infos = socket.getaddrinfo(
                    self.host, None, family=socket.AF_UNSPEC, type=socket.SOCK_STREAM
                )
                resolved_ip = infos[0][4][0]
                return NetworkAddress(resolved_ip, self.port)
            except socket.gaierror:
                return self

        def to_host_port_str(self) -> str:
            """Return '[IPv6]:port' or 'host:port'."""
            if self.is_ipv6:
                return f"[{self.host}]:{self.port}"
            return f"{self.host}:{self.port}"

        def to_tcp(self) -> str:
            """Return 'tcp://[IPv6]:port' or 'tcp://host:port'."""
            if self.is_ipv6:
                return f"tcp://[{self.host}]:{self.port}"
            return f"tcp://{self.host}:{self.port}"


# ---------------------------------------------------------------------------
# MMEncoder._encode() adapter
#
# 0.5.10+: _encode(mm_items, modality) -> (grid_dim, embedding, aux_data)
# 0.5.9:   _encode(mm_items)           -> (grid_dim, embedding)
#
# Imports are deferred to avoid pulling sgl_kernel (CUDA-only) at module
# level, which breaks test collection on arm64 CPU-only CI nodes.
# ---------------------------------------------------------------------------


async def mm_encode(encoder: Any, mm_items: Any, modality: Any) -> tuple:
    """Version-safe wrapper around MMEncoder._encode().

    Always returns (grid_dim, embedding, aux_data). On sglang 0.5.9
    _encode takes no modality arg and returns a 2-tuple; on 0.5.10+ it
    takes modality and returns a 3-tuple. We try the new signature first
    and fall back to the old one.
    """
    try:
        result = await encoder._encode(mm_items, modality)
    except TypeError:
        # sglang 0.5.9: _encode(mm_items) -> (grid_dim, embedding)
        result = await encoder._encode(mm_items)

    if len(result) == 2:
        return (*result, None)
    return result


def get_scheduler_info(engine: Any) -> dict:
    """Return the scheduler-info dict for rank-0 of an ``sgl.Engine``.

    SGLang exposes per-rank scheduler stats (``max_total_num_tokens``,
    ``max_req_input_len``, ...) on the ``Engine`` via ``_scheduler_init_result``.
    We return the rank-0 dict, or ``{}`` if it is not reachable on this build.

    Covers:
      - sglang 0.5.10+: ``engine._scheduler_init_result.scheduler_infos[0]``
        (canonical; also what ``Engine.get_server_info`` reads internally).
      - Older probed attributes (``engine.scheduler_info``,
        ``engine.tokenizer_manager.scheduler_info``) as a best-effort fallback
        for forks/experimental branches that surfaced the dict directly.
    """
    result = getattr(engine, "_scheduler_init_result", None)
    if result is not None:
        infos = getattr(result, "scheduler_infos", None)
        if infos:
            return infos[0]

    direct = getattr(engine, "scheduler_info", None)
    if direct:
        return direct

    tm = getattr(engine, "tokenizer_manager", None)
    tm_info = getattr(tm, "scheduler_info", None) if tm is not None else None
    if tm_info:
        return tm_info

    return {}


def enable_disjoint_streaming_output(server_args: Any) -> None:
    """
    Enable SGLang's disjoint streaming output across ServerArgs field renames.

    Covers sglang <= 0.5.x (`stream_output`) and newer releases
    (`incremental_streaming_output`).
    """
    fields = getattr(type(server_args), "__dataclass_fields__", None)
    if isinstance(fields, dict):
        if "incremental_streaming_output" in fields:
            server_args.incremental_streaming_output = True
            return
        if "stream_output" in fields:
            server_args.stream_output = True
            return
        raise AttributeError(
            "SGLang ServerArgs has neither 'incremental_streaming_output' nor "
            "'stream_output'"
        )

    if hasattr(server_args, "incremental_streaming_output"):
        server_args.incremental_streaming_output = True
        return
    if hasattr(server_args, "stream_output"):
        server_args.stream_output = True
        return

    logger.debug(
        "Skipping streaming output compatibility for non-ServerArgs object: %s",
        type(server_args).__name__,
    )


__all__ = [
    "NetworkAddress",
    "enable_disjoint_streaming_output",
    "ensure_sglang_top_level_exports",
    "filter_supported_async_generate_kwargs",
    "get_local_ip_auto",
    "get_scheduler_info",
    "get_zmq_socket",
    "mm_encode",
]
