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

import ipaddress
import logging
import re
import socket
from typing import Any

logger = logging.getLogger(__name__)

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


def _build_grouped_mm_token_regex(
    prefix_token: str, mm_token: str, suffix_token: str
) -> re.Pattern[str]:
    """
    Match one expanded multimodal block wrapped by start/end vision tokens.

    Qwen emits expanded multimodal prompts such as:
        <|vision_start|><|video_pad|><|video_pad|>...<|vision_end|>

    Grouping the repeated placeholder tokens into a single regex match keeps
    SGLang's legacy multimodal loader aligned with processor_output inputs.
    """

    return re.compile(
        f"{re.escape(prefix_token)}(?:{re.escape(mm_token)})+{re.escape(suffix_token)}"
    )


def _convert_token_id_to_token_str(tokenizer: Any, token_id: Any) -> str | None:
    """Best-effort token-id to token-string conversion."""
    if tokenizer is None or token_id is None:
        return None
    try:
        return tokenizer.convert_ids_to_tokens([token_id])[0]
    except Exception:
        return None


def _uses_default_single_token_regex(token_regex: re.Pattern[str] | None, token: str) -> bool:
    """Return True when the regex is the base default: one escaped token."""
    return token_regex is None or token_regex.pattern == re.escape(token)


def _maybe_upgrade_wrapped_video_token_regex(
    multimodal_tokens: Any, tokenizer: Any
) -> bool:
    """
    Extend grouped image-token matching to video when the processor exposes:

    - a wrapped image placeholder token, e.g.
      <|vision_start|><|image_pad|><|vision_end|>
    - a grouped image regex, e.g.
      <|vision_start|>(?:<|image_pad|>)+<|vision_end|>
    - only a default single-token video regex, e.g.
      <|video_pad|>

    This keeps Dynamo-side expanded placeholder tokens aligned with SGLang's
    legacy multimodal loader without hardcoding a specific model class.
    """

    if multimodal_tokens is None:
        return False

    image_token = getattr(multimodal_tokens, "image_token", None)
    video_token = getattr(multimodal_tokens, "video_token", None)
    image_token_regex = getattr(multimodal_tokens, "image_token_regex", None)
    video_token_regex = getattr(multimodal_tokens, "video_token_regex", None)

    if not isinstance(image_token, str) or not isinstance(video_token, str):
        return False
    if image_token_regex is None:
        return False
    if image_token_regex.pattern == re.escape(image_token):
        return False
    if not _uses_default_single_token_regex(video_token_regex, video_token):
        return False

    image_atomic_token = _convert_token_id_to_token_str(
        tokenizer, getattr(multimodal_tokens, "image_token_id", None)
    )
    video_atomic_token = _convert_token_id_to_token_str(
        tokenizer, getattr(multimodal_tokens, "video_token_id", None)
    )
    if not image_atomic_token or not video_atomic_token:
        return False
    if image_token == image_atomic_token:
        return False
    if video_token != video_atomic_token:
        return False
    if image_token.count(image_atomic_token) != 1:
        return False

    prefix_token, suffix_token = image_token.split(image_atomic_token, 1)
    if not prefix_token or not suffix_token:
        return False

    multimodal_tokens.video_token_regex = _build_grouped_mm_token_regex(
        prefix_token, video_token, suffix_token
    )
    multimodal_tokens.combined_regex = None
    multimodal_tokens.get_combined_regex()
    return True


def maybe_enable_grouped_video_token_regex(mm_processor: Any) -> bool:
    """
    Upgrade one multimodal processor instance when it uses wrapped grouped
    image placeholders but leaves video matching at the single-token default.

    This keeps Dynamo's expanded video placeholder tokens aligned with SGLang's
    multimodal loader without globally changing processor behavior.
    """
    if mm_processor is None:
        return False

    tokenizer = getattr(getattr(mm_processor, "_processor", None), "tokenizer", None)
    if tokenizer is None:
        tokenizer = getattr(mm_processor, "tokenizer", None)
    if tokenizer is None and hasattr(
        getattr(mm_processor, "_processor", None), "convert_ids_to_tokens"
    ):
        tokenizer = getattr(mm_processor, "_processor", None)

    upgraded = _maybe_upgrade_wrapped_video_token_regex(
        getattr(mm_processor, "mm_tokens", None), tokenizer
    )
    if upgraded:
        logger.info(
            "Enabled grouped video token regex compatibility for %s",
            type(mm_processor).__name__,
        )
    return upgraded


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
    "get_local_ip_auto",
    "get_zmq_socket",
    "maybe_enable_grouped_video_token_regex",
    "mm_encode",
]
