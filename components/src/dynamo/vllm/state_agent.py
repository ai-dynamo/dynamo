# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in vLLM attachment configuration for the external KV state agent."""

import asyncio
import ipaddress
import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.config import VllmConfig

from dynamo.llm import KvStateAttachmentOwner, WorkerType
from dynamo.runtime import Endpoint

from .constants import DisaggregationMode
from .dp_topology import get_dp_range_for_worker
from .kv_hints import resolve_kv_transfer_hint_sources

logger = logging.getLogger(__name__)

_CACHE_OWNER_PATTERN = re.compile(
    r"^(?P<domain>[0-9a-f]{32}:[0-9a-f]{32})/[0-9a-f]{16}/[0-9a-f]{32}$"
)
_STATE_AGENT_CLOSE_TIMEOUT_SECS = 30.0


@dataclass(frozen=True)
class StateAgentSettings:
    raw_advertise_host: str
    cache_owner_ids: dict[int, str]


class StateAgentLifecycle:
    """One idempotent async cleanup point shared by signals and worker paths."""

    def __init__(self) -> None:
        self._owner: KvStateAttachmentOwner | None = None
        self._closed = False
        self._lock = asyncio.Lock()

    async def install(self, owner: KvStateAttachmentOwner) -> None:
        try:
            async with self._lock:
                if self._closed:
                    raise RuntimeError("KV state attachment lifecycle is closed")
                if self._owner is not None:
                    raise RuntimeError("KV state attachment owner is already installed")
                self._owner = owner
        except BaseException:
            # A started owner already holds discovery registrations. Its Rust
            # cleanup task survives cancellation of this Python await.
            await asyncio.shield(_close_attachment_owner(owner))
            raise

    async def close(self) -> None:
        async with self._lock:
            self._closed = True
            owner, self._owner = self._owner, None
        if owner is not None:
            await _close_attachment_owner(owner)


async def _close_attachment_owner(owner: KvStateAttachmentOwner) -> None:
    try:
        await asyncio.wait_for(owner.close(), timeout=_STATE_AGENT_CLOSE_TIMEOUT_SECS)
    except asyncio.TimeoutError:
        logger.warning(
            "KV state attachment cleanup timed out after %.0fs; continuing worker shutdown",
            _STATE_AGENT_CLOSE_TIMEOUT_SECS,
        )


def state_agent_settings(config: Any) -> StateAgentSettings | None:
    kv_transfer = getattr(config.engine_args, "kv_transfer_config", None)
    extra = getattr(kv_transfer, "kv_connector_extra_config", None) or {}
    if not isinstance(extra, dict):
        raise ValueError("kv_connector_extra_config must be an object")
    section = extra.get("dynamo_state_agent")
    if section is None:
        return None
    if not isinstance(section, dict):
        raise ValueError("dynamo_state_agent must be an object")

    host = section.get("raw_advertise_host")
    if not isinstance(host, str) or not host:
        raise ValueError("dynamo_state_agent.raw_advertise_host is required")
    _validate_advertise_host(host)

    raw_owners = section.get("cache_owner_ids")
    if not isinstance(raw_owners, dict) or not raw_owners:
        raise ValueError("dynamo_state_agent.cache_owner_ids must be a nonempty object")
    owners: dict[int, str] = {}
    for raw_rank, owner in raw_owners.items():
        try:
            rank = int(raw_rank)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "CacheOwner mapping keys must be global DP ranks"
            ) from error
        if str(rank) != str(raw_rank):
            raise ValueError(f"non-canonical global DP rank key: {raw_rank!r}")
        if not isinstance(owner, str) or _CACHE_OWNER_PATTERN.fullmatch(owner) is None:
            raise ValueError(f"invalid canonical CacheOwnerId for global rank {rank}")
        owners[rank] = owner
    if len(set(owners.values())) != len(owners):
        raise ValueError(
            "CacheOwnerId values must be unique across managed global ranks"
        )
    return StateAgentSettings(host, owners)


def validate_state_agent_worker(config: Any) -> None:
    if state_agent_settings(config) is None:
        return
    unsupported = None
    if config.headless:
        unsupported = "headless"
    elif config.realtime:
        unsupported = "realtime"
    elif config.embedding_worker:
        unsupported = "embedding"
    elif config.classify_worker:
        unsupported = "classify"
    elif config.disaggregation_mode == DisaggregationMode.ENCODE:
        unsupported = "multimodal encode"
    if unsupported is not None:
        raise ValueError(f"dynamo_state_agent is unsupported for {unsupported} workers")
    if not config.engine_args.enable_prefix_caching:
        raise ValueError("dynamo_state_agent requires prefix caching")
    events = config.engine_args.kv_events_config
    if events is None or not events.enable_kv_cache_events:
        raise ValueError("dynamo_state_agent requires vLLM KV event emission")


async def start_attachment_owner(
    config: Any,
    generate_endpoint: Endpoint,
    vllm_config: "VllmConfig",
    image_token_id: int | None,
) -> KvStateAttachmentOwner | None:
    settings = state_agent_settings(config)
    if settings is None:
        return None
    dp_start, dp_size = get_dp_range_for_worker(vllm_config)
    expected_ranks = set(range(dp_start, dp_start + dp_size))
    configured_ranks = set(settings.cache_owner_ids)
    if configured_ranks != expected_ranks:
        missing = sorted(expected_ranks - configured_ranks)
        extra = sorted(configured_ranks - expected_ranks)
        raise ValueError(
            "dynamo_state_agent CacheOwner mapping must exactly cover this worker's "
            f"global DP ranks (missing={missing}, extra={extra})"
        )
    if vllm_config.additional_config.get("consolidator_endpoints"):
        raise ValueError(
            "dynamo_state_agent does not support the KV event consolidator"
        )

    events = config.engine_args.kv_events_config
    block_size = int(vllm_config.cache_config.block_size)
    kv_transfer_hint_sources = resolve_kv_transfer_hint_sources(
        config.engine_args,
        _state_agent_worker_type(config),
        (dp_start, dp_size),
    )
    kv_state_endpoint = config.kv_state_endpoint or (
        f"{config.namespace}/{config.component}/{config.endpoint}"
    )
    descriptors = []
    for rank in sorted(expected_ranks):
        cache_owner_id = settings.cache_owner_ids[rank]
        kv_transfer_hint_source = (
            kv_transfer_hint_sources.get(rank)
            if kv_transfer_hint_sources is not None
            else None
        )
        descriptors.append(
            {
                "cache_owner_id": cache_owner_id,
                "global_dp_rank": rank,
                "kv_state_endpoint": kv_state_endpoint,
                "indexer_domain_id": cache_owner_id.split("/", 1)[0],
                "kv_block_size": block_size,
                "raw_zmq_endpoint": _advertised_raw_endpoint(
                    events.endpoint, rank, settings.raw_advertise_host
                ),
                "raw_topic": "",
                "image_token_id": image_token_id,
                "router_hint_source": (
                    {
                        "source_control_endpoint": kv_transfer_hint_source.source_control_endpoint,
                        "worker_type": kv_transfer_hint_source.worker_type,
                    }
                    if kv_transfer_hint_source is not None
                    else None
                ),
            }
        )
    attachment_owner = KvStateAttachmentOwner(
        generate_endpoint,
        generate_endpoint.connection_id(),
        descriptors,
    )
    await attachment_owner.start()
    return attachment_owner


def _state_agent_worker_type(config: Any) -> WorkerType:
    mode = getattr(config, "disaggregation_mode", DisaggregationMode.AGGREGATED)
    if mode == DisaggregationMode.PREFILL:
        return WorkerType.Prefill
    if mode == DisaggregationMode.DECODE:
        return WorkerType.Decode
    return WorkerType.Aggregated


def _advertised_raw_endpoint(base: str, rank: int, host: str) -> str:
    from vllm.distributed.kv_events import ZmqEventPublisher

    endpoint = ZmqEventPublisher.offset_endpoint_port(base, data_parallel_rank=rank)
    if not endpoint.startswith("tcp://") or ":" not in endpoint[6:]:
        raise ValueError(f"unsupported vLLM KV event endpoint: {endpoint!r}")
    port = endpoint.rsplit(":", 1)[1]
    if not port.isdigit():
        raise ValueError(f"vLLM KV event endpoint has no numeric port: {endpoint!r}")
    return f"tcp://{host}:{port}"


def _validate_advertise_host(host: str) -> None:
    if host in {"*", "localhost", "0.0.0.0", "::", "[::]"}:
        raise ValueError("raw_advertise_host must be a non-loopback connectable host")
    if "://" in host or "/" in host:
        raise ValueError("raw_advertise_host must not include a scheme, port, or path")
    if host.startswith("["):
        if not host.endswith("]"):
            raise ValueError("raw_advertise_host has mismatched IPv6 brackets")
        literal = host[1:-1]
        try:
            address = ipaddress.ip_address(literal)
        except ValueError as error:
            raise ValueError(
                "bracketed raw_advertise_host must be an IPv6 literal"
            ) from error
        if address.version != 6:
            raise ValueError("only IPv6 literals may use brackets")
    else:
        if "[" in host or "]" in host or ":" in host or any(c.isspace() for c in host):
            raise ValueError(
                "raw_advertise_host must be a hostname or bracketed IPv6 literal without a port"
            )
        literal = host
        try:
            address = ipaddress.ip_address(literal)
        except ValueError:
            return

    if address.is_loopback or address.is_unspecified:
        raise ValueError("raw_advertise_host must be a non-loopback connectable host")
