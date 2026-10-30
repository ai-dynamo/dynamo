# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Attach actual, process-local KVCR inventory to the KV state agent."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Mapping
from urllib.parse import urlsplit

from dynamo.common.constants import DisaggregationMode
from dynamo.common.kv_state_agent import (
    CACHE_OWNER_PATTERN,
    bounded_close,
    validate_advertise_host,
    worker_raw_topic,
)
from dynamo.llm import KvStateAttachmentOwner
from dynamo.sglang._compat import (
    filter_supported_async_generate_kwargs,
    resolved_server_args,
)
from dynamo.sglang.capacity import kv_event_block_size, model_card_dp_rank_bounds
from dynamo.sglang.kv_hints import (
    is_supported_kvcr_linker_topology,
    kvcr_hint_endpoints,
)

logger = logging.getLogger(__name__)


def _hint_sources(engine, wanted) -> dict[int, Mapping[str, object]]:
    """Read capabilities from scheduler metadata, never backend configuration."""
    initialization = getattr(engine, "_scheduler_init_result", None)
    if initialization is None:
        return {}
    sources: dict[int, Mapping[str, object]] = {}
    for info in initialization.scheduler_infos:
        hints = info.get("kv_hint_sources", [])
        if not isinstance(hints, list):
            raise TypeError("SGLang KV hint sources must be a list")
        for source in hints:
            if not isinstance(source, Mapping):
                raise TypeError("SGLang KV hint source must be an object")
            capabilities = source.get("capabilities", [])
            if not isinstance(capabilities, list):
                raise TypeError("SGLang KV hint capabilities must be a list")
            if not wanted(source, capabilities):
                continue
            rank = source.get("dp_rank")
            if type(rank) is not int or rank < 0:
                raise ValueError("KVCR inventory source requires a nonnegative DP rank")
            if rank in sources and sources[rank] != source:
                raise ValueError("Conflicting KVCR inventory sources for one DP rank")
            sources[rank] = source
    return sources


def inventory_sources(engine) -> dict[int, Mapping[str, object]]:
    return _hint_sources(engine, lambda _, caps: "kvcr_inventory" in caps)


def guard_sources(engine) -> dict[int, Mapping[str, object]]:
    return _hint_sources(
        engine,
        lambda source, caps: (
            "router_hint" in caps and source.get("inventory_lifetime") == "guard"
        ),
    )


def _guard_descriptors(engine, config, sources) -> list[dict]:
    """One descriptor per Guard-backed hint source; the Guard owns the cache."""
    owner = getattr(config.dynamo_args, "kvcr_state_agent_owner", None)
    if not isinstance(owner, str) or CACHE_OWNER_PATTERN.fullmatch(owner) is None:
        raise ValueError(
            "KVCR Guard mode requires a canonical --kvcr-state-agent-owner"
        )
    server_args = resolved_server_args(config.server_args)
    if (
        not is_supported_kvcr_linker_topology(server_args)
        or config.serving_mode != DisaggregationMode.AGGREGATED
        or not config.dynamo_args.use_kv_events
        or not filter_supported_async_generate_kwargs(engine, {"kv_hints": None})
    ):
        raise ValueError(
            "KVCR Guard mode requires an aggregated TP1 hint-forwarding linker "
            "worker with KV event emission"
        )
    bounds = model_card_dp_rank_bounds(server_args)
    if set(sources) != set(range(*bounds)):
        raise ValueError("KVCR Guard sources must cover every managed DP rank")
    infos = engine._scheduler_init_result.scheduler_infos
    endpoints = kvcr_hint_endpoints(infos, bounds, require="guard")
    events = server_args.kv_events_config
    events = json.loads(events) if isinstance(events, str) else events
    if not isinstance(events, dict) or events.get("publisher") != "zmq":
        raise ValueError("KVCR Guard mode requires a ZMQ event publisher")
    if events.get("topic", "") != "":
        raise ValueError("KVCR Guard mode requires an empty event topic")
    event_endpoint = urlsplit(events.get("endpoint", ""))
    if (
        event_endpoint.scheme != "tcp"
        or not event_endpoint.hostname
        or not event_endpoint.port
    ):
        raise ValueError("KVCR Guard mode requires an explicit TCP event port")
    descriptors = []
    for rank in sorted(sources):
        host = urlsplit(endpoints[str(rank)]).netloc.rsplit(":", 1)[0]
        descriptors.append(
            {
                "cache_owner_id": owner,
                "global_dp_rank": rank,
                "kv_state_endpoint": config.dynamo_args.kv_state_endpoint
                or f"{config.dynamo_args.namespace}/{config.dynamo_args.component}/"
                f"{config.dynamo_args.endpoint}",
                "indexer_domain_id": owner.split("/", 1)[0],
                "kv_block_size": kv_event_block_size(server_args),
                "ingress_protocol": "vllm_residency_v1",
                "raw_zmq_endpoint": f"tcp://{host}:{event_endpoint.port}",
                "raw_topic": "",
                "image_token_id": None,
                "video_token_id": None,
                "router_hint_source": {
                    "source_control_endpoint": endpoints[str(rank)],
                    "worker_type": "aggregated",
                },
            }
        )
    return descriptors


def state_agent_descriptors(engine, config) -> list[dict]:
    sources = inventory_sources(engine)
    guards = guard_sources(engine)
    if sources and guards:
        raise ValueError("KVCR inventory and Guard sources cannot be mixed")
    if guards:
        return _guard_descriptors(engine, config, guards)
    if not sources:
        return []
    server_args = resolved_server_args(config.server_args)
    linker = is_supported_kvcr_linker_topology(server_args)
    if (
        not (getattr(server_args, "enable_hierarchical_cache", False) or linker)
        or server_args.nnodes != 1
        or server_args.pp_size != 1
        or server_args.tokenizer_worker_num != 1
        or not config.dynamo_args.use_kv_events
    ):
        raise ValueError(
            "KVCR inventory requires a single-host HiCache or supported linker worker, one tokenizer "
            "worker, PP=1 and KV event emission"
        )
    if linker and not filter_supported_async_generate_kwargs(
        engine, {"kv_hints": None}
    ):
        raise ValueError("KVCR linker inventory requires a hint-forwarding worker")
    roles = {
        DisaggregationMode.AGGREGATED: "aggregated",
        DisaggregationMode.PREFILL: "prefill",
        DisaggregationMode.DECODE: "decode",
    }
    role = roles.get(config.serving_mode)
    if role is None:
        raise ValueError("KVCR inventory requires a hint-forwarding LLM worker")
    bounds = model_card_dp_rank_bounds(server_args)
    if set(sources) != set(range(*bounds)):
        raise ValueError("KVCR inventory must cover every managed DP rank")
    if linker:
        for source in sources.values():
            if source.get("inventory_lifetime") != "process_local":
                raise ValueError("KVCR linker must report process-local inventory")
            if (
                source.get("reset_clears_owner_inventory") is not True
                or source.get("owner_incarnation_scoped") is not True
            ):
                raise ValueError(
                    "KVCR linker inventory requires reset and incarnation fencing"
                )
    infos = engine._scheduler_init_result.scheduler_infos
    endpoints = kvcr_hint_endpoints(
        infos, bounds, require="inventory" if linker else None
    )
    if set(endpoints) != {str(rank) for rank in sources}:
        raise ValueError(
            "KVCR inventory requires a router_hint source for every managed DP rank"
        )
    raw_sources: dict[int, Mapping[str, object]] = {}
    for info in infos:
        for source in info.get("kv_event_sources", []):
            rank = source.get("dp_rank")
            if type(rank) is not int or rank not in sources:
                raise ValueError("KVCR raw event source has an invalid DP rank")
            if rank in raw_sources and raw_sources[rank] != source:
                raise ValueError("Conflicting raw KV event sources for one DP rank")
            raw_sources[rank] = source
    if set(raw_sources) != set(sources):
        raise ValueError("KVCR inventory requires a reported raw event source per rank")
    descriptors = []
    owners = set()
    for rank, source in sorted(sources.items()):
        owner = source.get("cache_owner_id")
        if not isinstance(owner, str) or CACHE_OWNER_PATTERN.fullmatch(owner) is None:
            raise ValueError("KVCR inventory source requires a canonical CacheOwnerId")
        if owner in owners:
            raise ValueError("KVCR inventory owners must be unique across DP ranks")
        owners.add(owner)
        raw = raw_sources[rank]
        if raw.get("format") != "dynamo":
            raise ValueError("KVCR inventory requires Dynamo-format KV events")
        endpoint = raw.get("endpoint")
        if not isinstance(endpoint, str):
            raise TypeError("KVCR inventory requires a reported TCP event endpoint")
        parsed = urlsplit(endpoint)
        if (
            parsed.scheme != "tcp"
            or not parsed.hostname
            or parsed.port is None
            or not 0 < parsed.port <= 65535
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path
            or parsed.query
            or parsed.fragment
            or any(char.isspace() for char in parsed.hostname)
            or parsed.hostname == "*"
        ):
            raise ValueError("Invalid reported KVCR raw event endpoint")
        host = parsed.hostname
        validate_advertise_host(f"[{host}]" if ":" in host else host)
        block_size = raw.get("block_size")
        if type(block_size) is not int or block_size <= 0:
            raise ValueError("KVCR raw event source requires a positive block size")
        if linker and block_size != kv_event_block_size(server_args):
            raise ValueError("KVCR linker event block size must match the model card")
        topic = raw.get("topic", "")
        if not isinstance(topic, str) or not topic.strip():
            raise ValueError(
                "KVCR raw event topic must identify this engine incarnation"
            )
        if linker and topic != worker_raw_topic([owner]):
            raise ValueError("KVCR linker event topic must identify its owner")
        descriptors.append(
            {
                "cache_owner_id": owner,
                "global_dp_rank": rank,
                "kv_state_endpoint": config.dynamo_args.kv_state_endpoint
                or f"{config.dynamo_args.namespace}/{config.dynamo_args.component}/"
                f"{config.dynamo_args.endpoint}",
                "indexer_domain_id": owner.split("/", 1)[0],
                "kv_block_size": block_size,
                "ingress_protocol": "sglang_hicache_v1",
                "raw_zmq_endpoint": endpoint,
                "raw_topic": topic,
                "image_token_id": None,
                "video_token_id": None,
                "router_hint_source": {
                    "source_control_endpoint": endpoints[str(rank)],
                    "worker_type": role,
                },
            }
        )
    return descriptors


class StateAgentLifecycle:
    def __init__(self):
        self._owner = None
        self._cleanup_task = None

    async def start(self, engine, config, endpoint):
        descriptors = state_agent_descriptors(engine, config)
        if not descriptors:
            return
        owner = KvStateAttachmentOwner(endpoint, endpoint.connection_id(), descriptors)
        startup = asyncio.ensure_future(owner.start())
        try:
            await asyncio.shield(startup)
        except asyncio.CancelledError:
            self._cleanup_task = asyncio.create_task(
                self._close_after_start(owner, startup)
            )
            raise
        # Registered intent is not an attachment: until the agent has subscribed,
        # events the engine publishes are lost. Keep the owner set so a timeout
        # or cancellation here is withdrawn by close(); the error fails the
        # worker closed before it can become routable.
        self._owner = owner
        await owner.await_attached()
        logger.info("Attached %d KVCR inventory owners", len(descriptors))

    async def _close_after_start(self, owner, startup):
        # Let the native binding initialize its OnceCell before withdrawal.
        await startup
        await owner.close()

    async def close(self):
        if self._cleanup_task is not None:
            await bounded_close(asyncio.shield(self._cleanup_task))
            return
        owner, self._owner = self._owner, None
        if owner is not None:
            await bounded_close(owner.close())
