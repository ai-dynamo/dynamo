# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for router-orchestrated GMS KV placement metadata.

The router stays transport-agnostic. It learns source placement descriptors from
the existing KV event/indexer plane and stamps optional ``gms_placement`` onto
the request for the destination worker. Worker-local GMS daemon sockets remain
private to the worker process; the router does not open a GMS side channel.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

_LOG = logging.getLogger(__name__)


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() not in ("", "0", "false", "no", "off")


def resolve_vllm_gms_daemon_socket(vllm_config: Any) -> str | None:
    """Return the vLLM GMS KV daemon socket when the GMS connector is active."""

    kv_cfg = getattr(vllm_config, "kv_transfer_config", None)
    if kv_cfg is None:
        return None

    extra = getattr(kv_cfg, "kv_connector_extra_config", None) or {}
    socket_path = extra.get("gms_daemon_socket")
    if socket_path:
        return str(socket_path)

    connector = str(getattr(kv_cfg, "kv_connector", ""))
    module_path = str(getattr(kv_cfg, "kv_connector_module_path", ""))
    if "GMSKVCacheConnector" not in connector and "gds_connector" not in module_path:
        return None

    try:
        from gpu_memory_service.common.utils import get_socket_path

        return str(get_socket_path(0, "kv_cache"))
    except Exception:  # noqa: BLE001
        return None


def resolve_env_gms_daemon_socket(
    env_name: str,
    default_when_cross_node: str | None = None,
) -> str | None:
    socket_path = os.environ.get(env_name)
    if socket_path:
        return socket_path
    if default_when_cross_node and _truthy(os.environ.get("GMS_KVR_CROSS_NODE")):
        return default_when_cross_node
    return None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_gms_descriptor(
    descriptor: dict[str, Any],
    *,
    tier: str,
    bytes_size: int,
) -> dict[str, Any] | None:
    remote_ptr = _optional_int(descriptor.get("remote_ptr", descriptor.get("ptr")))
    size = _optional_int(descriptor.get("size", bytes_size))
    if remote_ptr is None or size is None or size <= 0:
        return None

    descriptor_tier = str(descriptor.get("tier") or tier or "external")
    ranges_raw = descriptor.get("ranges")
    if not isinstance(ranges_raw, list) or not ranges_raw:
        ranges_raw = [descriptor]

    ranges: list[dict[str, Any]] = []
    for item in ranges_raw:
        if not isinstance(item, dict):
            continue
        range_ptr = _optional_int(item.get("remote_ptr", item.get("ptr", remote_ptr)))
        range_size = _optional_int(item.get("size", size))
        if range_ptr is None or range_size is None or range_size <= 0:
            continue
        region: dict[str, Any] = {
            "remote_ptr": range_ptr,
            "size": range_size,
            "tier": str(item.get("tier") or descriptor_tier),
        }
        layer = _optional_int(item.get("layer"))
        if layer is not None:
            region["layer"] = layer
        offset = _optional_int(item.get("offset"))
        if offset is not None:
            region["offset"] = offset
        ranges.append(region)

    if not ranges:
        return None

    normalized: dict[str, Any] = {
        "remote_ptr": remote_ptr,
        "size": size,
        "tier": descriptor_tier,
        "ranges": ranges,
        "sealed": bool(descriptor.get("sealed", True)),
    }
    generation = _optional_int(descriptor.get("generation"))
    if generation is not None:
        normalized["generation"] = generation
    return normalized


class DynamoGmsPlacementPublisher:
    """Publish GMS placements through Dynamo's existing KV event plane."""

    def __init__(
        self,
        kv_publisher: Any,
        *,
        logger: logging.Logger | None = None,
        dp_rank: int | None = None,
    ) -> None:
        self._kv_publisher = kv_publisher
        self._logger = logger or _LOG
        self._dp_rank = dp_rank
        self._closed = False

    def publish_stored(
        self,
        content_hash: bytes,
        tier: str,
        bytes_size: int,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._closed:
            return
        metadata = metadata or {}
        descriptor = metadata.get("gms_descriptor") or metadata.get("descriptor")
        if not isinstance(descriptor, dict):
            self._logger.debug("Skipping GMS placement without descriptor")
            return
        source_name = str(metadata.get("source_nixl_agent_name") or "")
        source_metadata = str(
            metadata.get("source_nixl_agent_metadata_hex")
            or metadata.get("source_agent_metadata_hex")
            or ""
        )
        if not source_name or not source_metadata:
            self._logger.debug("Skipping GMS placement without NIXL source metadata")
            return

        normalized = _normalize_gms_descriptor(
            descriptor,
            tier=tier,
            bytes_size=bytes_size,
        )
        if normalized is None:
            self._logger.debug("Skipping malformed GMS placement descriptor")
            return

        kwargs: dict[str, Any] = {
            "source_nixl_ip": metadata.get("source_nixl_ip"),
            "source_nixl_listen_port": _optional_int(
                metadata.get("source_nixl_listen_port"),
            ),
        }
        if self._dp_rank is not None:
            kwargs["dp_rank"] = self._dp_rank

        self._kv_publisher.publish_gms_placement_stored(
            source_name,
            source_metadata,
            [{"content_hash_hex": content_hash.hex(), "descriptor": normalized}],
            **kwargs,
        )

    def publish_removed(self, content_hash: bytes, tier: str) -> None:
        if self._closed:
            return
        kwargs: dict[str, Any] = {}
        if self._dp_rank is not None:
            kwargs["dp_rank"] = self._dp_rank
        self._kv_publisher.publish_gms_placement_removed(
            [content_hash.hex()],
            **kwargs,
        )

    def close(self) -> None:
        self._closed = True


def _fetch_gms_placement_sync(
    *,
    placement: dict[str, Any],
    local_daemon_socket: str,
    timeout_s: float,
    batch_size: int,
) -> dict[str, int] | None:
    source_nixl_name = str(placement.get("source_nixl_agent_name") or "")
    if not source_nixl_name:
        return None

    from gms_kv_ring.daemon.client import DaemonClient

    metadata_hex = str(
        placement.get("source_nixl_agent_metadata_hex")
        or placement.get("source_agent_metadata_hex")
        or ""
    )
    hashes_hex = placement.get("hashes") or []
    descriptors = placement.get("descriptors") or []
    if not metadata_hex or not isinstance(hashes_hex, list):
        return None
    if not isinstance(descriptors, list) or len(hashes_hex) != len(descriptors):
        return None

    hashes: list[bytes] = []
    for value in hashes_hex:
        if not isinstance(value, str):
            return None
        try:
            hashes.append(bytes.fromhex(value))
        except ValueError:
            return None
    if not hashes or all(descriptor is None for descriptor in descriptors):
        return None

    with DaemonClient(
        local_daemon_socket,
        connect_timeout=min(5.0, max(0.5, timeout_s)),
        op_timeout=max(timeout_s + 5.0, timeout_s),
    ) as client:
        result = client.read_bootstrap_into_staging(
            source_nixl_name=source_nixl_name,
            source_agent_metadata_hex=metadata_hex,
            hashes=hashes,
            descriptors=descriptors,
            timeout_s=timeout_s,
            batch_size=batch_size,
        )
    return result


async def maybe_fetch_gms_placement(
    request: dict[str, Any],
    local_daemon_socket: str | None,
    *,
    logger: logging.Logger | None = None,
    request_id: str | None = None,
) -> dict[str, int] | None:
    """Fetch router-selected remote KV into the local daemon staging tier.

    Placements use in-band source NIXL metadata and descriptors.
    """

    placement = request.get("gms_placement")
    if not isinstance(placement, dict) or not local_daemon_socket:
        return None

    log = logger or _LOG
    try:
        timeout_s = float(os.environ.get("DYNAMO_GMS_ROUTER_FETCH_TIMEOUT_S", "30.0"))
    except ValueError:
        timeout_s = 30.0
    try:
        batch_size = int(os.environ.get("DYNAMO_GMS_ROUTER_FETCH_BATCH_SIZE", "0"))
    except ValueError:
        batch_size = 0

    try:
        result = await asyncio.to_thread(
            _fetch_gms_placement_sync,
            placement=placement,
            local_daemon_socket=local_daemon_socket,
            timeout_s=timeout_s,
            batch_size=batch_size,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "GMS router placement fetch failed for request %s: %s",
            request_id or "",
            exc,
            exc_info=True,
        )
        return None

    if result and result.get("failed", 0):
        log.warning(
            "GMS router placement fetch had failures for request %s: %s",
            request_id or "",
            result,
        )
    elif result:
        log.debug(
            "GMS router placement fetch completed for request %s: %s",
            request_id or "",
            result,
        )
    return result
