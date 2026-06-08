# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Best-effort KV headroom reclamation for planned shadow transitions."""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from dataclasses import dataclass

from gpu_memory_service.integrations.common.kv_lease_client import (
    SharedMemoryKVLeaseClient,
    kv_leases_enabled,
    read_kv_lease_namespace_total_blocks,
    read_kv_lease_reservation,
)

logger = logging.getLogger(__name__)


@dataclass
class KVTransitionReclaimWatcher:
    """Small polling handle returned by start_kv_transition_reclaim_watcher."""

    stop_event: threading.Event
    thread: threading.Thread

    def stop(self, timeout: float = 1.0) -> None:
        self.stop_event.set()
        self.thread.join(timeout=timeout)


def start_kv_transition_reclaim_watcher(
    engine: str,
    reclaim_cached_kv: Callable[[int], int],
    *,
    device: int | None = None,
    namespace_suffix: str = "kv",
    poll_s: float | None = None,
) -> KVTransitionReclaimWatcher | None:
    """Start a low-rate watcher for reservation-driven transition reclaim.

    The reservation file is the request plane for this local action: when it
    reserves blocks for a different owner, this process tries to free enough
    already-mirrored cached KV to satisfy the reserve without stopping traffic.
    """

    if not kv_leases_enabled(engine):
        return None
    engine_upper = engine.upper().replace("-", "_")
    enabled = os.environ.get(
        f"GMS_{engine_upper}_TRANSITION_RECLAIM",
        os.environ.get("GMS_KV_TRANSITION_RECLAIM", "1"),
    )
    if str(enabled).strip().lower() in ("", "0", "false", "no", "off"):
        return None
    if device is None:
        try:
            device = int(
                os.environ.get(
                    f"GMS_{engine_upper}_KV_LEASE_DEVICE",
                    os.environ.get("GMS_KV_LEASE_DEVICE", "0"),
                )
            )
        except ValueError:
            device = 0
    if poll_s is None:
        try:
            poll_s = float(
                os.environ.get(
                    f"GMS_{engine_upper}_TRANSITION_RECLAIM_POLL_S",
                    os.environ.get("GMS_KV_TRANSITION_RECLAIM_POLL_S", "0.25"),
                )
            )
        except ValueError:
            poll_s = 0.25
    poll_s = max(0.05, float(poll_s))
    owner_id = os.environ.get(
        f"GMS_{engine_upper}_KV_LEASE_OWNER_ID",
        os.environ.get("GMS_KV_LEASE_OWNER_ID", f"{engine}-{os.getpid()}-{device}"),
    )
    stop_event = threading.Event()

    def _loop() -> None:
        client: SharedMemoryKVLeaseClient | None = None
        total_blocks: int | None = None
        while not stop_event.wait(poll_s):
            try:
                _namespace, reservation = read_kv_lease_reservation(
                    engine, int(device), namespace_suffix=namespace_suffix
                )
                if reservation.reserved_blocks <= 0:
                    continue
                if reservation.reserved_for_owner is not None and (
                    str(reservation.reserved_for_owner) == str(owner_id)
                ):
                    continue
                if client is None:
                    _ns, total_blocks = read_kv_lease_namespace_total_blocks(
                        engine, int(device), namespace_suffix=namespace_suffix
                    )
                    if not total_blocks:
                        deficit = int(reservation.reserved_blocks)
                    else:
                        client = SharedMemoryKVLeaseClient.from_env(
                            engine,
                            int(device),
                            total_blocks=int(total_blocks),
                            owner_id=f"{owner_id}:reclaim-watcher",
                            namespace_suffix=namespace_suffix,
                        )
                        deficit = max(
                            0,
                            int(reservation.reserved_blocks)
                            - int(client.raw_free_count()),
                        )
                else:
                    deficit = max(
                        0,
                        int(reservation.reserved_blocks) - int(client.raw_free_count()),
                    )
                if deficit <= 0:
                    continue
                reclaimed = int(reclaim_cached_kv(deficit) or 0)
                if reclaimed > 0:
                    logger.info(
                        "[GMS transition] %s reclaimed %d cached KV blocks "
                        "toward shadow reserve deficit=%d owner=%s reserved_for=%s",
                        engine,
                        reclaimed,
                        deficit,
                        owner_id,
                        reservation.reserved_for_owner,
                    )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "[GMS transition] reclaim watcher iteration failed",
                    exc_info=True,
                )
                if client is not None:
                    try:
                        client.close()
                    except Exception:  # noqa: BLE001
                        pass
                    client = None
                    total_blocks = None
        if client is not None:
            try:
                client.close()
            except Exception:  # noqa: BLE001
                pass

    thread = threading.Thread(
        target=_loop,
        name=f"gms-{engine}-transition-reclaim",
        daemon=True,
    )
    thread.start()
    return KVTransitionReclaimWatcher(stop_event=stop_event, thread=thread)
