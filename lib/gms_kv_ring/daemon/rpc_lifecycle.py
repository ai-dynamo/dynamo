# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine lifecycle and local storage RPC handlers."""
from __future__ import annotations
from typing import TYPE_CHECKING
from gms_kv_ring.daemon.consumers import LayerDesc
from gms_kv_ring.daemon.rpc_types import Handler, Message, Response
if TYPE_CHECKING:
    from gms_kv_ring.daemon.server import Daemon


def handle_ping(daemon: "Daemon", msg: Message) -> Response:
    return {"ok": True}


def handle_attach_engine_pool(daemon: "Daemon", msg: Message) -> Response:
    layers = [
        LayerDesc(
            layer_idx=int(layer["layer_idx"]),
            va=int(layer["va"]),
            size=int(layer["size"]),
            stride=int(layer["stride"]),
        )
        for layer in msg["layers"]
    ]
    daemon.attach_engine_pool(str(msg["engine_id"]), layers)
    return {"ok": True}


def handle_detach_engine_pool(daemon: "Daemon", msg: Message) -> Response:
    ok = daemon.detach_engine_pool(str(msg["engine_id"]))
    return {"ok": True, "found": ok}


def handle_attach_evict_ring(daemon: "Daemon", msg: Message) -> Response:
    daemon.attach_evict_ring(
        str(msg["engine_id"]),
        str(msg["ring_path"]),
        counter_host_addr=int(msg.get("counter_host_addr", 0)),
        num_counters=int(msg.get("num_counters", 0)),
        counter_path=str(msg.get("counter_path", "")),
    )
    return {"ok": True}


def handle_attach_restore_ring(daemon: "Daemon", msg: Message) -> Response:
    daemon.attach_restore_ring(
        str(msg["engine_id"]),
        str(msg["ring_path"]),
        str(msg["counter_path"]),
        int(msg.get("num_counters", 512)),
        counter_host_addr=int(msg.get("counter_host_addr", 0)),
    )
    return {"ok": True}


def handle_demote_to_storage(daemon: "Daemon", msg: Message) -> Response:
    ok = daemon.demote_to_storage(
        str(msg["engine_id"]),
        int(msg["layer"]),
        int(msg["offset"]),
    )
    return {"ok": True, "demoted": ok}


def handle_promote_from_storage(daemon: "Daemon", msg: Message) -> Response:
    ok = daemon.promote_from_storage(
        str(msg["engine_id"]),
        int(msg["layer"]),
        int(msg["offset"]),
    )
    return {"ok": True, "promoted": ok}


def handle_demote_hbm_to_storage(daemon: "Daemon", msg: Message) -> Response:
    ok = daemon.demote_hbm_to_storage(
        str(msg["engine_id"]),
        int(msg["layer"]),
        int(msg["offset"]),
        int(msg["size"]),
        generation=int(msg.get("generation", 0)),
    )
    return {"ok": True, "demoted": ok}


def handle_promote_storage_to_hbm(daemon: "Daemon", msg: Message) -> Response:
    ok = daemon.promote_storage_to_hbm(
        str(msg["engine_id"]),
        int(msg["layer"]),
        int(msg["offset"]),
        int(msg["size"]),
        dest_offset=msg.get("dest_offset"),
        expected_generation=int(msg.get("expected_generation", 0)),
    )
    return {"ok": True, "promoted": ok}


def handle_capabilities(daemon: "Daemon", msg: Message) -> Response:
    return {"ok": True, "capabilities": daemon.capabilities()}


def handle_release_engine_storage(daemon: "Daemon", msg: Message) -> Response:
    n = daemon.release_engine_storage(str(msg["engine_id"]))
    return {"ok": True, "released": n}


def handle_prune_storage(daemon: "Daemon", msg: Message) -> Response:
    n = daemon.prune_storage(
        max_age_seconds=msg.get("max_age_seconds"),
        max_bytes=msg.get("max_bytes"),
        max_bytes_per_engine=msg.get("max_bytes_per_engine"),
    )
    return {"ok": True, "evicted": n}


def handle_storage_stats(daemon: "Daemon", msg: Message) -> Response:
    return {"ok": True, "stats": daemon.storage_stats()}

HANDLERS: dict[str, Handler] = {
    "attach_engine_pool": handle_attach_engine_pool,
    "attach_evict_ring": handle_attach_evict_ring,
    "attach_restore_ring": handle_attach_restore_ring,
    "capabilities": handle_capabilities,
    "demote_hbm_to_storage": handle_demote_hbm_to_storage,
    "demote_to_storage": handle_demote_to_storage,
    "detach_engine_pool": handle_detach_engine_pool,
    "ping": handle_ping,
    "promote_from_storage": handle_promote_from_storage,
    "promote_storage_to_hbm": handle_promote_storage_to_hbm,
    "prune_storage": handle_prune_storage,
    "release_engine_storage": handle_release_engine_storage,
    "storage_stats": handle_storage_stats,
}
