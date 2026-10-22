# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NIXL pull transfer (vLLM ``NixlConnector`` pull idiom), one agent per DP rank.

Both roles register ``gpu_blocks`` and ``gpu_states`` once and prepare a local
descriptor list with one descriptor per row: id = region offset + row. The
prefill handoff carries the agent's NIXL metadata plus its region layout
(base, rows, row bytes), so the decode side adds the peer lazily (cached by
agent name) and prepares the remote list without a side channel. Decode READs
with ``make_prepped_xfer`` and the notification ``"{request_id}:release"``,
which tells the prefill side it may unpin the KV.
"""

from __future__ import annotations

import base64
import logging
import os
import sys
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

import msgspec
import numpy as np

from .base import KVLayout, KVRegion, KVTransfer
from .rows import plan_read

if "nixl" in sys.modules and "UCX_RCACHE_MAX_UNRELEASED" not in os.environ:
    logging.getLogger(__name__).warning(
        "NIXL was imported before UCX_RCACHE_MAX_UNRELEASED was set; set it to 1024"
    )
# UCX reads this when NIXL loads; it bounds a UCX registration-cache leak.
os.environ.setdefault("UCX_RCACHE_MAX_UNRELEASED", "1024")

from nixl._api import nixl_agent, nixl_agent_config  # noqa: E402

if TYPE_CHECKING:
    from dynamo.deepinfer.disagg import DeepInferHandoff

logger = logging.getLogger(__name__)

RELEASE = "release"
# More UCX threads can exhaust the NIC's doorbell pages (UARs) that DeepEP's
# NVSHMEM/GIN also allocates.
MAX_NUM_THREADS = 4


@dataclass
class _Peer:
    name: str
    mem_type: str
    # (name, base_ptr, num_items, item_bytes, device_id) per region, in order.
    regions: list[tuple[str, int, int, int, int]]
    dlist: Any = None

    @property
    def shapes(self) -> dict[str, tuple[int, int]]:
        return {r[0]: (r[2], r[3]) for r in self.regions}


@dataclass
class _Read:
    handle: Any
    handoff: DeepInferHandoff
    failed: bool = False


@dataclass
class _Registered:
    layout: KVLayout
    mem_type: str
    reg_descs: Any
    dlist: Any
    offsets: dict[str, int] = field(default_factory=dict)
    metadata: str = ""


class NixlKVTransfer(KVTransfer):
    def __init__(
        self,
        *,
        rank: int,
        engine_id: str,
        num_threads: int = MAX_NUM_THREADS,
        **options: Any,
    ) -> None:
        super().__init__(rank=rank, engine_id=engine_id)
        # A random suffix keeps a restarted worker (same host and pid, as in a
        # container) from matching a peer cached under its old metadata.
        self.agent_name = f"{engine_id}-dp{rank}-{uuid.uuid4().hex[:8]}"
        config = nixl_agent_config(
            backends=["UCX"], num_threads=max(1, min(num_threads, MAX_NUM_THREADS))
        )
        self._agent = nixl_agent(self.agent_name, config)
        if "UCX" not in self._agent.backends:
            raise RuntimeError("NIXL has no UCX backend")
        self._reg: Optional[_Registered] = None
        self._peers: dict[str, _Peer] = {}
        self._reads: dict[str, _Read] = {}
        # (agent, request) pairs whose completed READ already sent the release.
        self._notified: set[tuple[str, str]] = set()
        self._closed = False

    # -- both roles -------------------------------------------------------------

    def register(self, layout: KVLayout) -> None:
        if self._reg is not None:
            raise RuntimeError("NixlKVTransfer.register() runs once")
        mem_type = _mem_type(layout.regions)
        regions = [
            (r.name, r.base_ptr, r.num_items, r.item_bytes, r.device_id)
            for r in layout.regions
        ]
        reg_descs = self._agent.register_memory(
            np.array(
                [(base, n * size, max(dev, 0)) for _, base, n, size, dev in regions],
                dtype=np.uint64,
            ),
            mem_type,
        )
        dlist = self._agent.prep_xfer_dlist(
            "NIXL_INIT_AGENT", _row_descs(regions), mem_type
        )
        blob = {
            "md": self._agent.get_agent_metadata(),
            "mem_type": mem_type,
            "regions": regions,
        }
        self._reg = _Registered(
            layout=layout,
            mem_type=mem_type,
            reg_descs=reg_descs,
            dlist=dlist,
            offsets=_offsets(regions),
            metadata=base64.b64encode(msgspec.msgpack.encode(blob)).decode(),
        )
        logger.info(
            "NIXL agent %s registered %s (%s)",
            self.agent_name,
            ", ".join(
                f"{r.name}[{r.num_items}x{r.item_bytes}B]" for r in layout.regions
            ),
            mem_type,
        )

    def agent_info(self) -> dict[str, str]:
        return {
            "agent_name": self.agent_name,
            "agent_metadata": self._registered().metadata,
        }

    def poll(self) -> tuple[list[str], list[str], list[str]]:
        done: list[str] = []
        failed: list[str] = []
        for rid, read in list(self._reads.items()):
            if not read.failed:
                try:
                    state = self._agent.check_xfer_state(read.handle)
                except Exception:
                    logger.exception("NIXL check_xfer_state failed for %s", rid)
                    state = "ERR"
                if state == "PROC":
                    continue
                if state != "DONE":
                    logger.error(
                        "KV READ for %s from %s failed (%s)",
                        rid,
                        read.handoff.agent_name,
                        state,
                    )
                    read.failed = True
            if not self._release_handle(rid, read):
                continue  # the backend may still write into our rows; retry
            del self._reads[rid]
            if read.failed:
                failed.append(rid)
            else:
                self._notified.add((read.handoff.agent_name, read.handoff.request_id))
                done.append(rid)
        return done, failed, self._released()

    def shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        for rid, read in self._reads.items():
            self._release_handle(rid, read)
        self._reads.clear()
        for peer in self._peers.values():
            try:
                if peer.dlist is not None:
                    peer.dlist.release()
                self._agent.remove_remote_agent(peer.name)
            except Exception:
                logger.exception("NIXL cleanup of peer %s failed", peer.name)
        self._peers.clear()
        if self._reg is not None:
            try:
                self._reg.dlist.release()
                self._agent.deregister_memory(self._reg.reg_descs)
            except Exception:
                logger.exception("NIXL deregistration failed")
            self._reg = None

    # -- decode role ------------------------------------------------------------

    def start_read(
        self,
        req_id: str,
        handoff: DeepInferHandoff,
        local_block_ids: list[int],
        local_state_id: int,
    ) -> None:
        reg = self._registered()
        if req_id in self._reads:
            raise ValueError(f"a read for {req_id} is already in flight")
        peer = self._peer(handoff)
        plan = plan_read(
            reg.layout, peer.shapes, handoff, local_block_ids, local_state_id
        )
        if peer.dlist is None:
            peer.dlist = self._agent.prep_xfer_dlist(
                peer.name, _row_descs(peer.regions), peer.mem_type
            )
        peer_offsets = _offsets(peer.regions)
        local_ids = [reg.offsets[name] + i for name, dst, _ in plan for i in dst]
        remote_ids = [peer_offsets[name] + i for name, _, src in plan for i in src]
        handle = self._agent.make_prepped_xfer(
            "READ",
            reg.dlist,
            local_ids,
            peer.dlist,
            remote_ids,
            notif_msg=f"{handoff.request_id}:{RELEASE}".encode(),
        )
        read = self._reads[req_id] = _Read(handle, handoff)
        try:
            status = self._agent.transfer(handle)
        except Exception:
            logger.exception("NIXL transfer() raised for %s", req_id)
            status = "ERR"
        if status == "ERR":
            # Reported by the next poll(), once the handle is released.
            logger.error("KV READ for %s could not be posted", req_id)
            read.failed = True

    def release_remote(self, handoff: DeepInferHandoff) -> None:
        key = (handoff.agent_name, handoff.request_id)
        if key in self._notified:
            self._notified.discard(key)
            return
        peer = self._peer(handoff)
        self._agent.send_notif(peer.name, f"{handoff.request_id}:{RELEASE}".encode())

    # -- internals --------------------------------------------------------------

    def _registered(self) -> _Registered:
        if self._reg is None:
            raise RuntimeError("NixlKVTransfer.register() has not run")
        return self._reg

    def _peer(self, handoff: DeepInferHandoff) -> _Peer:
        peer = self._peers.get(handoff.agent_name)
        if peer is not None:
            return peer
        try:
            blob = msgspec.msgpack.decode(base64.b64decode(handoff.agent_metadata))
            md, mem_type = blob["md"], blob["mem_type"]
            regions = [tuple(r) for r in blob["regions"]]
        except Exception as e:
            raise ValueError(f"bad NIXL metadata from {handoff.agent_name}: {e}") from e
        name = self._agent.add_remote_agent(md)
        name = name.decode() if isinstance(name, bytes) else name
        if name != handoff.agent_name:
            self._agent.remove_remote_agent(name)
            raise ValueError(
                f"NIXL metadata names agent {name!r}, handoff says {handoff.agent_name!r}"
            )
        peer = self._peers[name] = _Peer(name, mem_type, regions)
        logger.info("Added NIXL peer %s", name)
        return peer

    def _released(self) -> list[str]:
        released: list[str] = []
        for msgs in self._agent.get_new_notifs().values():
            for msg in msgs:
                rid, _, tag = bytes(msg).decode().rpartition(":")
                if tag == RELEASE and rid:
                    released.append(rid)
                else:
                    logger.warning("Ignoring unexpected NIXL notification %r", msg)
        return released

    def _release_handle(self, rid: str, read: _Read) -> bool:
        try:
            self._agent.release_xfer_handle(read.handle)
        except Exception:
            # A failed release does not prove the backend stopped writing.
            logger.exception("Could not release the NIXL handle of %s; retrying", rid)
            return False
        return True


def _mem_type(regions: Sequence[KVRegion]) -> str:
    kinds = {r.device_id >= 0 for r in regions}
    if len(kinds) != 1:
        raise ValueError("KV regions mix host and device memory")
    return "VRAM" if kinds.pop() else "DRAM"


def _offsets(regions: Sequence[tuple[str, int, int, int, int]]) -> dict[str, int]:
    offsets: dict[str, int] = {}
    total = 0
    for name, _, num_items, _, _ in regions:
        offsets[name] = total
        total += num_items
    return offsets


def _row_descs(regions: Sequence[tuple[str, int, int, int, int]]) -> np.ndarray:
    """Nx3 ``(addr, len, dev)``: one descriptor per row, regions in order."""
    parts = []
    for _, base, num_items, item_bytes, dev in regions:
        rows = np.empty((num_items, 3), dtype=np.uint64)
        rows[:, 0] = base + np.arange(num_items, dtype=np.uint64) * np.uint64(
            item_bytes
        )
        rows[:, 1] = item_bytes
        rows[:, 2] = max(dev, 0)
        parts.append(rows)
    return np.concatenate(parts)
