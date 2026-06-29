# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GMS KV ring daemon — server + ring consumers.

One daemon process serves N engines. Control plane is a tiny Unix
socket protocol (msgpack-free, JSON-only for simplicity):

  attach_engine_pool(engine_id, layers[]):
      Caller has already CUDA-mapped the engine's HBM pool. Daemon
      records the per-layer (VA, size, stride) so consumers know
      where to DMA into. Real engine integration (cuMemImport via
      VMM-IPC shareable handles) is the engine hook's job; for tests
      and same-process use, pass pre-mapped VAs directly.

  attach_evict_ring(engine_id, ring_path):
      Spawn an evict-ring consumer for this engine. Records pushed
      to `ring_path` become D2H copies into the daemon's host tier.

  attach_restore_ring(engine_id, ring_path, counter_path, num_counters):
      Spawn a restore-ring consumer. Records become H2D copies +
      cuStreamWriteValue32 signal on the shared counter array.

  detach_engine_pool(engine_id):
      Stop consumers, free host-tier slots.

  ping():
      Liveness check.

Each engine has its OWN dedicated CUDA stream in the daemon process.
Engines do NOT serialize on each other — only on the asyncio control
loop, which is only used for setup/teardown.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import struct
from typing import Optional

from gms_kv_ring.daemon.consumers import EnginePool, LayerDesc
from gms_kv_ring.daemon.kv_cache_manager import GmsKvCacheManager

logger = logging.getLogger(__name__)

__all__ = ["Daemon", "EnginePool", "LayerDesc"]


class Daemon:
    """Unix-socket process shell for :class:`GmsKvCacheManager`."""

    def __init__(self, listen_socket: str, storage_dir=None, **manager_kwargs) -> None:
        self.listen_socket = listen_socket
        self.kv_cache_manager = GmsKvCacheManager(
            storage_dir=storage_dir,
            **manager_kwargs,
        )
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event = None

    def __getattr__(self, name):
        # Preserve the established Daemon surface while ownership moves to
        # the dedicated manager. New code should use ``kv_cache_manager``.
        return getattr(self.kv_cache_manager, name)

    @property
    def transport(self):
        return self.kv_cache_manager.transport

    @transport.setter
    def transport(self, value) -> None:
        self.kv_cache_manager.transport = value

    @property
    def placement_publisher(self):
        return self.kv_cache_manager.placement_publisher

    @placement_publisher.setter
    def placement_publisher(self, value) -> None:
        self.kv_cache_manager.placement_publisher = value

    async def serve(self) -> None:
        try:
            os.unlink(self.listen_socket)
        except FileNotFoundError:
            pass
        self._stop_event = asyncio.Event()
        self._server = await asyncio.start_unix_server(
            self._handle,
            path=self.listen_socket,
        )
        logger.info("daemon listening on %s", self.listen_socket)
        self.kv_cache_manager.start()
        try:
            await self._stop_event.wait()
        finally:
            self._server.close()
            await self._server.wait_closed()
            self.kv_cache_manager.close()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()
        self.kv_cache_manager.close_connections()

    async def _handle(self, reader, writer) -> None:
        try:
            while True:
                msg = await _read_frame(reader)
                if msg is None:
                    return
                resp = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self._dispatch,
                    msg,
                )
                # Stamp every response with the daemon epoch. The
                # connector reads this on each RPC; a change means
                # the daemon restarted and its in-memory state
                # (slot_generations, host_tier slots, etc.) has been
                # zeroed — the connector must drop its prefix index
                # to avoid restoring against slots the new daemon
                # doesn't know about.
                if isinstance(resp, dict):
                    resp["daemon_epoch"] = self.epoch
                await _write_frame(writer, resp)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _dispatch(self, msg: dict) -> dict:
        from gms_kv_ring.daemon.rpc_dispatch import dispatch

        return dispatch(self.kv_cache_manager, msg)


# ---- minimal length-prefixed JSON framing ----


async def _read_frame(reader) -> Optional[dict]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    n = struct.unpack("<I", header)[0]
    body = await reader.readexactly(n)
    return json.loads(body.decode("utf-8"))


async def _write_frame(writer, msg: dict) -> None:
    body = json.dumps(msg).encode("utf-8")
    writer.write(struct.pack("<I", len(body)) + body)
    await writer.drain()
