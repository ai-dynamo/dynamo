# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory-only Unix-socket service for persistent HBM metadata.

The first GMS failover loop only needs content identity, writer fencing and
HBM slot generations.  Host/storage movement is deliberately absent; the
full KV daemon composes the same RPC handlers when those tiers are enabled.
"""

from __future__ import annotations

import asyncio
import json
import os
import struct
import threading
import time
from collections import deque
from typing import Optional

from gms_kv_ring.daemon.rpc_directory import DIRECTORY_HANDLERS
from gms_kv_ring.daemon.rpc_types import error_response


class DirectoryState:
    """Minimal state required by the engine-neutral directory handlers."""

    def __init__(self) -> None:
        self.epoch = time.time_ns()
        self._content_directory: dict[tuple[str, bytes], dict] = {}
        self._content_directory_by_slot: dict[tuple[str, str, int], bytes] = {}
        self._content_directory_claims: dict[str, dict] = {}
        self._content_directory_access_seq = 0
        self._content_directory_epoch = 1
        self._content_directory_revision = 0
        self._content_directory_changes = deque(maxlen=131_072)
        self._content_directory_writer_id: Optional[str] = None
        self._content_hash_lock = threading.Condition()


class DirectoryDaemon:
    """Small JSON-RPC process shell around :class:`DirectoryState`."""

    def __init__(self, listen_socket: str) -> None:
        self.listen_socket = listen_socket
        self.state = DirectoryState()
        self._server: Optional[asyncio.AbstractServer] = None
        self._stop_event: Optional[asyncio.Event] = None

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
        # Restrict the directory socket to the owning user: any client on this
        # socket can promote the directory writer (fencing the real writer), so
        # do not leave it world-accessible under the process umask.
        try:
            os.chmod(self.listen_socket, 0o600)
        except OSError:
            pass
        try:
            await self._stop_event.wait()
        finally:
            self._server.close()
            await self._server.wait_closed()

    def stop(self) -> None:
        if self._stop_event is not None:
            self._stop_event.set()

    def _dispatch(self, msg: dict) -> dict:
        op = msg.get("op")
        if op == "ping":
            return {"ok": True}
        handler = DIRECTORY_HANDLERS.get(op) if isinstance(op, str) else None
        if handler is None:
            return {"ok": False, "error": f"unknown op {op!r}"}
        try:
            return handler(self.state, msg)
        except Exception as exc:  # noqa: BLE001
            return error_response(exc)

    async def _handle(self, reader, writer) -> None:
        try:
            while True:
                msg = await _read_frame(reader)
                if msg is None:
                    return
                response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._dispatch,
                    msg,
                )
                response["daemon_epoch"] = self.state.epoch
                await _write_frame(writer, response)
        except (asyncio.IncompleteReadError, ConnectionResetError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass


async def _read_frame(reader) -> Optional[dict]:
    try:
        header = await reader.readexactly(4)
    except asyncio.IncompleteReadError:
        return None
    length = struct.unpack("<I", header)[0]
    return json.loads((await reader.readexactly(length)).decode("utf-8"))


async def _write_frame(writer, message: dict) -> None:
    body = json.dumps(message).encode("utf-8")
    writer.write(struct.pack("<I", len(body)) + body)
    await writer.drain()
