# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Directory-only Unix-socket service for persistent HBM metadata.

The first GMS failover loop only needs content identity, writer fencing and
HBM slot generations.  Host/storage movement is deliberately absent; the
full KV daemon composes the same RPC handlers when those tiers are enabled.
"""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import os
import threading
import time
from collections import deque
from typing import Optional

from gms_kv_ring.daemon.framing import FrameProtocolError, read_frame, write_frame
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
        self._endpoint_lock_fd: Optional[int] = None

    def _acquire_endpoint_lock(self) -> None:
        lock_path = f"{self.listen_socket}.lock"
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        os.fchmod(lock_fd, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(lock_fd)
            raise RuntimeError(
                f"directory endpoint already has a live owner: {self.listen_socket}"
            ) from exc
        self._endpoint_lock_fd = lock_fd

    def _release_endpoint_lock(self) -> None:
        if self._endpoint_lock_fd is None:
            return
        os.close(self._endpoint_lock_fd)
        self._endpoint_lock_fd = None

    async def serve(self) -> None:
        self._acquire_endpoint_lock()
        try:
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
                try:
                    os.unlink(self.listen_socket)
                except FileNotFoundError:
                    pass
        finally:
            self._release_endpoint_lock()

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
                msg = await read_frame(reader, allow_eof=True)
                if msg is None:
                    return
                response = await asyncio.get_running_loop().run_in_executor(
                    None,
                    self._dispatch,
                    msg,
                )
                response["daemon_epoch"] = self.state.epoch
                await write_frame(writer, response)
        except (ConnectionResetError, FrameProtocolError):
            return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("listen_socket", help="Unix socket used by directory clients")
    args = parser.parse_args()
    asyncio.run(DirectoryDaemon(args.listen_socket).serve())


if __name__ == "__main__":
    main()
