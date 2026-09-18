# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
from gms_kv_ring.daemon.directory_server import DirectoryDaemon
from gms_kv_ring.daemon.framing import read_frame, write_frame

pytestmark = pytest.mark.pre_merge


@pytest.mark.asyncio
async def test_live_directory_endpoint_cannot_be_replaced(tmp_path):
    socket_path = str(tmp_path / "directory.sock")
    first = DirectoryDaemon(socket_path)
    first_task = asyncio.create_task(first.serve())
    try:
        for _ in range(100):
            if first._server is not None:
                break
            await asyncio.sleep(0.01)
        assert first._server is not None

        second = DirectoryDaemon(socket_path)
        with pytest.raises(RuntimeError, match="live owner"):
            await second.serve()

        reader, writer = await asyncio.open_unix_connection(socket_path)
        await write_frame(writer, {"op": "ping"})
        assert (await read_frame(reader))["ok"] is True
        writer.close()
        await writer.wait_closed()
    finally:
        first.stop()
        await first_task


@pytest.mark.asyncio
async def test_complete_publish_frame_survives_client_disconnect(tmp_path):
    """The daemon commits a frame even when the engine never reads its ACK."""
    socket_path = str(tmp_path / "directory.sock")
    daemon = DirectoryDaemon(socket_path)
    daemon_task = asyncio.create_task(daemon.serve())
    try:
        for _ in range(100):
            if daemon._server is not None:
                break
            await asyncio.sleep(0.01)
        assert daemon._server is not None

        reader, writer = await asyncio.open_unix_connection(socket_path)
        await write_frame(
            writer,
            {"op": "directory_promote", "writer_id": "engine-0", "expected_epoch": 1},
        )
        promotion = await read_frame(reader)
        assert promotion["promoted"] is True
        epoch = promotion["directory_epoch"]
        writer.close()
        await writer.wait_closed()

        # Model an engine crash immediately after sendall(): the complete frame
        # is in the daemon-owned socket buffer, but no engine thread remains to
        # consume the acknowledgement.
        _, writer = await asyncio.open_unix_connection(socket_path)
        await write_frame(
            writer,
            {
                "op": "directory_publish_batch",
                "manifest_id": "m",
                "writer_id": "engine-0",
                "expected_epoch": epoch,
                "items": [
                    {
                        "content_hash": b"ready".hex(),
                        "engine_id": "0",
                        "slot_ids": [7],
                        "generations": [3],
                        "tier": "hbm",
                    }
                ],
            },
        )
        writer.close()
        await writer.wait_closed()

        reader, writer = await asyncio.open_unix_connection(socket_path)
        for _ in range(100):
            await write_frame(
                writer,
                {
                    "op": "directory_lookup",
                    "manifest_id": "m",
                    "hashes": [b"ready".hex()],
                },
            )
            response = await read_frame(reader)
            if response["entries"][0] is not None:
                break
            await asyncio.sleep(0.01)
        assert response["entries"][0]["slot_ids"] == [7]
        assert response["entries"][0]["generations"] == [3]
        writer.close()
        await writer.wait_closed()
    finally:
        daemon.stop()
        await daemon_task
