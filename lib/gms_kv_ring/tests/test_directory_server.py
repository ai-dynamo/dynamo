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
