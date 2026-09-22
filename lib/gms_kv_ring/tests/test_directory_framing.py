# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The faster codec must remain compatible with existing JSON clients."""

import asyncio
import json
import struct

import pytest
from gms_kv_ring.daemon.framing import (
    FrameProtocolError,
    encode_frame,
    read_frame,
    recv_frame,
)

pytestmark = pytest.mark.pre_merge


MESSAGE = {
    "writer_id": "writer-λ",
    "directory_epoch": 2**64 - 1,
    "published": 2,
    "items": [{"hash": "ab" * 32, "slots": [0, 42], "active": False}],
    "error": None,
}


class Fragments:
    def __init__(self, data):
        self.data = data

    def recv(self, count):
        # Exercise header and body split across multiple socket reads.
        result, self.data = self.data[: min(count, 3)], self.data[min(count, 3) :]
        return result


def legacy_frame(message):
    body = json.dumps(message).encode("utf-8")
    return struct.pack("<I", len(body)) + body


def test_new_encoder_is_readable_by_existing_clients():
    frame = encode_frame(MESSAGE)
    assert struct.unpack("<I", frame[:4])[0] == len(frame) - 4
    assert json.loads(frame[4:].decode("utf-8")) == MESSAGE


@pytest.mark.parametrize("encode", [encode_frame, legacy_frame])
def test_sync_decoder_accepts_both_codecs(encode):
    assert recv_frame(Fragments(encode(MESSAGE))) == MESSAGE


@pytest.mark.asyncio
@pytest.mark.parametrize("encode", [encode_frame, legacy_frame])
async def test_async_decoder_accepts_both_codecs(encode):
    reader = asyncio.StreamReader()
    reader.feed_data(encode(MESSAGE))
    reader.feed_eof()
    assert await read_frame(reader) == MESSAGE
    assert await read_frame(reader, allow_eof=True) is None


@pytest.mark.parametrize("data", [b"", b"\x01", legacy_frame(MESSAGE)[:-1]])
def test_truncated_frames_still_fail_closed(data):
    with pytest.raises(FrameProtocolError):
        recv_frame(Fragments(data))


def test_malformed_json_is_not_accepted():
    with pytest.raises(FrameProtocolError, match="invalid JSON"):
        recv_frame(Fragments(struct.pack("<I", 1) + b"{"))
