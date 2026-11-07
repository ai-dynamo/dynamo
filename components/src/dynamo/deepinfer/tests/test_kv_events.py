# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM-format ZMQ KV event wire, decoded the way Dynamo's ZmqSource does
(lib/llm/src/kv_router/publisher/zmq_listener.rs, lib/kv-router/src/zmq_wire)."""

import shutil
import tempfile
import time
from typing import Any

import msgspec
import pytest
import zmq

from dynamo.deepinfer.kv_cache.base import KVEvent, make_event_sink
from dynamo.deepinfer.protocol import AddReq, Cmd, Op
from dynamo.deepinfer.rank import RankRunner, RankSpec
from dynamo.deepinfer.request import prompt_bytes

try:
    from dynamo.deepinfer.kv_cache.events import ZmqEventSink
    from dynamo.deepinfer.kv_cache.vllm.kv_events import KVEventBatch
except ImportError as e:  # cbor2, needed by vLLM's block hashing
    pytest.skip(f"Skip (missing dependency): {e}", allow_module_level=True)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def endpoint():
    # Short path: ipc:// socket paths are limited to ~107 bytes.
    workdir = tempfile.mkdtemp(prefix="di-ev-")
    yield f"ipc://{workdir}/events"
    shutil.rmtree(workdir, ignore_errors=True)


@pytest.fixture
def subscriber(endpoint):
    sock = zmq.Context.instance().socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"")
    sock.setsockopt(zmq.LINGER, 0)
    yield sock
    sock.close()


def receive(sock: zmq.Socket, publish, timeout_s: float = 5.0) -> list[bytes]:
    """Publish until the subscription is live (PUB drops messages before it)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        publish()
        if sock.poll(100):
            return sock.recv_multipart()
    raise AssertionError("no KV event batch received")


def decode_like_dynamo(frames: list[bytes]) -> tuple[int, list[Any], int]:
    """Mirror Dynamo's decoder: three frames, a big-endian u64 sequence, and a
    msgpack array ``[ts: f64, events, data_parallel_rank]`` of tagged maps."""
    assert len(frames) == 3
    topic, seq, payload = frames
    assert topic == b"" and len(seq) == 8
    batch = msgspec.msgpack.decode(payload)
    assert isinstance(batch, list) and len(batch) == 3
    ts, events, dp_rank = batch
    assert isinstance(ts, float) and isinstance(dp_rank, int)
    for event in events:
        assert isinstance(event, dict) and "type" in event
        if event["type"] == "BlockStored":
            hashes = event["block_hashes"]
            assert all(isinstance(h, int) and 0 <= h < 1 << 64 for h in hashes)
            # Dynamo drops stores whose hashes (with the parent) repeat.
            chain = hashes + [event["parent_block_hash"]]
            assert len(set(chain)) == len(chain)
            assert len(event["token_ids"]) == len(hashes) * event["block_size"]
        if event["type"] in ("BlockStored", "BlockRemoved"):
            # Dynamo indexes "GPU" as the device tier; unknown media are dropped.
            assert event["medium"] == "GPU"
    return int.from_bytes(seq, "big"), events, dp_rank


def test_sink_publishes_vllm_wire_format(endpoint, subscriber):
    sink = make_event_sink(endpoint, dp_rank=3, block_size=4)
    assert isinstance(sink, ZmqEventSink)
    subscriber.connect(endpoint)
    events = [
        KVEvent(
            kind="stored",
            block_hashes=[(1 << 64) - 1, 7],
            parent_block_hash=5,
            token_ids=list(range(8)),
            block_size=4,
        ),
        KVEvent(kind="removed", block_hashes=[7]),
        KVEvent(kind="cleared"),
    ]
    frames = receive(subscriber, lambda: sink.publish(events))
    seq, raw, dp_rank = decode_like_dynamo(frames)
    assert dp_rank == 3
    sink.publish(events[2:])
    assert subscriber.poll(2000)
    assert decode_like_dynamo(subscriber.recv_multipart())[0] == seq + 1
    assert raw == [
        {
            "type": "BlockStored",
            "block_hashes": [(1 << 64) - 1, 7],
            "parent_block_hash": 5,
            "token_ids": list(range(8)),
            "block_size": 4,
            "lora_id": None,
            "medium": "GPU",
            "lora_name": None,
        },
        {"type": "BlockRemoved", "block_hashes": [7], "medium": "GPU"},
        {"type": "AllBlocksCleared"},
    ]
    typed = msgspec.msgpack.Decoder(KVEventBatch).decode(frames[2])
    assert [type(e).__name__ for e in typed.events] == [
        "BlockStored",
        "BlockRemoved",
        "AllBlocksCleared",
    ]
    sink.close()


def test_sink_rejects_unknown_event_kind(endpoint):
    sink = ZmqEventSink(endpoint, dp_rank=0)
    with pytest.raises(ValueError, match="unknown KV event kind"):
        sink.publish([KVEvent(kind="moved")])  # type: ignore[arg-type]
    sink.close()


def test_rank_streams_cache_events_in_chain_order(endpoint, subscriber):
    runner = RankRunner(
        RankSpec(
            rank=2,
            dp_size=1,
            engine_args={
                "model": "fake",
                "chunked_prefill": True,
                "max_model_len": 4096,
                "data_parallel_size": 1,
                "engine_rank": 0,
            },
            backend="fake",
            cache_manager="prefix",
            cache_options={"snapshot_interval": 256, "snapshot_slots": 4},
            kv_events_endpoint=endpoint,
        )
    )
    subscriber.connect(endpoint)
    probe = [KVEvent(kind="cleared")]
    receive(subscriber, lambda: runner.sink.publish(probe))
    while subscriber.poll(100):
        subscriber.recv_multipart()  # drain extra probes

    tokens = [2 + (i * 7) % 900 for i in range(600)]
    add = AddReq("r", prompt_bytes(tokens), sampling={"max_tokens": 2})
    published = 0
    for i in range(20):
        reply = runner.apply(Cmd(seq=i, adds=[add] if i == 0 else []))
        published += reply.kv_events_published
    stores: list[dict[str, Any]] = []
    while len(stores) < published and subscriber.poll(2000):
        _, events, dp_rank = decode_like_dynamo(subscriber.recv_multipart())
        assert dp_rank == 2
        stores += [e for e in events if e["type"] == "BlockStored"]
    # Checkpoints at 256 and 384: blocks 0-1, then block 2 chained to block 1.
    assert [len(e["block_hashes"]) for e in stores] == [2, 1]
    assert stores[0]["parent_block_hash"] is None
    assert stores[1]["parent_block_hash"] == stores[0]["block_hashes"][-1]
    assert stores[0]["token_ids"] + stores[1]["token_ids"] == tokens[:384]
    runner.apply(Cmd(seq=100, op=Op.CLOSE))
