# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""KV transfer backends: LocalKVTransfer, and NixlKVTransfer over host memory."""

import importlib
import time
import uuid
from typing import Any, Callable

import numpy as np
import pytest

from dynamo.deepinfer.disagg import HANDOFF_VERSION, DeepInferHandoff
from dynamo.deepinfer.kv_transfer.base import (
    KVLayout,
    KVRegion,
    KVTransfer,
    make_kv_transfer,
)
from dynamo.deepinfer.kv_transfer.local import LocalKVTransfer

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

BLOCK_BYTES = 256
STATE_BYTES = 64


def make_layout(
    num_blocks: int = 16, block_bytes: int = BLOCK_BYTES, seed: int = 0
) -> KVLayout:
    rng = np.random.default_rng(seed)
    blocks = rng.integers(0, 256, (num_blocks, block_bytes), dtype=np.uint8)
    states = rng.integers(0, 256, (6, STATE_BYTES), dtype=np.uint8)
    return KVLayout(
        regions=[
            KVRegion(
                "gpu_blocks", blocks.ctypes.data, num_blocks, block_bytes, -1, blocks
            ),
            KVRegion("gpu_states", states.ctypes.data, 6, STATE_BYTES, -1, states),
        ],
        layout_hash="h",
    )


def make_handoff(
    peer: KVTransfer, block_ids: list[int], state_id: int = 3, request_id: str = "r#0"
) -> DeepInferHandoff:
    info = peer.agent_info()
    return DeepInferHandoff(
        version=HANDOFF_VERSION,
        engine_id="P",
        dp_rank=0,
        request_id=request_id,
        agent_name=info["agent_name"],
        agent_metadata=info["agent_metadata"],
        layout_hash="h",
        block_ids=block_ids,
        state_id=state_id,
        num_tokens=128 * len(block_ids),
        first_token=7,
        expiry_ts=time.time() + 60,
    )


def rows(layout: KVLayout, region: str, ids: list[int]) -> np.ndarray:
    return layout.region(region).buffer[ids]


class Polled:
    """Accumulates ``poll()`` results across calls."""

    def __init__(self, transfer: KVTransfer) -> None:
        self.transfer = transfer
        self.done: list[str] = []
        self.failed: list[str] = []
        self.released: list[str] = []

    def poll(self) -> "Polled":
        done, failed, released = self.transfer.poll()
        self.done += done
        self.failed += failed
        self.released += released
        return self

    def until(self, predicate: Callable[["Polled"], bool], timeout: float = 10.0):
        deadline = time.monotonic() + timeout
        while not predicate(self.poll()):
            assert time.monotonic() < deadline, vars(self)
            time.sleep(0.001)
        return self


def _new_local(engine_id: str, **options: Any) -> KVTransfer:
    return LocalKVTransfer(rank=0, engine_id=engine_id, **options)


def _new_nixl(engine_id: str, **options: Any) -> KVTransfer:
    nixl = pytest.importorskip("dynamo.deepinfer.kv_transfer.nixl")
    try:
        return nixl.NixlKVTransfer(rank=0, engine_id=engine_id, **options)
    except Exception as e:  # no UCX on this host
        pytest.skip(f"NIXL/UCX cannot initialize here: {e}")


@pytest.fixture(params=["local", "nixl"])
def new_transfer(request):
    """Factory for registered transfers of one backend; shut down after the test."""
    made: list[KVTransfer] = []
    factory = _new_local if request.param == "local" else _new_nixl

    def make(layout: KVLayout, **options: Any) -> KVTransfer:
        transfer = factory(f"{request.param}-{uuid.uuid4().hex[:6]}", **options)
        transfer.register(layout)
        made.append(transfer)
        return transfer

    yield make
    for transfer in made:
        transfer.shutdown()


def test_read_copies_rows_and_completion_releases_the_peer(new_transfer):
    src_layout, dst_layout = make_layout(seed=1), make_layout(seed=2)
    src, dst = new_transfer(src_layout), new_transfer(dst_layout)
    handoff = make_handoff(src, block_ids=[5, 2, 9])

    dst.start_read("d1", handoff, local_block_ids=[1, 6, 0], local_state_id=2)
    got = Polled(dst).until(lambda p: p.done or p.failed)
    assert (got.done, got.failed) == (["d1"], [])
    assert np.array_equal(
        rows(dst_layout, "gpu_blocks", [1, 6, 0]),
        rows(src_layout, "gpu_blocks", [5, 2, 9]),
    )
    assert np.array_equal(
        rows(dst_layout, "gpu_states", [2]), rows(src_layout, "gpu_states", [3])
    )
    # Untouched rows stay as they were.
    assert not np.array_equal(
        rows(dst_layout, "gpu_blocks", [3]), rows(src_layout, "gpu_blocks", [3])
    )
    assert Polled(src).until(lambda p: p.released).released == ["r#0"]

    # The completed read already released it; this must not notify twice.
    dst.release_remote(handoff)
    time.sleep(0.05)
    assert Polled(src).poll().released == []


def test_release_remote_without_a_read_notifies_the_peer(new_transfer):
    src, dst = new_transfer(make_layout()), new_transfer(make_layout())
    dst.release_remote(make_handoff(src, block_ids=[1], request_id="r#4"))
    assert Polled(src).until(lambda p: p.released).released == ["r#4"]


@pytest.mark.parametrize(
    "case",
    ["block_bytes", "count", "remote_range", "local_range", "state_range"],
)
def test_layout_or_id_mismatch_refuses_to_start(new_transfer, case):
    src = new_transfer(make_layout(block_bytes=128 if case == "block_bytes" else 256))
    dst = new_transfer(make_layout())
    block_ids, local_ids, state_id = [1, 2], [3, 4], 1
    if case == "count":
        local_ids = [3]
    elif case == "remote_range":
        block_ids = [1, 16]
    elif case == "local_range":
        local_ids = [3, 99]
    elif case == "state_range":
        state_id = 6
    handoff = make_handoff(src, block_ids=block_ids, state_id=state_id)
    with pytest.raises(ValueError):
        dst.start_read("d1", handoff, local_ids, local_state_id=0)
    assert Polled(dst).poll().done == []


def test_local_unknown_peer_raises():
    dst = LocalKVTransfer(rank=0, engine_id="D")
    dst.register(make_layout())
    peer = LocalKVTransfer(rank=0, engine_id="gone")
    with pytest.raises(ConnectionError):
        dst.start_read("d1", make_handoff(peer, [1]), [1], 0)
    dst.release_remote(make_handoff(peer, [1]))  # unknown peer: nothing to do
    dst.shutdown()


def test_local_delay_polls_keeps_the_read_in_flight():
    src = LocalKVTransfer(rank=0, engine_id="P")
    dst = LocalKVTransfer(rank=0, engine_id="D", delay_polls=2)
    src.register(make_layout(seed=1))
    dst.register(dst_layout := make_layout(seed=2))
    dst.start_read("d1", make_handoff(src, [4]), [5], 1)
    assert [dst.poll()[0] for _ in range(3)] == [[], [], ["d1"]]
    assert np.array_equal(
        rows(dst_layout, "gpu_blocks", [5]), rows(src._layout, "gpu_blocks", [4])
    )
    src.shutdown()
    dst.shutdown()


def test_local_peer_gone_mid_read_fails():
    src = LocalKVTransfer(rank=0, engine_id="P")
    dst = LocalKVTransfer(rank=0, engine_id="D", delay_polls=1)
    src.register(make_layout())
    dst.register(make_layout())
    dst.start_read("d1", make_handoff(src, [4]), [5], 1)
    assert dst.poll() == ([], [], [])
    src.shutdown()
    assert dst.poll() == ([], ["d1"], [])
    dst.shutdown()


def test_make_kv_transfer():
    assert make_kv_transfer("none", rank=0, engine_id="e") is None
    local = make_kv_transfer("local", rank=1, engine_id="e", delay_polls=3)
    assert isinstance(local, LocalKVTransfer) and local.delay_polls == 3
    assert local.agent_info()["agent_name"].startswith("e-dp1-")
    with pytest.raises(ValueError, match="unknown kv transfer"):
        make_kv_transfer("rdma", rank=0, engine_id="e")


def test_nixl_metadata_and_threads():
    transfer = _new_nixl("meta", num_threads=64)
    nixl = importlib.import_module("dynamo.deepinfer.kv_transfer.nixl")
    assert transfer._agent.backend_options["UCX"]["num_threads"] == str(
        nixl.MAX_NUM_THREADS
    )
    transfer.register(make_layout())
    info = transfer.agent_info()
    peer = _new_nixl("reader")
    peer.register(make_layout())
    handoff = make_handoff(transfer, [1])
    # A handoff whose metadata names a different agent is refused.
    with pytest.raises(ValueError, match="names agent"):
        peer.start_read(
            "d1", DeepInferHandoff(**{**handoff.to_dict(), "agent_name": "x"}), [1], 0
        )
    assert info["agent_name"].startswith("meta-dp0-")
    for t in (transfer, peer):
        t.shutdown()
        t.shutdown()  # idempotent
