# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PrefixCacheManager over the CPU fake of DeepInfer: hits, checkpoints,
eviction, refcounts and events."""

import fcntl
import hashlib
import os
import pickle
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from dynamo.deepinfer.kv_cache.base import Admission
from dynamo.deepinfer.protocol import AddReq, Cmd, Op
from dynamo.deepinfer.rank import RankRunner, RankSpec
from dynamo.deepinfer.request import prompt_bytes

pytest.importorskip("cbor2", reason="vLLM's block hashing (kv_cache/vllm) needs it")

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

BS = 128


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.delenv("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", raising=False)
    monkeypatch.delenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", raising=False)


def make_runner(
    cache: str = "prefix", engine_type: str = "NORMAL", **options: Any
) -> RankRunner:
    engine_args = {
        "model": "fake",
        "chunked_prefill": options.pop("chunked_prefill", True),
        "max_model_len": 4096,
        "data_parallel_size": 1,
        "engine_rank": 0,
    }
    if "max_num_seqs" in options:
        engine_args["max_num_seqs"] = options.pop("max_num_seqs")
    if cache == "prefix":
        options = {"snapshot_interval": 256, "snapshot_slots": 16, **options}
    return RankRunner(
        RankSpec(
            rank=0,
            dp_size=1,
            engine_args=engine_args,
            engine_type=engine_type,
            backend="fake",
            cache_manager=cache,
            cache_options=options if cache == "prefix" else {},
        )
    )


class Driver:
    """Steps one rank directly and records admissions and KV events."""

    def __init__(self, runner: RankRunner) -> None:
        self.runner = runner
        self.cache = runner.cache
        self.bm = runner.engine.block_manager
        self.seq = 0
        self.admissions: dict[str, Admission] = {}
        self.events: list[Any] = []
        # The blocks the events left stored (whatever the tests clear from
        # ``events``), and any stored twice or removed unknown.
        self.live: set[int] = set()
        self.event_errors: list[str] = []
        admit, take = self.cache.admit, self.cache.take_events

        def spy_admit(seq: Any, tokens: Any) -> Admission:
            result = admit(seq, tokens)
            self.admissions[seq.request_id] = result
            return result

        def spy_take() -> list[Any]:
            events = take()
            self.events.extend(events)
            for e in events:
                for h in e.block_hashes:
                    if (e.kind == "stored") == (h in self.live):
                        self.event_errors.append(f"{e.kind} {h} twice or unknown")
                    (self.live.add if e.kind == "stored" else self.live.discard)(h)
            return events

        self.cache.admit = spy_admit
        self.cache.take_events = spy_take

    def apply(self, **kw: Any):
        self.seq += 1
        return self.runner.apply(Cmd(seq=self.seq, **kw))

    def add(self, rid: str, tokens: list[int], max_tokens: int = 4, **kw: Any):
        sampling = {"max_tokens": max_tokens}
        return self.apply(
            adds=[AddReq(rid, prompt_bytes(tokens), sampling=sampling, **kw)]
        )

    def run(self, rid: str, tokens: list[int], max_tokens: int = 4) -> list[int]:
        out: list[int] = []
        reply = self.add(rid, tokens, max_tokens)
        for _ in range(200):
            for o in reply.outputs:
                if o.request_id == rid:
                    out += o.token_ids
                    if o.finish_reason is not None:
                        assert o.finish_reason == "length", o
                        return out
            reply = self.apply()
        raise AssertionError(f"{rid} did not finish")

    def close(self) -> None:
        self.apply(op=Op.CLOSE)


def prompt(n: int, salt: int = 0) -> list[int]:
    return [2 + (i * 7 + salt * 131) % 900 for i in range(n)]


def vllm_hash(parent: bytes | None, tokens: list[int]) -> bytes:
    # vLLM's sha256 prefix hash with the default NONE_HASH seed, written out.
    if parent is None:
        parent = hashlib.sha256(
            pickle.dumps("vllm-none-hash", protocol=pickle.HIGHEST_PROTOCOL)
        ).digest()
    data = (parent, tuple(tokens), None)
    return hashlib.sha256(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)).digest()


def as_u64(h: bytes) -> int:
    return int.from_bytes(h, "big") & ((1 << 64) - 1)


def cached_ids(cache: Any) -> set[int]:
    return {b.block_id for b in cache._blocks.values()}


def check_quiescent(d: Driver) -> None:
    """No live requests: every cached block is pinned exactly once."""
    total = d.runner.engine.cache_config.num_gpu_blocks
    for block in d.cache._blocks.values():
        assert d.bm.ref_count(block.block_id) == 1 and block.users == 0
    assert d.bm.get_num_free_blocks() == total - 1 - len(d.cache._blocks)
    snapshots = sum(b.snapshot >= 0 for b in d.cache._blocks.values())
    assert d.cache._pool.num_free == 16 - snapshots


def test_checkpoints_are_block_aligned_and_keep_the_tail(monkeypatch):
    cache = make_runner(snapshot_interval=300).cache
    assert cache.keep_tokens == 140 and cache.snapshot_interval == 256
    assert cache._checkpoints(0, 600) == [256, 384]  # tail = (600-140)//128*128
    assert cache._checkpoints(256, 600) == [384]
    assert cache._checkpoints(384, 600) == []
    assert cache._checkpoints(0, 268) == [128]
    assert cache._checkpoints(0, 267) == []  # too short to cache a block
    assert cache._checkpoints(0, 1200) == [256, 512, 768, 1024]
    monkeypatch.setenv("DEEPINFER_YOCO_PREFILL_KEEP_TOKENS", "300")
    cache = make_runner(snapshot_interval=0).cache
    assert cache._checkpoints(0, 1200) == [896]


def test_requires_chunked_prefill():
    with pytest.raises(ValueError, match="chunked_prefill"):
        make_runner(chunked_prefill=False)


@pytest.mark.parametrize("device", ["gpu", "host"])
def test_full_hit_reproduces_cold_tokens_and_publishes_vllm_hashes(device):
    d = Driver(make_runner(snapshot_device=device))
    cold = Driver(make_runner("none"))
    p = prompt(600)
    first = d.run("a", p)
    assert d.admissions["a"].hit_tokens == 0
    assert d.run("b", p) == first == cold.run("c", p)
    assert d.admissions["b"].hit_tokens == 384
    assert d.cache.stats().hit_rate == pytest.approx(384 / 1200)

    stored = [e for e in d.events if e.kind == "stored"]
    assert [len(e.block_hashes) for e in stored] == [2, 1]  # checkpoints 256, 384
    h0 = vllm_hash(None, p[:128])
    h1 = vllm_hash(h0, p[128:256])
    h2 = vllm_hash(h1, p[256:384])
    assert stored[0].block_hashes == [as_u64(h0), as_u64(h1)]
    assert stored[0].parent_block_hash is None
    assert stored[0].token_ids == p[:256]
    assert stored[1].block_hashes == [as_u64(h2)]
    assert stored[1].parent_block_hash == as_u64(h1)
    assert stored[1].token_ids == p[256:384]
    check_quiescent(d)
    d.close()
    cold.close()


def test_partial_prefix_hits_the_deepest_snapshot():
    d = Driver(make_runner())
    cold = Driver(make_runner("none"))
    p = prompt(600)
    d.run("a", p)
    q = p[:300] + prompt(200, salt=1)
    assert d.run("q", q) == cold.run("q", q)
    assert d.admissions["q"].hit_tokens == 256
    check_quiescent(d)
    d.close()
    cold.close()


def test_miss_when_the_first_block_differs():
    d = Driver(make_runner())
    p = prompt(600)
    d.run("a", p)
    d.run("b", [p[0] + 1] + p[1:])
    assert d.admissions["b"].hit_tokens == 0
    # Both prompts are cached side by side; no blocks are shared.
    assert len(d.cache._blocks) == 6
    check_quiescent(d)
    d.close()


def test_restored_state_matters():
    """Guards the fake: skipping the snapshot restore changes the tokens."""
    d = Driver(make_runner())
    p = prompt(600)
    first = d.run("a", p)
    d.cache._pool.load = lambda slot, state_id, pos: None
    assert d.run("b", p) != first
    d.close()


def test_hit_pins_shared_blocks_once_per_holder():
    d = Driver(make_runner())
    p = prompt(600)
    d.run("a", p)
    check_quiescent(d)
    shared = [b.block_id for b in d.cache._blocks.values()]
    d.add("b", p, max_tokens=50)
    [req] = d.cache._requests.values()
    seq = req.seq
    assert d.admissions["b"].hit_tokens == 384 and seq.block_table[:3] == shared
    assert all(d.bm.ref_count(b) == 2 for b in shared)
    assert all(b.users == 1 for b in d.cache._blocks.values())
    d.apply(aborts=["b"])
    d.apply()
    check_quiescent(d)
    d.close()


def test_eviction_is_lru_tail_first_and_frees_everything():
    d = Driver(make_runner())
    prompts = {name: prompt(600, salt=i) for i, name in enumerate("ABC")}
    for name, p in prompts.items():
        d.run(name, p)
    d.run("A2", prompts["A"])  # A becomes the most recently used
    assert d.admissions["A2"].hit_tokens == 384
    chain_b = [vllm_hash(None, prompts["B"][:128])]
    for i in (1, 2):
        chain_b.append(vllm_hash(chain_b[-1], prompts["B"][i * BS : (i + 1) * BS]))
    d.events.clear()

    free = d.bm.get_num_free_blocks()
    assert d.cache.ensure_free(free + 3) == 3
    d.cache.take_events()
    assert [e.kind for e in d.events] == ["removed"]
    assert d.events[0].block_hashes == [as_u64(h) for h in chain_b[::-1]]
    assert not set(chain_b) & set(d.cache._blocks)
    check_quiescent(d)

    total = d.runner.engine.cache_config.num_gpu_blocks
    assert d.cache.ensure_free(total - 1) == 6
    assert not d.cache._blocks and d.bm.get_num_free_blocks() == total - 1
    assert d.cache._pool.num_free == 16
    d.close()


def test_waiting_miss_evicts_only_when_it_can_fit(monkeypatch):
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "24")  # 23 usable
    d = Driver(make_runner())
    assert d.cache.floor_blocks == 3  # ceil(10% of 24): the prefill watermark
    d.run("A", prompt(600, salt=1))
    d.run("B", prompt(600, salt=2))
    old = set(d.cache._blocks)
    assert len(old) == 6
    # R holds 10 blocks while it decodes; its own captures are not evictable.
    d.add("R", prompt(1200, salt=3), max_tokens=10_000)
    scheduler = d.runner.engine.scheduler
    while not scheduler.running:
        d.apply()
    assert d.bm.get_num_free_blocks() == 23 - 6 - 10

    # A miss is queued without blocks; DeepInfer allocates when it prefills.
    # 14 blocks: 7 free + 6 evictable cannot fit, so it waits and the cache stays.
    d.add("X", prompt(14 * BS - 1, salt=4))
    for _ in range(3):
        d.apply()
    [x] = scheduler.waiting
    assert d.admissions["X"].admitted and not x.block_table
    assert old <= set(d.cache._blocks)
    d.apply(aborts=["X"])

    # 12 blocks: evict what it takes (all of A and B here), then prefill.
    d.add("Y", prompt(12 * BS - 1, salt=5))
    while scheduler.waiting:
        d.apply()
    assert not old & set(d.cache._blocks)
    d.close()


def test_queued_misses_above_the_prefill_watermark_all_finish(monkeypatch):
    # Two 3700-token misses take 29 blocks each: 59 of 64 (92%) if they were
    # allocated while waiting, which would keep DeepInfer's prefill gate shut.
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "64")
    d = Driver(make_runner(snapshot_interval=0))
    d.apply(
        adds=[
            AddReq("a", prompt_bytes(prompt(3700, salt=1)), {"max_tokens": 4}),
            AddReq("b", prompt_bytes(prompt(3700, salt=2)), {"max_tokens": 4}),
        ]
    )
    done: dict[str, str] = {}
    for _ in range(100):
        for o in d.apply().outputs:
            if o.finish_reason:
                done[o.request_id] = o.finish_reason
        if len(done) == 2:
            break
    assert done == {"a": "length", "b": "length"}
    d.close()


def test_hits_leave_headroom_for_running_decodes(monkeypatch):
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "40")
    d = Driver(make_runner())
    p = prompt(1000)
    d.run("warm", p)  # caches 6 blocks with a snapshot at 768
    d.add("R", prompt(1500, salt=1), max_tokens=600)
    scheduler = d.runner.engine.scheduler
    while not scheduler.running:
        d.apply()
    # 20 hits of 2 new blocks each arrive at once: more than the free blocks.
    d.apply(
        adds=[AddReq(f"h{i}", prompt_bytes(p), {"max_tokens": 4}) for i in range(20)]
    )
    finish: dict[str, str] = {}
    min_free = d.bm.get_num_free_blocks()
    for _ in range(2000):
        for o in d.apply().outputs:
            if o.finish_reason:
                finish[o.request_id] = o.finish_reason
        if scheduler.running:
            min_free = min(min_free, d.bm.get_num_free_blocks())
        if len(finish) == 21:
            break
    # Admission never drained the pool, so DeepInfer never preempted R.
    assert finish == {"R": "length", **{f"h{i}": "length" for i in range(20)}}
    assert min_free >= 1
    assert all(d.admissions[f"h{i}"].hit_tokens == 768 for i in range(20))
    d.close()


def test_hits_leave_state_slots_for_queued_misses():
    # max_num_seqs=2 gives int(2 * 1.2) + 1 = 3 state slots, 2 usable. A miss
    # gets its slot only at the head of DeepInfer's FCFS queue: the hits queued
    # behind it in the same iteration must not take both.
    d = Driver(make_runner(engine_type="PROMPT", max_num_seqs=2))
    p = prompt(1000)
    d.run("warm", p, max_tokens=1)  # snapshot at 768
    reply = d.apply(
        adds=[AddReq("m", prompt_bytes(prompt(300, salt=1)), {"max_tokens": 1})]
        + [AddReq(f"h{i}", prompt_bytes(p), {"max_tokens": 1}) for i in range(2)]
    )
    finish: dict[str, str] = {}
    for _ in range(100):
        for o in reply.outputs:
            if o.finish_reason:
                finish[o.request_id] = o.finish_reason
        if len(finish) == 3:
            break
        reply = d.apply()
    assert finish == {"m": "length", "h0": "length", "h1": "length"}
    assert d.admissions["h1"].hit_tokens == 768
    d.close()


def test_cached_blocks_count_as_free_in_stats():
    d = Driver(make_runner())
    d.run("a", prompt(600))
    assert len(d.cache._blocks) == 3
    assert d.cache.stats().used_blocks == 0
    d.close()


def test_snapshot_slots_recover_from_a_preempted_hit():
    d = Driver(make_runner(snapshot_slots=2))
    p = prompt(600)
    d.run("a", p)  # snapshots at 256 and 384 fill the pool
    assert d.cache._pool.num_free == 0
    d.add("b", p, max_tokens=50)
    assert d.admissions["b"].hit_tokens == 384
    scheduler = d.runner.engine.scheduler
    while not scheduler.running:
        d.apply()
    [seq] = scheduler.running
    # As Scheduler::_preempt: freed and stopped without finalize_seq.
    seq.status = type(seq.status).FINISHED_PREEMPTED
    scheduler.free_seq(seq)
    scheduler.running = []
    d.run("c", prompt(600, salt=1))
    c_tail = vllm_hash(None, prompt(600, salt=1)[:BS])
    assert c_tail in d.cache._blocks and not d.cache._requests
    d.close()


def test_finishing_requests_keep_the_free_floor(monkeypatch):
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "40")
    d = Driver(make_runner(free_watermark=0.25))
    assert d.cache.floor_blocks == 10
    for i in range(12):
        d.run(f"r{i}", prompt(600, salt=i))
        assert d.bm.get_num_free_blocks() >= d.cache.floor_blocks
    # 39 usable - 10 floor leaves room for 9 entries of 3 blocks.
    assert len(d.cache._blocks) <= 29
    check_quiescent(d)
    d.close()


def test_snapshot_slots_are_recycled_by_eviction():
    d = Driver(make_runner(snapshot_slots=16))
    for i in range(10):  # two snapshots each: the pool runs out at the 9th
        d.run(f"r{i}", prompt(600, salt=i))
    assert d.cache._pool.num_free == 0
    assert vllm_hash(None, prompt(600, salt=0)[:BS]) not in d.cache._blocks
    check_quiescent(d)
    d.run("again", prompt(600, salt=9))
    assert d.admissions["again"].hit_tokens == 384
    d.close()


def test_preempted_request_is_reaped():
    d = Driver(make_runner())
    p = prompt(600)
    d.run("a", p)
    d.add("b", p, max_tokens=50)
    scheduler = d.runner.engine.scheduler
    [req] = d.cache._requests.values()
    seq = req.seq
    # As Scheduler::_preempt with stop_preempted_request: freed, never finalized.
    seq.status = type(seq.status).FINISHED_PREEMPTED
    scheduler.free_seq(seq)
    scheduler.running = [s for s in scheduler.running if s is not seq]
    assert all(b.users == 1 for b in d.cache._blocks.values())
    total = d.runner.engine.cache_config.num_gpu_blocks
    assert d.cache.ensure_free(total - 1) == 3
    assert not d.cache._requests and not d.cache._blocks
    d.close()


def test_prefill_hold_keeps_cached_blocks_until_released():
    d = Driver(make_runner(engine_type="PROMPT"))
    p = prompt(600)
    d.add("a", p, max_tokens=1, hold_kv=True)
    for _ in range(10):
        if d.runner.prefill.num_held:
            break
        d.apply()
    [(held, _)] = d.runner.prefill._held.values()
    cached = cached_ids(d.cache)
    assert len(cached) == 3 and set(held.block_ids[:3]) == cached
    assert all(d.bm.ref_count(b) == 2 for b in cached)  # cache + hold
    assert all(b.users == 1 for b in d.cache._blocks.values())
    total = d.runner.engine.cache_config.num_gpu_blocks
    assert d.cache.ensure_free(total - 1) == 0  # held blocks are not evictable
    d.runner.prefill.close()
    check_quiescent(d)
    d.close()


# -- host tier ----------------------------------------------------------------

ROW = BS * 4  # the fake's KV block row, in bytes
SNAP = 32  # the fake's stored state, in bytes


def host_bytes(rows: int, snapshots: int = 16) -> int:
    """A host budget of ``rows`` block rows plus ``snapshots`` snapshots."""
    return rows * ROW + snapshots * SNAP


def check_tiers(d: Driver) -> None:
    """No live requests: GPU blocks pinned once and only below GPU blocks,
    host blocks in distinct host rows, child counts consistent."""
    cache, blocks = d.cache, list(d.cache._blocks.values())
    on_host = [b for b in blocks if not b.on_gpu]
    assert len({b.host_slot for b in on_host}) == len(on_host)
    assert all(b.host_slot >= 0 for b in on_host)
    assert cache._host.num_free == cache._host.capacity - len(on_host)
    for b in blocks:
        assert b.users == 0
        kids = [c for c in blocks if c.parent is b]
        assert b.children == len(kids)
        assert b.gpu_children == sum(c.on_gpu for c in kids)
        if b.on_gpu:
            assert d.bm.ref_count(b.block_id) == 1 and b.host_slot == -1
            assert b.parent is None or b.parent.on_gpu
    total = d.runner.engine.cache_config.num_gpu_blocks
    gpu = sum(b.on_gpu for b in blocks)
    assert d.bm.get_num_free_blocks() == total - 1 - gpu
    snapshots = sum(b.snapshot >= 0 for b in blocks)
    assert cache._pool.num_free == cache._pool.num_slots - snapshots


def chain(p: list[int], n: int = 3) -> list[bytes]:
    hashes = [vllm_hash(None, p[:BS])]
    for i in range(1, n):
        hashes.append(vllm_hash(hashes[-1], p[i * BS : (i + 1) * BS]))
    return hashes


def test_host_tier_round_trip_keeps_the_bytes_and_the_tokens():
    d = Driver(make_runner(host_bytes=host_bytes(8)))
    cold = Driver(make_runner("none"))
    assert d.cache._host.capacity == 8 and d.cache._pool._host
    p = prompt(600)
    first = d.run("a", p)
    kv = d.runner.engine.driver_worker.cache_engine.gpu_blocks
    blocks = dict(d.cache._blocks)
    before = {h: kv[b.block_id].copy() for h, b in blocks.items()}
    d.events.clear()

    # Out of GPU memory, onto host: still a hit, so no BlockRemoved.
    total = d.runner.engine.cache_config.num_gpu_blocks
    assert d.cache.ensure_free(total - 1) == 3
    d.cache.take_events()
    assert not d.events
    assert not any(b.on_gpu for b in blocks.values())
    for h, b in blocks.items():
        assert (d.cache._host._rows.row(b.host_slot) == before[h]).all()
    check_tiers(d)

    kv[:] = 0  # every GPU row is free now: only the host has the KV
    assert d.run("b", p) == first == cold.run("c", p)
    assert d.admissions["b"].hit_tokens == 384
    assert d.cache.host_hit_tokens == 384
    assert (d.cache.host_stores, d.cache.host_loads) == (3, 3)
    for h, b in d.cache._blocks.items():
        assert b.on_gpu and (kv[b.block_id] == before[h]).all()
    check_tiers(d)
    d.close()
    cold.close()


def test_host_hit_after_gpu_eviction(monkeypatch):
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", "16")
    d = Driver(make_runner(host_bytes=host_bytes(32)))
    cold = Driver(make_runner("none"))
    prompts = [prompt(600, salt=i) for i in range(5)]
    tokens = [d.run(f"r{i}", p) for i, p in enumerate(prompts)]
    # The GPU (15 blocks, 2 free for the floor, 5 for a running prompt) keeps
    # the newest prompts: LRU demoted all of the first one and the leaf of the
    # second, and dropped nothing.
    on_gpu = [[d.cache._blocks[h].on_gpu for h in chain(p)] for p in prompts]
    assert on_gpu[:2] == [[False] * 3, [True, True, False]]
    assert not any(e.kind == "removed" for e in d.events)
    check_tiers(d)
    # The second prompt: two shared GPU blocks plus one from host.
    assert d.run("again1", prompts[1]) == tokens[1] == cold.run("c1", prompts[1])
    assert d.admissions["again1"].hit_tokens == 384
    assert d.cache.host_hit_tokens == 128
    # The first: all three from host.
    assert d.run("again0", prompts[0]) == tokens[0] == cold.run("c0", prompts[0])
    assert d.admissions["again0"].hit_tokens == 384
    assert d.cache.host_hit_tokens == 128 + 384
    check_tiers(d)
    d.close()
    cold.close()


def test_host_tier_evicts_lru_leaves_and_reports_them_removed():
    d = Driver(make_runner(host_bytes=host_bytes(4)))
    cold = Driver(make_runner("none"))
    total = d.runner.engine.cache_config.num_gpu_blocks
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    d.cache.ensure_free(total - 1)  # a's 3 blocks to host
    expected = d.run("b", b)
    d.events.clear()
    d.cache.ensure_free(total - 1)  # b's 3 blocks need a's two leaves' rows
    d.cache.take_events()
    ha, hb = chain(a), chain(b)
    assert [e.kind for e in d.events] == ["removed"]
    assert d.events[0].block_hashes == [as_u64(ha[2]), as_u64(ha[1])]
    assert set(d.cache._blocks) == {ha[0], *hb}
    assert d.cache.host_evictions == 2
    check_tiers(d)
    # a's root alone has no snapshot to resume from; b is a full host hit.
    assert d.run("b2", b) == expected == cold.run("c", b)
    assert d.admissions["b2"].hit_tokens == 384
    d.run("a2", a)
    assert d.admissions["a2"].hit_tokens == 0
    d.close()
    cold.close()


def test_host_tier_snapshots_on_host_and_clamped_rows(monkeypatch):
    from dynamo.deepinfer.kv_cache import host

    # 2 rows per pinned chunk; the host runs short after the snapshot chunk
    # and two row chunks.
    monkeypatch.setattr(host, "CHUNK_BYTES", 2 * ROW)
    calls = iter(range(100))
    monkeypatch.setattr(
        host,
        "mem_available",
        lambda: (1 << 40 if next(calls) < 3 else 1 << 30, 1 << 40),
    )
    d = Driver(make_runner(host_bytes=16384))
    cache = d.cache
    # Snapshots: a 16th of the budget (more than the 16 configured), on host;
    # the rest is 30 rows, clamped to 4.
    assert cache._pool._host and cache._pool.num_slots == 16384 // 16 // SNAP == 32
    assert cache._host.capacity == 4
    cold = Driver(make_runner("none"))
    p = prompt(600)
    first = d.run("a", p)
    d.cache.ensure_free(d.runner.engine.cache_config.num_gpu_blocks - 1)
    slots = sorted(b.host_slot for b in cache._blocks.values())
    assert len(slots) == 3 and slots[-1] >= 2  # rows in both chunks
    assert d.run("b", p) == first == cold.run("c", p)
    assert d.admissions["b"].hit_tokens == 384
    check_tiers(d)
    d.close()
    cold.close()


# -- SSD tier -----------------------------------------------------------------

SSD_STRIDE = 4096  # the fake's 512-byte rows, padded to the SSD's alignment


class ManualIO:
    """Stands in for the SSD pool's threads: runs I/O inline, or holds it until
    released. Completions are applied by the cache's next ``before_step``."""

    def __init__(self) -> None:
        self.hold = False
        self.jobs: list[Any] = []

    def submit(self, fn: Any, *args: Any) -> None:
        if self.hold:
            self.jobs.append((fn, args))
        else:
            fn(*args)

    def release(self) -> None:
        jobs, self.jobs = self.jobs, []
        for fn, args in jobs:
            fn(*args)

    def shutdown(self, wait: bool = True) -> None:
        self.release()


@pytest.fixture
def io(monkeypatch):
    from dynamo.deepinfer.kv_cache import prefix

    manual = ManualIO()

    class Pool(prefix.SSDPool):
        def __init__(self, *args: Any, **kw: Any) -> None:
            super().__init__(*args, **kw)
            self._executor.shutdown()
            self._executor = manual

    monkeypatch.setattr(prefix, "SSDPool", Pool)
    monkeypatch.setattr(prefix, "SSD_MIN_READ_BLOCKS", 1)
    return manual


def ssd_runner(tmp_path, rows: int, ssd_blocks: int = 32, **options: Any):
    return make_runner(
        host_bytes=host_bytes(rows),
        ssd_dir=str(tmp_path),
        ssd_bytes=ssd_blocks * SSD_STRIDE,
        **options,
    )


def one_block(i: int) -> list[int]:
    """A prompt that caches one block, with a snapshot after it."""
    return prompt(300, salt=i)


def evict_all(d: Driver) -> None:
    d.cache.ensure_free(d.runner.engine.cache_config.num_gpu_blocks - 1)
    d.cache.take_events()


def spy_io(d: Driver) -> list[tuple[str, list[int]]]:
    """The SSD stores and loads the cache submits, as (kind, slots)."""
    calls: list[tuple[str, list[int]]] = []
    pool = d.cache._ssd
    store, load = pool.store, pool.load

    def spy_store(rows, slots, *args, **kw):
        calls.append(("store", list(slots)))
        return store(rows, slots, *args, **kw)

    def spy_load(slots, rows, *args, **kw):
        calls.append(("load", list(slots)))
        return load(slots, rows, *args, **kw)

    pool.store, pool.load = spy_store, spy_load
    return calls


def fail_io(d: Driver, kind: str) -> None:
    """Make the pool's reads or writes fail with an OSError."""
    pool = d.cache._ssd
    transfer = pool._transfer

    def failing(slot, view, *, write):
        if write == (kind == "store"):
            raise OSError("injected")
        transfer(slot, view, write=write)

    pool._transfer = failing


def file_row(d: Driver, slot: int) -> bytes:
    pool = d.cache._ssd
    with open(pool.path, "rb") as f:
        f.seek(slot * pool.stride)
        return f.read(pool.row_bytes)


def _external(block: Any) -> int:
    from dynamo.deepinfer.kv_cache.prefix import _external

    return _external(block)


def check_ssd(d: Driver) -> None:
    """Quiescent (no I/O in flight): tiers only go down the chain, child counts,
    host rows and SSD slots add up, a block is in one place, every evictable
    block is queued in its heap and the events match the cached blocks."""
    cache, blocks = d.cache, list(d.cache._blocks.values())
    cache.take_events()
    assert not d.event_errors and d.live == {_external(b) for b in blocks}
    queued = {
        name: {(k, id(b)) for k, _, b in getattr(cache, name)}
        for name in ("_heap", "_host_heap", "_ssd_heap", "_ssd_snap_heap")
    }
    for b in blocks:
        if b.tier.name == "GPU" and not b.gpu_children:
            assert (b.heap_key, id(b)) in queued["_heap"]
        if b.tier.name == "HOST" and not (b.gpu_children or b.host_children):
            # Nothing above it needs the row: it must be reclaimable.
            assert cache._host_leaf(b)
            assert (b.heap_key, id(b)) in queued["_host_heap"]
        if b.ssd_state.name == "DONE" and cache._ssd_victim(b):
            assert (b.ssd_key, id(b)) in queued["_ssd_heap"]
        if b.tier.name == "SSD" and not b.children and b.snapshot >= 0:
            assert (b.snap_key, id(b)) in queued["_ssd_snap_heap"]
    on_host = [b for b in blocks if b.tier.name == "HOST"]
    assert len({b.host_slot for b in on_host}) == len(on_host)
    assert all(b.host_slot >= 0 for b in on_host)
    assert cache._host.num_free == cache._host.capacity - len(on_host)
    slots = [b.ssd_slot for b in blocks if b.ssd_slot >= 0]
    assert len(set(slots)) == len(slots)
    assert cache._ssd.num_free == cache._ssd.capacity - len(slots)
    assert cache._num_ssd == sum(b.tier.name == "SSD" for b in blocks)
    assert cache._reading_rows == sum(b.reading for b in blocks) and (
        cache._reads_inflight == 0
    )
    assert cache._num_host == len(on_host)
    for b in blocks:
        assert b.users == 0 and not b.reading and b.write is None
        assert b.ssd_state.name == ("DONE" if b.ssd_slot >= 0 else "NONE")
        kids = [c for c in blocks if c.parent is b]
        assert b.children == len(kids) == len(b.kids())
        assert {id(k) for k in b.kids()} == {id(c) for c in kids}
        assert (b.kid is None) == (not kids) and (b.more_kids is None) == (
            len(kids) < 2
        )
        assert b.gpu_children == sum(c.tier.name == "GPU" for c in kids)
        assert b.host_children == sum(c.tier.name == "HOST" for c in kids)
        assert all(c.tier >= b.tier for c in kids)
        assert (b.block_id >= 0) == (b.tier.name == "GPU")
        assert (b.host_slot >= 0) == (b.tier.name == "HOST")
        if b.tier.name == "SSD":
            assert b.ssd_slot >= 0
        if b.on_gpu:
            assert d.bm.ref_count(b.block_id) == 1
    total = d.runner.engine.cache_config.num_gpu_blocks
    assert d.bm.get_num_free_blocks() == total - 1 - sum(b.on_gpu for b in blocks)
    snapshots = sum(b.snapshot >= 0 for b in blocks)
    assert cache._pool.num_free == cache._pool.num_slots - snapshots


def removed(d: Driver) -> list[int]:
    return [h for e in d.events if e.kind == "removed" for h in e.block_hashes]


def test_host_eviction_of_a_written_block_moves_it_to_ssd_without_io(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=4))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()  # the writes completed: a's copies are complete
    ha, hb = chain(a), chain(b)
    blocks = [d.cache._blocks[h] for h in ha]
    assert [x.ssd_state.name for x in blocks] == ["DONE"] * 3
    assert d.cache.ssd_stores == 3 and d.cache._ssd.num_free == 29
    d.run("b", b)
    d.events.clear()
    calls = spy_io(d)
    evict_all(d)  # b needs 3 host rows: a's two leaves give theirs
    assert [x.tier.name for x in blocks] == ["HOST", "SSD", "SSD"]
    assert blocks[2].snapshot >= 0 and blocks[1].snapshot >= 0
    # Only b's own copies are written; a's move was not I/O, and not an event.
    assert [kind for kind, _ in calls] == ["store"] * 3 and not removed(d)
    assert d.cache.host_evictions == 2 and d.cache._num_ssd == 2
    d.apply()
    assert d.cache.ssd_stores == 6
    check_ssd(d)
    assert all(h in d.cache._blocks for h in hb)
    d.close()


def test_ssd_hit_warms_host_and_the_next_request_hits_from_host(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_wait_ms=0))
    cold = Driver(make_runner("none"))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    first = d.run("a", a)
    kv = d.runner.engine.driver_worker.cache_engine.gpu_blocks
    before = {h: kv[x.block_id].copy() for h, x in d.cache._blocks.items()}
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)  # a is on SSD now, b on host
    d.apply()
    blocks = [d.cache._blocks[h] for h in chain(a)]
    assert [x.tier.name for x in blocks] == ["SSD"] * 3
    io.hold = True
    # The first request is a miss and starts the read; it is not delayed.
    d.add("x", a, max_tokens=50)
    assert d.admissions["x"].admitted and d.admissions["x"].hit_tokens == 0
    assert all(x.reading for x in blocks) and d.cache.ssd_loads == 0
    d.apply(aborts=["x"])
    io.hold = False
    io.release()
    d.apply()
    assert [x.tier.name for x in blocks] == ["HOST"] * 3
    assert d.cache.ssd_loads == 3 and all(x.ssd_warm for x in blocks)
    for h, x in zip(chain(a), blocks):
        assert (d.cache._host.row(x.host_slot) == before[h]).all()
    check_ssd(d)
    kv[:] = 0
    assert d.run("a2", a) == first == cold.run("c", a)
    assert d.admissions["a2"].hit_tokens == 384
    assert d.cache.host_hit_tokens == 384 and d.cache.ssd_hit_tokens == 384
    for h, x in d.cache._blocks.items():
        if h in before:
            assert x.on_gpu and (kv[x.block_id] == before[h]).all()
    d.close()
    cold.close()


def test_small_ssd_hits_are_not_read(tmp_path, io, monkeypatch):
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MIN_READ_BLOCKS", 4)
    d = Driver(ssd_runner(tmp_path, rows=3))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    calls = spy_io(d)
    d.run("a2", a)
    assert d.admissions["a2"].hit_tokens == 0 and not calls
    d.close()


def test_promotion_while_the_write_is_in_flight_keeps_the_original_bytes(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    a, c = prompt(600, salt=1), prompt(600, salt=3)
    d.run("a", a)
    d.run("c", c)
    kv = d.runner.engine.driver_worker.cache_engine.gpu_blocks
    blocks = [d.cache._blocks[h] for h in chain(a)]
    before = [kv[x.block_id].copy() for x in blocks]
    io.hold = True
    d.cache.ensure_free(d.bm.get_num_free_blocks() + 3)  # a (the coldest) to host
    assert [x.ssd_state.name for x in blocks] == ["WRITING"] * 3
    assert d.cache._host.num_free == 0
    d.run("a2", a)  # hit from host: the blocks go back to the GPU
    assert d.admissions["a2"].hit_tokens == 384
    assert all(x.on_gpu and x.ssd_state.name == "WRITING" for x in blocks)
    # Their host rows are not free: the write has not read them yet.
    assert d.cache._host.num_free == 0
    d.events.clear()
    d.cache.ensure_free(d.bm.get_num_free_blocks() + 3)  # c has no row to take
    d.cache.take_events()
    assert len(removed(d)) == 3 and d.cache._host.num_free == 0
    assert all(x.on_gpu for x in blocks)
    io.hold = False
    io.release()
    d.apply()
    assert d.cache._host.num_free == 3  # the rows came back with the writes
    assert [x.ssd_state.name for x in blocks] == ["DONE"] * 3
    for x, want in zip(blocks, before):
        assert file_row(d, x.ssd_slot) == want.tobytes()
    check_ssd(d)
    d.close()


def test_writes_over_the_cap_wait_their_turn_unless_evicted_first(
    tmp_path, io, monkeypatch
):
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MAX_INFLIGHT_WRITES", 1)
    d = Driver(ssd_runner(tmp_path, rows=3))
    for i in (1, 2, 3, 4):
        d.run(f"r{i}", one_block(i))
    h1, h2, h3, h4 = (chain(one_block(i), 1)[0] for i in (1, 2, 3, 4))
    io.hold = True
    d.events.clear()
    evict_all(d)
    blocks = d.cache._blocks
    # r4's demotion took the row of r1 (queued, unwritten, unpinned) before its
    # turn; r2 is written, and r3 and r4 wait.
    assert h1 not in blocks and removed(d) == [as_u64(h1)]
    assert [blocks[h].ssd_state.name for h in (h2, h3, h4)] == [
        "WRITING",
        "NONE",
        "NONE",
    ]
    assert len(d.cache._pending_writes) == 2  # r1's stale entry was skipped
    io.hold = False
    io.release()
    for _ in range(4):
        d.apply()
    assert [blocks[h].ssd_state.name for h in (h2, h3, h4)] == ["DONE"] * 3
    assert not d.cache._pending_writes and d.cache.ssd_stores == 3
    check_ssd(d)
    d.close()


def test_a_drained_write_waits_for_the_copies_queued_before_it(
    tmp_path, io, monkeypatch
):
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MAX_INFLIGHT_WRITES", 1)
    d = Driver(ssd_runner(tmp_path, rows=3))
    for i in (1, 2, 3):
        d.run(f"r{i}", one_block(i))
    host, ssd = d.cache._host, d.cache._ssd
    store, ssd_store = host.store, ssd.store
    log: list[tuple[Any, ...]] = []
    fence_event = object()

    def spy_store(*args, **kw):
        log.append(("copy", event := store(*args, **kw)))
        return event

    def spy_ssd_store(rows, slots, after=None, tag=None):
        log.append(("write", after))
        return ssd_store(rows, slots, after=after, tag=tag)

    host.store, ssd.store = spy_store, spy_ssd_store
    host.fence = lambda: log.append(("fence", fence_event)) or fence_event
    io.hold = True
    evict_all(d)
    # The first write rides on the copy made in its own pass.
    seen = len(log)
    io.hold = False
    io.release()
    d.apply()
    # The next write is started by a poll, with no copy of its own: it waits for
    # a fence taken after the copies queued so far.
    assert [e[0] for e in log[:seen]].index("copy") < [e[0] for e in log[:seen]].index(
        "write"
    )
    assert [e[0] for e in log[seen:]] == ["fence", "write"]
    assert log[seen + 1][1] is fence_event
    for _ in range(3):
        d.apply()
    check_ssd(d)
    d.close()


def test_a_burst_over_the_cap_is_written_root_first_within_the_cap(
    tmp_path, io, monkeypatch
):
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MAX_INFLIGHT_WRITES", 3)
    d = Driver(ssd_runner(tmp_path, rows=12))
    p = prompt(1800, salt=1)
    d.run("a", p)
    blocks = [d.cache._blocks[h] for h in chain(p, 12)]
    calls = spy_io(d)
    io.hold = True
    evict_all(d)
    # Ancestors go first: the root three are written, the other nine wait.
    assert [b.ssd_state.name for b in blocks] == ["WRITING"] * 3 + ["NONE"] * 9
    assert len(d.cache._pending_writes) == 9 and d.cache._ssd.inflight_writes == 3
    for _ in range(6):
        io.release()
        d.apply()
        assert d.cache._ssd.inflight_writes <= 3
    io.hold = False
    io.release()
    d.apply()
    assert [b.ssd_state.name for b in blocks] == ["DONE"] * 12
    by_slot = {b.ssd_slot: b.index for b in blocks}
    assert [by_slot[slot[0]] for _, slot in calls] == list(range(12))
    check_ssd(d)
    d.close()


def test_pending_writes_keep_every_live_entry_and_prune_only_stale_ones(
    tmp_path, io, monkeypatch
):
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MAX_INFLIGHT_WRITES", 1)
    d = Driver(ssd_runner(tmp_path, rows=8))
    p = prompt(1200, salt=1)
    d.run("a", p)
    io.hold = True
    evict_all(d)
    # The root is written; all seven others wait, however many the cap allows.
    assert sorted(e[0] for e in d.cache._pending_writes) == [1, 2, 3, 4, 5, 6, 7]
    # Past the prune threshold the stale entries go and the live ones stay.
    block = d.cache._pending_writes[0][2]
    for _ in range(1100):
        d.cache._queue_write(block, row=-1)
    d.cache._drain_writes(prefix._Evicted())
    assert sorted(e[0] for e in d.cache._pending_writes) == [1, 2, 3, 4, 5, 6, 7]
    io.hold = False
    io.release()
    for _ in range(12):
        d.apply()
    check_ssd(d)
    d.close()


def test_full_ssd_clears_backup_copies_silently(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=2))
    p1, p2, p3 = one_block(1), one_block(2), one_block(3)
    d.run("r1", p1)
    d.run("r2", p2)
    evict_all(d)
    d.apply()
    h1, h2, h3 = (chain(p, 1)[0] for p in (p1, p2, p3))
    assert d.cache._ssd.num_free == 0
    d.run("r3", p3)
    d.events.clear()
    evict_all(d)
    d.apply()
    # r1's backup (the coldest) gave its slot to r3; r1 is still cached on host.
    b1, b3 = d.cache._blocks[h1], d.cache._blocks[h3]
    assert (b1.tier.name, b1.ssd_state.name) == ("HOST", "NONE")
    assert (b3.tier.name, b3.ssd_state.name) == ("HOST", "DONE")
    assert not removed(d) and d.cache.ssd_evictions == 1
    assert d.cache._blocks[h2].ssd_state.name == "DONE"
    check_ssd(d)
    d.close()


def test_full_ssd_drops_ssd_only_leaves_with_a_removed_event(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=2, ssd_blocks=2))
    p1, p2, p3 = one_block(1), one_block(2), one_block(3)
    d.run("r1", p1)
    d.run("r2", p2)
    evict_all(d)
    d.apply()
    h1, h2, h3 = (chain(p, 1)[0] for p in (p1, p2, p3))
    d.run("r3", p3)
    d.events.clear()
    evict_all(d)
    # r3 takes r1's host row (r1 keeps its SSD copy), then r1's slot: r1 goes.
    assert h1 not in d.cache._blocks and removed(d) == [as_u64(h1)]
    d.apply()
    assert d.cache._blocks[h3].ssd_state.name == "DONE"
    assert d.cache._blocks[h2].tier.name == "HOST"
    assert d.cache.ssd_evictions == 1 and d.cache._pool.num_free == 14
    check_ssd(d)
    d.close()


def test_snapshot_pool_pressure_drops_ssd_only_snapshots_first(tmp_path, io):
    d = Driver(
        make_runner(
            host_bytes=host_bytes(2, snapshots=0),
            ssd_dir=str(tmp_path),
            ssd_bytes=6 * SSD_STRIDE,
            snapshot_slots=3,
            snapshot_device="gpu",
        )
    )
    ps = [one_block(i) for i in (1, 2, 3)]
    for i, p in enumerate(ps):
        d.run(f"r{i}", p)
    h1, h2, h3 = (chain(p, 1)[0] for p in ps)
    assert d.cache._pool.num_free == 0
    d.cache.ensure_free(d.bm.get_num_free_blocks() + 2)  # r1 and r2 to host
    d.cache.take_events()
    d.apply()
    d.cache.ensure_free(d.bm.get_num_free_blocks() + 1)  # r3 pushes r1 to SSD
    d.cache.take_events()
    assert d.cache._blocks[h1].tier.name == "SSD"
    assert d.cache._blocks[h2].tier.name == d.cache._blocks[h3].tier.name == "HOST"
    d.events.clear()
    d.apply()
    d.run("r4", one_block(4))  # needs a snapshot slot
    # The SSD-only snapshot went; host blocks were not touched on its behalf.
    assert h1 not in d.cache._blocks and removed(d) == [as_u64(h1)]
    assert d.cache._blocks[h2].tier.name == d.cache._blocks[h3].tier.name == "HOST"
    assert d.cache.host_evictions == 1
    d.apply()
    check_ssd(d)
    d.close()


def test_gpu_leaf_with_a_complete_copy_goes_to_ssd_when_host_is_full(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=1))
    p, q = one_block(1), one_block(2)
    hp, hq = (chain(x, 1)[0] for x in (p, q))
    d.run("p", p)
    evict_all(d)
    d.apply()  # p: on host, copy complete
    d.run("p2", p)  # back on the GPU, keeping the copy
    assert d.cache._blocks[hp].on_gpu and d.cache._blocks[hp].ssd_state.name == "DONE"
    d.run("q", q)
    d.run("p3", p)  # p is the more recent
    d.events.clear()
    calls = spy_io(d)
    io.hold = True
    d.cache.ensure_free(d.bm.get_num_free_blocks() + 1)  # q (the coldest) to host
    evict_all(d)
    # q holds the only host row (its write pins it); p has nowhere but the SSD.
    assert d.cache._blocks[hq].tier.name == "HOST"
    assert d.cache._blocks[hp].tier.name == "SSD"
    assert d.cache._blocks[hp].snapshot >= 0 and not removed(d)
    assert [kind for kind, _ in calls] == ["store"]  # q's write only
    io.hold = False
    io.release()
    d.apply()
    check_ssd(d)
    d.close()


def test_read_error_is_a_miss_with_removed_events_and_the_engine_goes_on(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    cold = Driver(make_runner("none"))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    ha = chain(a)
    assert all(d.cache._blocks[h].tier.name == "SSD" for h in ha)
    free = d.cache._ssd.num_free
    fail_io(d, "load")
    d.events.clear()
    assert d.run("a2", a) == cold.run("c", a)  # recomputed, correctly
    assert d.admissions["a2"].hit_tokens == 0
    assert d.cache.ssd_errors == 1 and d.cache.ssd_loads == 0
    # The failed blocks were reported removed, leaf first, and cached afresh.
    assert removed(d)[:3] == [as_u64(h) for h in ha[::-1]]
    assert all(d.cache._blocks[h].on_gpu for h in ha)
    assert d.cache._ssd.num_free == free + 3
    d.apply()
    assert d.run("b2", b) == cold.run("c2", b)
    check_ssd(d)
    d.close()
    cold.close()


def test_write_error_leaves_the_block_unwritten_and_cached(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=4))
    cold = Driver(make_runner("none"))
    p = prompt(600)
    first = d.run("a", p)
    fail_io(d, "store")
    evict_all(d)
    d.apply()
    blocks = [d.cache._blocks[h] for h in chain(p)]
    assert [x.ssd_state.name for x in blocks] == ["NONE"] * 3
    assert all(x.tier.name == "HOST" and x.ssd_slot == -1 for x in blocks)
    assert d.cache.ssd_errors == 3 and d.cache.ssd_stores == 0
    assert d.cache._ssd.num_free == d.cache._ssd.capacity
    assert not removed(d)
    check_ssd(d)
    assert d.run("b", p) == first == cold.run("c", p)  # still a host hit
    assert d.admissions["b"].hit_tokens == 384
    d.close()
    cold.close()


def test_failed_read_keeps_blocks_recomputed_meanwhile(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_wait_ms=0))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    blocks = [d.cache._blocks[h] for h in chain(a)]
    assert all(x.tier.name == "SSD" for x in blocks)
    io.hold = True
    fail_io(d, "load")
    # A miss: it starts the read, recomputes a and caches its blocks on the GPU.
    d.add("x", a, max_tokens=50)
    for _ in range(5):
        d.apply()
    assert all(x.on_gpu and x.reading and x.users == 1 for x in blocks)
    d.events.clear()
    io.hold = False
    io.release()
    d.apply()
    assert d.cache.ssd_errors == 1 and d.cache.ssd_loads == 0
    # The read failed, but the blocks are not on SSD any more: they stay.
    assert all(d.cache._blocks.get(h) is x for h, x in zip(chain(a), blocks))
    assert all(x.on_gpu and not x.reading and x.users == 1 for x in blocks)
    assert not removed(d) and d.cache._host.num_free == 3
    d.apply(aborts=["x"])
    d.apply()
    check_ssd(d)
    d.close()


def test_ssd_read_waits_for_copies_that_still_use_its_rows(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=8, ssd_blocks=64, ssd_wait_ms=0))
    a, b = prompt(1200, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    assert [d.cache._blocks[h].tier.name for h in chain(a, 8)] == (
        ["HOST"] * 5 + ["SSD"] * 3
    )
    log: list[tuple[Any, ...]] = []

    class Event:  # what HostBlockPool.fence() returns on the GPU
        def synchronize(self):
            log.append(("sync",))

    host, ssd = d.cache._host, d.cache._ssd
    h2d, load = host.load, ssd.load

    def spy_h2d(slots, ids):
        log.append(("h2d", set(slots)))
        return h2d(slots, ids)

    def spy_load(slots, rows, after=None, tag=None):
        log.append(("read", set(tag.rows), after))
        return load(slots, rows, after=after, tag=tag)

    event = Event()
    host.load, ssd.load = spy_h2d, spy_load
    host.fence = lambda: log.append(("fence",)) or event
    io.hold = True
    d.add("a2", a, max_tokens=4)
    assert d.admissions["a2"].hit_tokens == 512
    # The promotion freed rows that its queued copy still reads, and the read
    # was given them: it waits for everything queued up to now.
    [(_, copied), (_,), (_, rows, after)] = log
    assert copied & rows and after is event
    io.hold = False
    io.release()
    assert log[-1] == ("sync",)
    d.apply()
    assert d.cache.ssd_loads == 3 and d.cache.ssd_errors == 0
    d.apply(aborts=["a2"])
    d.close()


def test_read_skips_blocks_evicted_since_the_lookup(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    run = [d.cache._blocks[h] for h in chain(a)]
    assert all(x.tier.name == "SSD" for x in run)
    calls = spy_io(d)
    # What admit's _make_room does to an SSD leaf when it needs a slot.
    from dynamo.deepinfer.kv_cache.prefix import _Evicted

    out = _Evicted()
    d.cache._evict_ssd(run[2], out)
    d.cache._flush(out, [])
    assert run[2].ssd_slot == -1
    d.cache._read_ssd(run)
    # The first two blocks end at a snapshot: they are read, the dead one not.
    assert calls == [("load", [run[0].ssd_slot, run[1].ssd_slot])]
    d.apply()
    assert d.cache.ssd_errors == 0 and d.cache.ssd_loads == 2
    assert [x.tier.name for x in run[:2]] == ["HOST"] * 2
    check_ssd(d)
    d.close()


def test_blocks_being_read_keep_their_slots(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_wait_ms=0))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    blocks = [d.cache._blocks[h] for h in chain(a)]
    io.hold = True
    d.add("x", a, max_tokens=50)
    for _ in range(5):
        d.apply()
    # Recomputed into the GPU, with the read still in flight: the backup's
    # slot may not be reclaimed (and handed to another block) until it is done.
    assert all(x.on_gpu and x.reading and x.ssd_slot >= 0 for x in blocks)
    assert not any(d.cache._ssd_victim(x) for x in blocks)
    io.hold = False
    io.release()
    d.apply()
    assert all(d.cache._ssd_victim(x) for x in blocks)
    d.apply(aborts=["x"])
    d.apply()
    check_ssd(d)
    d.close()


def test_read_is_dropped_when_the_slot_changed_under_it(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    blocks = [d.cache._blocks[h] for h in chain(a)]
    io.hold = True
    d.add("x", a, max_tokens=2)
    assert all(x.reading for x in blocks)
    d.apply(aborts=["x"])
    slot, blocks[2].ssd_slot = blocks[2].ssd_slot, 31  # as if it were reused
    io.hold = False
    io.release()
    d.apply()
    # The last block's row is not trusted; the first two are fine.
    assert [x.tier.name for x in blocks] == ["HOST", "HOST", "SSD"]
    assert d.cache._host.num_free == 1 and d.cache.ssd_loads == 2
    blocks[2].ssd_slot = slot
    check_ssd(d)
    d.close()


def test_unwritten_parent_of_an_ssd_block_does_not_pin_its_host_row(tmp_path, io):
    from dynamo.deepinfer.kv_cache.prefix import _Evicted

    d = Driver(ssd_runner(tmp_path, rows=2))
    p = prompt(520, salt=1)
    d.run("p", p)
    evict_all(d)
    for _ in range(3):
        d.apply()
    parent, child = (d.cache._blocks[h] for h in chain(p, 2))
    # A full SSD reclaimed the parent's backup (the coldest) but not its child's.
    d.cache._evict_ssd(parent, _Evicted())
    assert (parent.tier.name, parent.ssd_state.name) == ("HOST", "NONE")
    assert (child.tier.name, child.ssd_state.name) == ("HOST", "DONE")
    for i in range(3, 6):
        d.run(f"q{i}", one_block(i))
        evict_all(d)
        d.apply()
    # Host pressure took the parent's row, and the child it could not keep
    # reachable went along: both are reported removed, leaf first.
    assert child.block_hash not in d.cache._blocks
    assert parent.block_hash not in d.cache._blocks
    ids = removed(d)
    assert ids.index(_external(child)) < ids.index(_external(parent))
    assert d.cache._host.num_free == 0 and len(ids) == len(set(ids))
    check_ssd(d)
    d.close()


def test_ssd_backup_with_an_ssd_child_is_not_reclaimed(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    p = prompt(520, salt=1)
    d.run("p", p)
    evict_all(d)
    d.apply()
    d.run("q", one_block(2))
    evict_all(d)
    d.apply()
    d.run("r", one_block(3))
    evict_all(d)
    d.apply()
    parent, child = (d.cache._blocks[h] for h in chain(p, 2))
    # The child went to SSD under host pressure; its parent is still on host.
    assert (parent.tier.name, child.tier.name) == ("HOST", "SSD")
    assert parent.ssd_state.name == "DONE" and parent.host_children == 0
    assert not d.cache._ssd_victim(parent) and d.cache._ssd_victim(child)
    check_ssd(d)
    d.close()


def test_read_that_gets_fewer_rows_is_cut_back_to_a_snapshot(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=64))
    a = prompt(1200, salt=1)
    d.run("a", a)
    for _ in range(8):  # one leaf at a time, each written before the next
        d.cache.ensure_free(d.bm.get_num_free_blocks() + 1)
        d.apply()
        d.apply()
    run = [d.cache._blocks[h] for h in chain(a, 8)]
    assert [x.tier.name for x in run] == ["HOST"] * 3 + ["SSD"] * 5
    calls = spy_io(d)
    before = d.cache.host_evictions
    d.events.clear()
    d.add("a2", a, max_tokens=4)
    # 256 tokens hit from host (two rows free up); the gap block behind them
    # stays. Two rows are taken for the five SSD blocks, which covers the
    # first and the gap behind it: the read is cut back to the first (the
    # deepest snapshot the rows cover) and the second row goes back.
    assert d.admissions["a2"].hit_tokens == 256
    assert calls == [("load", [run[3].ssd_slot])]
    # Nothing was evictable, so nothing was evicted for the rows.
    assert d.cache.host_evictions == before and not removed(d)
    assert [x.tier.name for x in run[:4]] == ["GPU", "GPU", "HOST", "HOST"]
    host = d.cache._host
    # Held rows: the gap block and the one being read; the third is free.
    assert host.num_free == 1 and run[4].reading is False
    d.apply(aborts=["a2"])
    d.apply()
    # Quiescent: no row leaked (check_ssd adds them up).
    check_ssd(d)
    d.close()


def test_read_that_cannot_fit_evicts_nothing(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=64, snapshot_interval=1024))
    a = prompt(1200, salt=1)
    d.run("a", a)
    for _ in range(10):  # a's blocks go down to SSD, one leaf at a time
        d.cache.ensure_free(d.bm.get_num_free_blocks() + 1)
        d.apply()
        d.apply()
    run = [d.cache._blocks[h] for h in chain(a, 8)]
    for i in range(3):  # fill the host with unrelated blocks
        d.run(f"q{i}", one_block(10 + i))
    evict_all(d)
    d.apply()
    d.apply()
    d.apply()
    assert [x.tier.name for x in run] == ["SSD"] * 8
    assert [x.snapshot >= 0 for x in run] == [False] * 7 + [True]
    host = d.cache._host
    assert host.num_free == 0
    calls = spy_io(d)
    before, free = d.cache.host_evictions, host.num_free
    held = {b.host_slot for b in d.cache._blocks.values() if b.host_slot >= 0}
    d.events.clear()
    d.add("a2", a, max_tokens=4)
    # Eight rows are needed to reach the only snapshot; the host has three.
    assert calls == [] and d.cache.host_evictions == before
    assert host.num_free == free and not removed(d)
    assert held == {b.host_slot for b in d.cache._blocks.values() if b.host_slot >= 0}
    assert not any(x.reading for x in run)
    d.apply(aborts=["a2"])
    d.apply()
    check_ssd(d)
    d.close()


def test_a_long_run_whose_read_cannot_start_is_admitted_at_once(
    tmp_path, io, monkeypatch
):
    wait_for_ssd(monkeypatch)
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=64, snapshot_interval=1024))
    a = prompt(1200, salt=1)
    d.run("a", a)
    for _ in range(10):
        d.cache.ensure_free(d.bm.get_num_free_blocks() + 1)
        d.apply()
        d.apply()
    run = [d.cache._blocks[h] for h in chain(a, 8)]
    for i in range(3):  # fill the host with unrelated blocks
        d.run(f"q{i}", one_block(10 + i))
    evict_all(d)
    d.apply()
    d.apply()
    d.apply()
    assert [x.tier.name for x in run] == ["SSD"] * 8
    assert d.cache._host.num_free == 0
    d.add("a2", a, max_tokens=4)
    # Eight rows are needed for the only snapshot and the host has three:
    # there is nothing to wait for.
    assert d.admissions["a2"].admitted and d.cache._retry is None
    assert d.cache.ssd_waits == 0 and not any(x.reading for x in run)
    d.apply(aborts=["a2"])
    d.apply()
    check_ssd(d)
    d.close()


def test_failed_load_submission_gives_the_rows_back(tmp_path, io):
    from dynamo.deepinfer.kv_cache.prefix import _Evicted

    d = Driver(ssd_runner(tmp_path, rows=8, ssd_blocks=64))
    a = prompt(1200, salt=1)
    d.run("a", a)
    evict_all(d)
    d.apply()
    run = [d.cache._blocks[h] for h in chain(a, 8)]
    for x in reversed(run):  # a leaves the host: only the SSD copies stay
        if x.tier.name == "HOST":
            d.cache._evict_host(x, _Evicted())
    d.apply()
    assert any(x.tier.name == "SSD" for x in run)

    submitted = []

    def boom(*args, **kwargs):
        submitted.append(1)
        raise OSError("injected")

    d.cache._ssd.load = boom
    free = d.cache._host.num_free
    d.add("a2", a, max_tokens=4)
    assert submitted and not any(x.reading for x in run)
    assert d.cache._host.num_free == free
    d.apply(aborts=["a2"])
    d.apply()
    check_ssd(d)
    d.close()


def test_reads_need_enough_ssd_blocks_not_just_a_long_run(tmp_path, io, monkeypatch):
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MIN_READ_BLOCKS", 4)
    d = Driver(ssd_runner(tmp_path, rows=8, ssd_blocks=64))
    a, b = prompt(1200, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    tiers = [d.cache._blocks[h].tier.name for h in chain(a, 8)]
    assert tiers == ["HOST"] * 5 + ["SSD"] * 3
    calls = spy_io(d)
    d.add("a2", a, max_tokens=4)
    # Four blocks lie behind the hit (one on host, without a snapshot), but
    # only three are on SSD.
    assert d.admissions["a2"].hit_tokens == 512 and calls == []
    d.apply(aborts=["a2"])
    d.close()


def test_a_warm_block_that_goes_back_to_ssd_is_cold_again(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    blocks = [d.cache._blocks[h] for h in chain(a)]
    d.add("x", a, max_tokens=2)
    d.apply(aborts=["x"])
    d.apply()
    assert all(x.tier.name == "HOST" and x.ssd_warm for x in blocks)
    d.run("c", prompt(600, salt=3))
    evict_all(d)
    d.apply()
    assert all(x.tier.name == "SSD" and not x.ssd_warm for x in blocks)
    d.close()


def test_a_block_read_back_from_ssd_is_not_the_first_host_victim(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=4))
    a, b = prompt(600, salt=1), prompt(600, salt=2)
    d.run("a", a)
    evict_all(d)
    d.apply()
    d.run("b", b)
    evict_all(d)
    d.apply()
    ba, bb = ([d.cache._blocks[h] for h in chain(p)] for p in (a, b))
    assert [x.tier.name for x in ba] == ["HOST", "SSD", "SSD"]
    d.add("x", a, max_tokens=2)
    d.apply(aborts=["x"])
    d.apply()
    # The read took b's two leaves' rows; b's root is the older host leaf.
    assert [x.tier.name for x in ba + bb] == ["HOST"] * 3 + ["HOST", "SSD", "SSD"]
    assert ba[2].ssd_warm and ba[2].last_used > bb[0].last_used
    assert d.cache._pop_evictable(gpu=False) is bb[0]
    d.close()


def park_on_ssd(d: Driver, prompts: list[list[int]], fillers: int) -> None:
    """Cache ``prompts`` and push them down to the SSD tier with ``fillers``
    other prompts that take over the host rows."""
    for i, p in enumerate(prompts):
        d.run(f"p{i}", p)
        evict_all(d)
        d.apply()
    for i in range(fillers):
        d.run(f"f{i}", prompt(600, salt=20 + i))
        evict_all(d)
        d.apply()


def wait_for_ssd(monkeypatch, min_blocks: int = 1):
    """Let short runs be read, and so waited for, with a clock the test sets."""
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_MIN_READ_BLOCKS", min_blocks)
    return [0.0]


def test_a_long_ssd_run_waits_for_its_read_and_then_hits_from_host(
    tmp_path, io, monkeypatch
):
    wait_for_ssd(monkeypatch)
    d = Driver(ssd_runner(tmp_path, rows=3))
    a = prompt(600, salt=1)
    park_on_ssd(d, [a], fillers=1)
    blocks = [d.cache._blocks[h] for h in chain(a)]
    assert [x.tier.name for x in blocks] == ["SSD"] * 3
    free = d.bm.get_num_free_blocks()
    io.hold = True
    d.add("x", a, max_tokens=4)
    d.apply()
    assert not d.admissions["x"].admitted and d.cache.ssd_waits == 1
    assert all(x.reading for x in blocks) and d.cache.ssd_loads == 0
    # Nothing is held while it waits: no pin, no GPU block, no state slot.
    assert d.cache._retry.held == {} and d.cache._requests == {}
    assert all(x.users == 0 for x in d.cache._blocks.values())
    assert d.bm.get_num_free_blocks() == free
    io.hold = False
    io.release()
    d.apply()
    assert d.admissions["x"].admitted and d.admissions["x"].hit_tokens == 384
    assert d.cache.ssd_hit_tokens == 384 and d.cache.host_hit_tokens == 384
    assert d.cache.ssd_wait_timeouts == 0
    d.apply(aborts=["x"])
    d.apply()
    check_ssd(d)
    d.close()


def test_the_wait_ends_at_its_deadline_and_the_request_recomputes(
    tmp_path, io, monkeypatch
):
    now = wait_for_ssd(monkeypatch)
    d = Driver(ssd_runner(tmp_path, rows=3))
    d.cache._now = lambda: now[0]
    a = prompt(600, salt=1)
    park_on_ssd(d, [a], fillers=1)
    run = [d.cache._blocks[h] for h in chain(a)]
    io.hold = True
    d.add("x", a, max_tokens=50)
    req = d.cache._retry
    now[0] += 0.001  # the deadline of three blocks is 2.3 ms
    d.apply()
    assert not d.admissions["x"].admitted and d.cache.ssd_wait_timeouts == 0
    now[0] += 1.0
    d.apply()
    assert d.admissions["x"].admitted and d.admissions["x"].hit_tokens == 0
    assert d.cache.ssd_waits == 1 and d.cache.ssd_wait_timeouts == 1
    # A request never waits twice, even if admission is retried for another reason.
    assert req.waited and not d.cache._wait_for_read(req, [], run)
    assert d.cache.ssd_waits == 1 and d.cache.ssd_wait_timeouts == 1
    d.apply(aborts=["x"])
    io.hold = False
    io.release()
    d.apply()
    d.apply()
    check_ssd(d)
    d.close()


def test_a_read_that_fails_while_the_request_waits_ends_the_wait(
    tmp_path, io, monkeypatch
):
    wait_for_ssd(monkeypatch)
    d = Driver(ssd_runner(tmp_path, rows=3))
    cold = Driver(make_runner("none"))
    a = prompt(600, salt=1)
    park_on_ssd(d, [a], fillers=1)
    fail_io(d, "load")
    io.hold = True
    d.add("x", a, max_tokens=4)
    assert not d.admissions["x"].admitted
    io.hold = False
    io.release()
    d.apply()
    # The blocks are gone, so the lookup misses: recomputed, not waited for.
    assert d.admissions["x"].admitted and d.admissions["x"].hit_tokens == 0
    assert d.cache.ssd_errors == 1 and d.cache.ssd_wait_timeouts == 0
    out: list[int] = []
    for _ in range(50):
        for o in d.apply().outputs:
            out += o.token_ids
        if len(out) >= 4:
            break
    assert out == cold.run("c", a)
    d.apply()
    check_ssd(d)
    d.close()
    cold.close()


@pytest.mark.parametrize(
    "min_blocks, wait_ms, reads",
    [(4, 50.0, False), (1, 0.0, True)],
    ids=["too short to read", "wait off"],
)
def test_short_runs_and_no_wait_are_admitted_at_once(
    tmp_path, io, monkeypatch, min_blocks, wait_ms, reads
):
    wait_for_ssd(monkeypatch, min_blocks)
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_wait_ms=wait_ms))
    a = prompt(600, salt=1)
    park_on_ssd(d, [a], fillers=1)
    blocks = [d.cache._blocks[h] for h in chain(a)]
    io.hold = True
    d.add("x", a, max_tokens=50)
    # A run below the read minimum is neither read nor waited for; with the wait
    # off the read is best effort: the request goes on and it still starts.
    assert d.admissions["x"].admitted and d.admissions["x"].hit_tokens == 0
    assert all(x.reading for x in blocks) == reads and d.cache.ssd_waits == 0
    d.apply(aborts=["x"])
    io.hold = False
    io.release()
    d.apply()
    d.close()


def test_the_deadline_is_capped_by_the_ceiling_and_grows_with_the_blocks(
    tmp_path, io, monkeypatch
):
    now = wait_for_ssd(monkeypatch)
    d = Driver(ssd_runner(tmp_path, rows=8, ssd_blocks=64))
    d.cache._now = lambda: now[0]
    a = prompt(1200, salt=1)
    park_on_ssd(d, [a], fillers=3)
    tiers = [d.cache._blocks[h].tier.name for h in chain(a, 8)]
    assert tiers == ["SSD"] * 8, tiers
    io.hold = True
    deadlines = []
    for rid, tokens, ceiling in [("x", a[:600], 50.0), ("y", a, 50.0), ("z", a, 2.5)]:
        d.cache.ssd_wait_ms = ceiling
        d.add(rid, tokens, max_tokens=4)
        assert not d.admissions[rid].admitted
        deadlines.append(d.cache._retry.deadline)
        d.apply(aborts=[rid])
    from dynamo.deepinfer.kv_cache import prefix

    base, per = prefix.SSD_WAIT_BASE_MS, prefix.SSD_WAIT_MS_PER_BLOCK
    # Two SSD blocks lie behind the hit for the short prompt, eight for the long.
    assert deadlines == pytest.approx(
        [(base + 2 * per) / 1000, (base + 8 * per) / 1000, 2.5 / 1000]
    )
    io.hold = False
    io.release()
    d.apply()
    d.close()


def parked_pair(tmp_path, monkeypatch):
    """Two prompts on SSD (3 blocks each) and a host tier of 6 rows, all of which
    prefetch may use."""
    from dynamo.deepinfer.kv_cache import prefix

    monkeypatch.setattr(prefix, "SSD_PREFETCH_ROWS_FRACTION", 1.0)
    d = Driver(ssd_runner(tmp_path, rows=6))
    a, c = prompt(600, salt=1), prompt(600, salt=2)
    park_on_ssd(d, [a, c], fillers=2)
    for p in (a, c):
        assert d.cache._blocks[chain(p, 1)[0]].tier.name == "SSD"
    return d, [np.asarray(p, dtype=np.int32) for p in (a, c)]


def enqueue(d: Driver, tokens: Any, seq_id: int = 90) -> None:
    d.cache.on_enqueue(SimpleNamespace(seq_id=seq_id), tokens)


def test_only_a_request_behind_others_is_prefetched_and_admit_reuses_its_hashes(
    tmp_path, io, monkeypatch
):
    d, (a, _) = parked_pair(tmp_path, monkeypatch)
    cache = d.cache
    queued, lookups = [], []
    on_enqueue, lookup = cache.on_enqueue, cache._lookup
    cache.on_enqueue = lambda seq, tokens: (
        queued.append(seq.request_id),
        on_enqueue(seq, tokens),
    )
    cache._lookup = lambda req, n: (lookups.append(req), lookup(req, n))[1]
    d.apply(
        adds=[
            AddReq(rid, prompt_bytes(a.tolist()), sampling={"max_tokens": 4})
            for rid in ("x1", "x2")
        ]
    )
    # x1 is first in line and x2 is behind it.
    assert queued == ["x2"]
    [prefetched] = [r for r in lookups if r.seq.request_id == "x2"][:1]
    assert [r for r in lookups if r.seq.request_id == "x2"][1] is prefetched
    assert not cache._prefetched
    io.hold = False
    io.release()
    d.apply(aborts=["x1", "x2"])
    d.close()


def test_enqueue_starts_the_read_and_the_request_then_hits_from_host(
    tmp_path, io, monkeypatch
):
    d, (a, _) = parked_pair(tmp_path, monkeypatch)
    blocks = [d.cache._blocks[h] for h in chain(a.tolist())]
    calls = spy_io(d)
    free = d.bm.get_num_free_blocks()
    io.hold = True
    enqueue(d, a)
    enqueue(d, a, seq_id=91)  # the blocks are being read: nothing more to start
    assert calls == [("load", [x.ssd_slot for x in blocks])]
    assert d.cache._reads_inflight == 1 and d.cache._reading_rows == 3
    # Only host rows are used: no GPU block, pin or promotion.
    assert d.bm.get_num_free_blocks() == free
    assert all(x.tier.name == "SSD" and x.users == 0 for x in blocks)
    io.hold = False
    io.release()
    d.apply()
    assert [x.tier.name for x in blocks] == ["HOST"] * 3
    assert d.cache._reads_inflight == 0 and d.cache._reading_rows == 0
    d.add("x", a.tolist(), max_tokens=4)
    assert d.admissions["x"].hit_tokens == 384 and d.cache.ssd_waits == 0
    assert d.cache.ssd_hit_tokens == 384
    d.apply(aborts=["x"])
    d.apply()
    check_ssd(d)
    d.close()


@pytest.mark.parametrize("cap", ["reads", "rows"])
def test_enqueue_skips_a_prefetch_over_either_cap(tmp_path, io, monkeypatch, cap):
    from dynamo.deepinfer.kv_cache import prefix

    d, (a, c) = parked_pair(tmp_path, monkeypatch)
    if cap == "reads":
        monkeypatch.setattr(prefix, "SSD_MAX_PREFETCH_READS", 1)
    else:  # room for one read of three blocks in six rows
        monkeypatch.setattr(prefix, "SSD_PREFETCH_ROWS_FRACTION", 0.5)
    calls = spy_io(d)
    io.hold = True
    enqueue(d, a)
    enqueue(d, c, seq_id=91)
    assert [kind for kind, _ in calls] == ["load"]
    assert d.cache._blocks[chain(c.tolist(), 1)[0]].tier.name == "SSD"
    # A read that landed frees its share of both caps.
    io.hold = False
    io.release()
    d.apply()
    enqueue(d, c, seq_id=92)
    assert [kind for kind, _ in calls] == ["load", "load"]
    io.release()
    d.apply()
    check_ssd(d)
    d.close()


def test_a_request_aborted_while_it_waits_is_not_presented_again(
    tmp_path, io, monkeypatch
):
    wait_for_ssd(monkeypatch)
    d = Driver(ssd_runner(tmp_path, rows=3))
    a = prompt(600, salt=1)
    park_on_ssd(d, [a], fillers=1)
    seen: list[str] = []
    admit = d.cache.admit

    def spy_seen(seq: Any, tokens: Any) -> Admission:
        seen.append(seq.request_id)
        return admit(seq, tokens)

    d.cache.admit = spy_seen
    io.hold = True
    d.add("x", a, max_tokens=4)
    assert seen == ["x"]
    d.apply(aborts=["x"])
    io.hold = False
    io.release()
    for _ in range(3):
        d.apply()
    assert seen == ["x"] and "x" not in d.cache._requests
    assert d.cache.ssd_loads == 3  # the read it started still landed
    check_ssd(d)
    d.close()


def test_closed_cache_still_drops_blocks_with_a_slot(tmp_path, io):
    from dynamo.deepinfer.kv_cache.prefix import _Evicted

    d = Driver(ssd_runner(tmp_path, rows=3))
    d.run("a", prompt(600, salt=1))
    evict_all(d)
    d.apply()
    block = next(iter(d.cache._blocks.values()))
    assert block.ssd_slot >= 0
    d.cache.close()
    out = _Evicted()
    d.cache._drop(block, out)
    assert block.ssd_slot == -1 and out.removed
    d.close()


def test_failing_close_steps_do_not_stop_the_rest_and_the_first_error_wins(
    tmp_path, io, caplog
):
    d = Driver(ssd_runner(tmp_path, rows=3))
    steps: list[str] = []
    close = d.cache.close

    class Prefill:
        def close(self):
            steps.append("prefill")
            raise RuntimeError("boom")

    class Transfer:
        def shutdown(self):
            steps.append("transfer")
            raise RuntimeError("second")

    d.runner.prefill = Prefill()
    d.runner.engine.dynamo_transfer = Transfer()
    d.cache.close = lambda: steps.append("cache") or close()
    sink_close = d.runner.sink.close
    d.runner.sink.close = lambda: steps.append("sink") or sink_close()
    with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="boom"):
        d.runner.close()
    assert steps == ["prefill", "cache", "transfer", "sink"]
    assert d.cache._ssd is None and "second" in caplog.text


def test_restart_starts_with_an_empty_ssd_tier(tmp_path, io):
    d = Driver(ssd_runner(tmp_path, rows=3))
    a = prompt(600, salt=1)
    d.run("a", a)
    evict_all(d)
    d.apply()
    path = d.cache._ssd.path
    assert any(file_row(d, x.ssd_slot) != bytes(512) for x in d.cache._blocks.values())
    # A crash leaves the file (and its blocks) behind, and its lock is gone: the
    # next start is empty.
    fcntl.flock(d.cache._ssd._lock_fd, fcntl.LOCK_UN)
    d2 = Driver(ssd_runner(tmp_path, rows=3))
    pool = d2.cache._ssd
    assert pool.path == path and pool.num_free == pool.capacity
    with open(path, "rb") as f:
        assert not any(f.read())
    assert not d2.cache._blocks
    assert d2.run("a", a) and d2.admissions["a"].hit_tokens == 0
    d2.close()
    d.close()


def test_a_stale_ssd_file_does_not_count_against_the_free_space(
    tmp_path, io, monkeypatch
):
    stale = tmp_path / "rank0.kv"
    stale.write_bytes(bytes(20 * SSD_STRIDE))
    real = os.statvfs

    def statvfs(path):
        st = tuple(real(path))
        # 11 blocks free, 5% of them kept back: 10; none while the file is there.
        free = 0 if stale.exists() else 11
        return os.statvfs_result((st[0], SSD_STRIDE, st[2], free, free) + st[5:])

    monkeypatch.setattr(os, "statvfs", statvfs)
    d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=20))
    assert d.cache._ssd is not None and d.cache._ssd.capacity == 10
    d.close()


def test_ssd_tier_is_clamped_to_snapshots_and_free_space(
    tmp_path, io, caplog, monkeypatch
):
    # 16 snapshots of 2 blocks (a 256-token interval) can serve 32 blocks.
    with caplog.at_level("WARNING"):
        d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=1000))
    assert d.cache._ssd.capacity == 32
    [msg] = [r.getMessage() for r in caplog.records if "clamped" in r.getMessage()]
    assert "32 of 1000" in msg and "snapshots can serve only 32" in msg
    d.close()

    caplog.clear()
    real = os.statvfs
    free_blocks = 11  # less the 5% kept back: 10

    def statvfs(path):
        st = tuple(real(path))
        return os.statvfs_result(
            (st[0], SSD_STRIDE, st[2], free_blocks, free_blocks) + st[5:]
        )

    monkeypatch.setattr(os, "statvfs", statvfs)
    with caplog.at_level("WARNING"):
        d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=20))
    assert d.cache._ssd.capacity == 10
    assert any(
        "clamped to 10 of 20" in r.getMessage()
        and "free for this node" in r.getMessage()
        for r in caplog.records
    )
    d.close()

    # Two ranks of a node share the directory: each takes half of what is free
    # (41 blocks, 5% kept back: 19 each). The space is used up as the ranks
    # start, and left over from a crash: neither changes the share.
    def statvfs_of_disk(path):
        used = sum(f.stat().st_blocks * 512 for f in tmp_path.glob("*.kv"))
        st = tuple(real(path))
        free = 41 - used // SSD_STRIDE
        return os.statvfs_result((st[0], SSD_STRIDE, st[2], free, free) + st[5:])

    monkeypatch.setattr(os, "statvfs", statvfs_of_disk)
    for _ in range(2):  # the second time, with the first run's files in place
        halves = []
        for r in (0, 1):
            halves.append(
                Driver(
                    ssd_runner(
                        tmp_path, rows=3, ssd_blocks=1000, ssd_rank=r, ssd_share=0.5
                    )
                )
            )
        assert [h.cache._ssd.capacity for h in halves] == [19, 19]
        for h in halves:  # a crash leaves the files behind: close only the pools
            os.close(h.cache._ssd._fd)
            os.close(h.cache._ssd._lock_fd)
            h.cache._ssd._closed = True
    for f in tmp_path.glob("*.kv"):
        f.unlink()
    monkeypatch.setattr(os, "statvfs", statvfs)
    caplog.clear()
    free_blocks = 0
    with caplog.at_level("WARNING"):
        d = Driver(ssd_runner(tmp_path, rows=3, ssd_blocks=20))
    assert d.cache._ssd is None
    assert any("no free space" in r.getMessage() for r in caplog.records)
    p = prompt(600)
    d.run("a", p)
    assert d.run("b", p) and d.admissions["b"].hit_tokens == 384
    d.close()


def test_ssd_tier_needs_the_host_tier(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        d = Driver(make_runner(ssd_dir=str(tmp_path), ssd_bytes=1 << 20))
    assert d.cache._ssd is None and "needs the host tier" in caplog.text
    d.close()


def test_ssd_tier_over_real_io_threads(tmp_path):
    d = Driver(ssd_runner(tmp_path, rows=4))
    p = prompt(600)
    d.run("a", p)
    evict_all(d)
    for _ in range(500):  # the pool's threads write; completions land in a step
        d.apply()
        if d.cache.ssd_stores == 3:
            break
        time.sleep(0.01)
    assert d.cache.ssd_stores == 3 and d.cache.ssd_errors == 0
    check_ssd(d)
    d.close()
