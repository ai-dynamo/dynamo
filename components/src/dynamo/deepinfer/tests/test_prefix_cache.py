# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PrefixCacheManager over the CPU fake of DeepInfer: hits, checkpoints,
eviction, refcounts and events."""

import hashlib
import pickle
from typing import Any

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
        admit, take = self.cache.admit, self.cache.take_events

        def spy_admit(seq: Any, tokens: Any) -> Admission:
            result = admit(seq, tokens)
            self.admissions[seq.request_id] = result
            return result

        def spy_take() -> list[Any]:
            events = take()
            self.events.extend(events)
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
