# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Seeded random traffic over the prefix cache with the host and SSD tiers on:
requests that share prefixes come and go, SSD I/O is held back and released, and
reads and writes fail, requests wait for their reads (on a fake clock in some
seeds) and are prefetched before they are admitted. At quiescent points every tier, pool, heap and event
must add up (``check_ssd``)."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.deepinfer.tests.test_prefix_cache import (  # noqa: F401
    Driver,
    _env,
    check_ssd,
    io,
    prompt,
    ssd_runner,
)

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]

SEEDS = range(40)


def quiesce(
    d: Driver,
    io,  # noqa: F811
    live: set[str],
    fail: dict[str, bool],
) -> None:
    """Finish or abort everything, let all I/O land and let the cache settle."""
    d.apply(aborts=sorted(live))
    live.clear()
    fail["load"] = fail["store"] = False
    io.hold = False
    io.release()
    cache = d.cache
    # Queued writes start a few per poll (and a poll applies the ones before).
    for _ in range(60):
        d.apply()
        io.release()
        if not any(b.write for b in cache._blocks.values()) and not any(
            cache._write_pending(e) for e in cache._pending_writes
        ):
            break
    check_ssd(d)
    # A queued write stays only when the SSD has no free slot and no copy to
    # reclaim for one.
    if any(cache._write_pending(e) for e in cache._pending_writes):
        assert not cache._ssd._free
        assert not any(
            b.ssd_state.name == "DONE" and cache._ssd_victim(b)
            for b in cache._blocks.values()
        )
    assert cache._num_unused == sum(b.on_gpu for b in cache._blocks.values())
    for pool in (cache._host, cache._ssd, cache._pool):
        assert len(set(pool._free)) == len(pool._free)
    slots = {b.ssd_slot for b in cache._blocks.values() if b.ssd_slot >= 0}
    assert not slots & set(cache._ssd._free)
    rows = {b.host_slot for b in cache._blocks.values() if b.host_slot >= 0}
    assert not rows & set(cache._host._free)


@pytest.mark.parametrize("seed", SEEDS)
def test_random_traffic_keeps_the_tiers_consistent(
    seed,
    tmp_path,
    io,  # noqa: F811
    monkeypatch,
):
    from dynamo.deepinfer.kv_cache import prefix

    rnd = random.Random(seed)
    gpu_blocks = rnd.choice([32, 40, 48, 64])
    monkeypatch.setenv("DEEPINFER_FAKE_NUM_GPU_BLOCKS", str(gpu_blocks))
    monkeypatch.setattr(prefix, "SSD_MAX_INFLIGHT_WRITES", rnd.choice([1, 2, 64]))
    monkeypatch.setattr(prefix, "SSD_MIN_READ_BLOCKS", rnd.choice([1, 1, 4, 8]))
    monkeypatch.setattr(prefix, "SSD_MAX_PREFETCH_READS", rnd.choice([1, 2, 8]))
    monkeypatch.setattr(prefix, "SSD_PREFETCH_ROWS_FRACTION", rnd.choice([0.25, 1.0]))
    d = Driver(
        ssd_runner(
            tmp_path,
            rows=rnd.choice([2, 4, 8, 12]),
            ssd_blocks=rnd.choice([4, 8, 16, 32]),
            snapshot_interval=rnd.choice([256, 1024]),
            ssd_wait_ms=rnd.choice([0, 50, 50]),
        )
    )
    clock = [0.0]
    if rnd.random() < 0.5:
        d.cache._now = lambda: clock[0]
    pool = d.cache._ssd
    transfer = pool._transfer
    fail = {"load": False, "store": False}

    def maybe_fail(slot, view, *, write):
        if fail["store" if write else "load"]:
            raise OSError("injected")
        transfer(slot, view, write=write)

    pool._transfer = maybe_fail
    cache = d.cache
    reads = []  # the tickets of the reads in flight
    load, read_done, acquire = pool.load, cache._read_done, cache._acquire

    def track_load(*args, **kwargs):
        ticket = load(*args, **kwargs)  # a failed submission raises first
        reads.append(kwargs["tag"])
        return ticket

    def track_done(ticket, out):
        reads.remove(ticket.tag)
        return read_done(ticket, out)

    def checked_acquire(req, block):
        # A hit is acquired only while it is cached, and not on SSD.
        assert cache._blocks.get(block.block_hash) is block
        assert block.tier.name != "SSD"
        return acquire(req, block)

    pool.load, cache._read_done, cache._acquire = (
        track_load,
        track_done,
        checked_acquire,
    )
    families = [prompt(1800, salt=s) for s in range(rnd.choice([2, 3]))]
    live: set[str] = set()
    steps = rnd.randint(150, 300)
    for step in range(steps):
        r = rnd.random()
        if r < 0.35 and len(live) < 4:
            length = rnd.choice([300, 520, 800, 1100, 1500, 1800])
            tokens = rnd.choice(families)[:length]
            if rnd.random() < 0.3:  # branch off the shared prefix
                half = length // 2
                tokens = tokens[:half] + prompt(length - half, salt=rnd.randint(10, 13))
            rid = f"r{step}"
            if rnd.random() < 0.3:  # a prefetch from a request queued earlier
                d.cache.on_enqueue(
                    SimpleNamespace(seq_id=10_000 + step), np.asarray(tokens, np.int32)
                )
            d.add(rid, tokens, max_tokens=rnd.choice([2, 6, 20]))
            live.add(rid)
        elif r < 0.45 and live:
            rid = rnd.choice(sorted(live))
            d.apply(aborts=[rid])
            live.discard(rid)
        elif r < 0.55:
            io.hold = not io.hold
            if not io.hold:
                io.release()
        elif r < 0.6:
            fail[rnd.choice(["load", "store"])] = rnd.random() < 0.5
        elif r < 0.75:
            d.cache.ensure_free(rnd.randint(1, gpu_blocks))
        else:
            for o in d.apply().outputs:
                if o.finish_reason is not None:
                    live.discard(o.request_id)
        clock[0] += rnd.choice([0, 0, 0.001, 0.02, 0.1])
        # A block dropped while it is read keeps its rows counted until the read
        # lands.
        assert cache._reads_inflight == len(reads)
        assert cache._reading_rows == sum(len(t.blocks) for t in reads)
        # A live queued write has its block on a host row: the host bounds them.
        waiting = {id(e[2]) for e in cache._pending_writes if cache._write_pending(e)}
        assert len(waiting) <= cache._host.capacity
        if step == steps // 2:
            quiesce(d, io, live, fail)
    quiesce(d, io, live, fail)
    d.close()
