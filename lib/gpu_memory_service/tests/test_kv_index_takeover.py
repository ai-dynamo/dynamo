# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The takeover decision: which situation we woke into, and what we do about it.

``test_kv_index.py`` covers the mirror's mechanics. This covers ``on_wake_up``:
which of the two situations we are in, and what we do about it. Both bugs found
by actually running this were in here, and neither was covered by a test:

- the mirror was wiped on every sleep, because vLLM clears its own index there
  and we faithfully mirrored that. Silent: the feature just quietly did nothing.
- "did I inherit these pages?" was read from the GMS grant, which cannot answer
  it -- ``commit_layout()`` regrants a *creating* writer to RW_DATA too. That
  one would have replayed a dead engine's labels onto fresh memory.

Real ``BlockPool``, fake engine core. No GPU.
"""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("vllm")

from gpu_memory_service.integrations.vllm import kv_index  # noqa: E402
from vllm.v1.core import kv_cache_utils  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]

N_BLOCKS = 64
BLOCK_SIZE = 16


class _Cfg:
    """The slice of VllmConfig that the identity digest reads."""

    class model_config:
        model = "test/model"

    class cache_config:
        block_size = BLOCK_SIZE
        cache_dtype = "auto"
        prefix_caching_hash_algo = "sha256"

    class parallel_config:
        tensor_parallel_size = 1


class FakeEngineCore:
    """Just enough EngineCore for on_wake_up: a pool, a scheduler, a probe."""

    def __init__(self, pool, adopted: bool, sleeping: bool = False):
        self._adopted = adopted
        self.vllm_config = _Cfg
        self.model_executor = type("Ex", (), {"is_sleeping": sleeping})()
        self.scheduler = type(
            "Sched",
            (),
            {
                "kv_cache_manager": type("M", (), {"block_pool": pool})(),
                "schedule": lambda *a, **k: None,
                "update_from_output": lambda *a, **k: None,
            },
        )()

    def collective_rpc(self, fn):
        return [self._adopted]


def wake(engine):
    """A wake, as the plugin performs it: on_wake_up only runs after a sleep."""
    kv_index._IN_SLEEP = True
    try:
        kv_index.on_wake_up(engine)
    finally:
        kv_index._IN_SLEEP = False


def key_for(tag: str):
    return make_block_hash_with_group_id(
        BlockHash(hashlib.sha256(tag.encode()).digest()), 0
    )


@pytest.fixture(autouse=True)
def _seeded_none_hash(monkeypatch):
    # NONE_HASH is a module global only assigned by init_none_hash(); the
    # identity digest reads it, so a unit test has to seed it explicitly.
    from vllm.utils.hashing import sha256

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    kv_cache_utils.init_none_hash(sha256)
    monkeypatch.setattr(kv_index, "_IN_SLEEP", False, raising=False)


@pytest.fixture
def path(tmp_path, monkeypatch):
    p = str(tmp_path / "kvidx.mirror")
    monkeypatch.setenv("GMS_KV_INDEX_PATH", p)
    return p


def new_pool():
    return BlockPool(N_BLOCKS, True, BLOCK_SIZE)


def populate(pool, n=5):
    """Serve a little: label n blocks and let their batch complete."""
    pool._dyn_mirror.on_schedule()
    blocks = pool.get_new_blocks(n)
    for i, b in enumerate(blocks):
        pool._insert_block_hash(key_for(f"k{i}"), b, BLOCK_SIZE * (i + 1))
    pool._dyn_mirror.on_update()
    pool.free_blocks(blocks)
    return blocks


# ---------------------------------------------------------------------------
# bug 1: sleep must not wipe the mirror
# ---------------------------------------------------------------------------


def test_sleep_does_not_wipe_the_mirror(path):
    """vLLM clears its index at sleep; under GMS the bytes survive, so we keep ours.

    Regression: mirroring that clear made the feature silently do nothing --
    every sleep destroyed exactly what the sleep was supposed to preserve.
    """
    pool = new_pool()
    engine = FakeEngineCore(pool, adopted=False)
    wake(engine)
    blocks = populate(pool)
    assert pool._dyn_mirror.live_block_ids() == {b.block_id for b in blocks}

    # Entering the sleep window, vLLM drops its own index.
    kv_index._IN_SLEEP = True
    assert pool.reset_prefix_cache() is True
    assert not any(b.block_hash for b in pool.blocks), "vLLM's index should be gone"
    assert pool._dyn_mirror.live_block_ids() == {
        b.block_id for b in blocks
    }, "the mirror must survive a sleep -- GMS still holds the pages"


def test_reset_outside_a_sleep_does_wipe_the_mirror(path):
    """An RLHF weight update is the case the wipe exists for."""
    pool = new_pool()
    engine = FakeEngineCore(pool, adopted=False)
    wake(engine)
    populate(pool)

    assert kv_index._IN_SLEEP is False
    assert pool.reset_prefix_cache() is True
    assert (
        pool._dyn_mirror.live_block_ids() == set()
    ), "outside a sleep the operator meant it: the labels are dead"


# ---------------------------------------------------------------------------
# bug 2: replay only onto pages we actually inherited
# ---------------------------------------------------------------------------


def test_inherited_pages_replay(path):
    """The happy path: a successor rebuilds the predecessor's index."""
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    blocks = populate(pool_a)
    expected = {b.block_id: b.block_hash for b in blocks}

    pool_b = new_pool()  # a different engine, same geometry
    wake(FakeEngineCore(pool_b, adopted=True))

    got = {b.block_id: b.block_hash for b in pool_b.blocks if b.block_hash}
    assert got == expected


def test_fresh_pages_never_replay(path):
    """The bug that would have served wrong tokens.

    An engine that built its own pool must not install a predecessor's labels:
    they name pages that no longer exist. Starting a fresh mirror is also what
    makes a stale one inert for everybody after us.
    """
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    populate(pool_a)

    pool_b = new_pool()
    wake(FakeEngineCore(pool_b, adopted=False))  # built its own

    assert not any(
        b.block_hash for b in pool_b.blocks
    ), "labels were installed onto pages this engine allocated itself"
    assert pool_b._dyn_mirror.live_block_ids() == set(), "stale mirror not discarded"


def test_partial_adoption_across_ranks_refuses(path):
    """Every rank, or nobody.

    The index is engine-wide but the bytes are per-rank, so a partial adoption
    is correct on the ranks that adopted and garbage on the one that did not --
    which is the shape that produces plausible-looking wrong output.
    """
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    populate(pool_a)

    pool_b = new_pool()
    engine = FakeEngineCore(pool_b, adopted=True)
    engine.collective_rpc = lambda fn: [True, False]  # rank 1 did not adopt
    wake(engine)

    assert not any(b.block_hash for b in pool_b.blocks)


def test_a_probe_that_fails_refuses(path):
    """Losing the signal is a miss, never a guess."""
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    populate(pool_a)

    pool_b = new_pool()
    engine = FakeEngineCore(pool_b, adopted=True)

    def boom(fn):
        raise RuntimeError("workers unreachable")

    engine.collective_rpc = boom
    wake(engine)
    assert not any(b.block_hash for b in pool_b.blocks)


# ---------------------------------------------------------------------------
# the ruler
# ---------------------------------------------------------------------------


def test_a_different_ruler_refuses(path):
    """Same block id, different meaning: the one failure that is not a miss."""
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    populate(pool_a)

    class Changed(_Cfg):
        class cache_config(_Cfg.cache_config):
            block_size = BLOCK_SIZE * 2  # different geometry

    pool_b = new_pool()
    engine = FakeEngineCore(pool_b, adopted=True)
    engine.vllm_config = Changed
    wake(engine)

    assert not any(b.block_hash for b in pool_b.blocks)


def test_a_different_block_count_refuses(path):
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    populate(pool_a)

    pool_b = BlockPool(N_BLOCKS * 2, True, BLOCK_SIZE)
    wake(FakeEngineCore(pool_b, adopted=True))
    assert not any(b.block_hash for b in pool_b.blocks)


# ---------------------------------------------------------------------------
# the known gap
# ---------------------------------------------------------------------------


def test_known_gap_a_non_participating_engine_is_invisible(path):
    """DOCUMENTS A HOLE. This test asserts behaviour we want to change.

    An engine with DYN_GMS_PERSIST_KV=1 but this feature *off* adopts the pages,
    overwrites blocks, and dies without touching the mirror. Nothing it does is
    observable to us: same geometry, same identity digest, and it leaves no
    counter behind. The next participant therefore replays labels describing
    bytes that engine overwrote -- a wrong answer, not a miss.

    Closing it needs a writer epoch on the GMS handshake (a wire change, hence
    a follow-up). When that lands, this test SHOULD start failing: replace it
    with one asserting the replay is refused.
    """
    pool_a = new_pool()
    wake(FakeEngineCore(pool_a, adopted=False))
    blocks = populate(pool_a)
    stale = {b.block_id: b.block_hash for b in blocks}

    # Engine B adopts the same pages with the feature off: it never opens the
    # mirror, so the file still describes engine A's contents while B is free
    # to overwrite those very blocks.

    # Engine C, a participant again, takes over after B died.
    pool_c = new_pool()
    wake(FakeEngineCore(pool_c, adopted=True))
    got = {b.block_id: b.block_hash for b in pool_c.blocks if b.block_hash}

    assert (
        got == stale
    ), "if this now differs, the gap may be closed -- rewrite this test"


def test_a_wake_that_did_not_follow_a_sleep_does_nothing(path):
    """A spurious wake must not re-run the takeover.

    Against a live pool it would find every block already labelled, install
    nothing, and then ``retain_only(set())`` its way through the whole mirror --
    discarding a perfectly good index without touching a byte of KV.
    """
    pool = new_pool()
    wake(FakeEngineCore(pool, adopted=False))
    populate(pool)
    before = pool._dyn_mirror.live_block_ids()
    assert before, "nothing to protect -- test is vacuous"

    kv_index.on_wake_up(FakeEngineCore(pool, adopted=True))  # no preceding sleep

    assert pool._dyn_mirror.live_block_ids() == before


def test_a_partial_wake_does_nothing(path):
    """Some tags still asleep: not our moment, and touching the pool would be wrong."""
    pool = new_pool()
    wake(FakeEngineCore(pool, adopted=True, sleeping=True))
    assert not hasattr(pool, "_dyn_mirror")


def test_disabled_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_KV_INDEX_PATH", raising=False)
    pool = new_pool()
    wake(FakeEngineCore(pool, adopted=True))
    assert not hasattr(pool, "_dyn_mirror")
