# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Whether a woken engine may reuse its predecessor's index.

``test_block_index.py`` covers the store's mechanics. This covers the decision:
the pause/unpause transitions that bracket a sleep, and the handover they
trigger.

These mix the *real* ``MirrorsBlockIndex`` into a stand-in base, exactly as
``scheduler.py`` does, so the transition logic under test is the shipped code.
"""

from __future__ import annotations

import hashlib
import os

import pytest

pytest.importorskip("vllm")

from gpu_memory_service.integrations.vllm.kv_cache.scheduler import (  # noqa: E402
    MirrorsBlockIndex,
)
from vllm.v1.core import kv_cache_utils  # noqa: E402
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.interface import PauseState  # noqa: E402

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
        world_size = 1


class FakeBaseScheduler:
    """The surface MirrorsBlockIndex expects of whatever it derives from."""

    def __init__(self, pool):
        self.kv_cache_manager = type("M", (), {"block_pool": pool})()
        self.vllm_config = _Cfg
        self.parallel_config = _Cfg.parallel_config
        self.pause_state = PauseState.UNPAUSED

    def set_pause_state(self, pause_state) -> None:
        self.pause_state = pause_state

    def schedule(self):
        return None

    def update_from_output(self, *a, **k):
        return None


def new_scheduler(pool):
    return type("T", (MirrorsBlockIndex, FakeBaseScheduler), {})(pool)


def key_for(tag: str):
    return make_block_hash_with_group_id(
        BlockHash(hashlib.sha256(tag.encode()).digest()), 0
    )


@pytest.fixture(autouse=True)
def _seeded_none_hash(monkeypatch):
    from vllm.utils.hashing import sha256

    monkeypatch.setenv("PYTHONHASHSEED", "0")
    kv_cache_utils.init_none_hash(sha256)


@pytest.fixture
def path(tmp_path, monkeypatch):
    p = str(tmp_path / "index.store")
    monkeypatch.setenv("GMS_KV_INDEX_PATH", p)
    return p


def set_adoption(path: str, *ranks: bool) -> None:
    """Stand in for what GMSWorker.wake_up publishes, one file per rank."""
    for rank, adopted in enumerate(ranks):
        with open(f"{path}.rank{rank}", "w") as f:
            f.write("1" if adopted else "0")


def new_pool():
    return BlockPool(N_BLOCKS, True, BLOCK_SIZE)


def sleep_wake(sched):
    """One full pause/unpause cycle, as EngineCore drives it."""
    sched.set_pause_state(PauseState.PAUSED_ALL)
    sched.set_pause_state(PauseState.UNPAUSED)


def populate(pool, n=5):
    pool._dyn_store.on_schedule()
    blocks = pool.get_new_blocks(n)
    for i, b in enumerate(blocks):
        pool._insert_block_hash(key_for(f"k{i}"), b, BLOCK_SIZE * (i + 1))
    pool._dyn_store.on_complete()
    pool.free_blocks(blocks)
    return blocks


def test_sleep_does_not_wipe_the_store(path):
    pool = new_pool()
    set_adoption(path, False)
    sched = new_scheduler(pool)
    sleep_wake(sched)
    blocks = populate(pool)
    assert pool._dyn_store.live() == {b.block_id for b in blocks}

    sched.set_pause_state(PauseState.PAUSED_ALL)
    assert pool.reset_prefix_cache() is True
    assert not any(b.block_hash for b in pool.blocks), "vLLM's index should be gone"
    assert pool._dyn_store.live() == {
        b.block_id for b in blocks
    }, "the store must survive a sleep -- GMS still holds the pages"


def test_reset_outside_a_sleep_does_wipe_the_store(path):
    pool = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool))
    populate(pool)

    assert pool.reset_prefix_cache() is True
    assert pool._dyn_store.live() == set()


def test_unpause_without_a_preceding_pause_does_nothing(path):
    pool = new_pool()
    set_adoption(path, False)
    sched = new_scheduler(pool)
    sleep_wake(sched)
    populate(pool)
    before = pool._dyn_store.live()
    assert before, "nothing to protect -- test is vacuous"

    set_adoption(path, True)
    sched.set_pause_state(PauseState.UNPAUSED)

    assert pool._dyn_store.live() == before


def test_inherited_pages_replay(path):
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    blocks = populate(pool_a)
    expected = {b.block_id: b.block_hash for b in blocks}

    pool_b = new_pool()
    set_adoption(path, True)
    sleep_wake(new_scheduler(pool_b))

    assert {b.block_id: b.block_hash for b in pool_b.blocks if b.block_hash} == expected


def test_fresh_pages_never_replay(path):
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    populate(pool_a)

    pool_b = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_b))

    assert not any(b.block_hash for b in pool_b.blocks)
    assert pool_b._dyn_mirror.live_block_ids() == set()


def test_partial_adoption_across_ranks_refuses(path):
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    populate(pool_a)

    class TwoRank(_Cfg):
        class parallel_config(_Cfg.parallel_config):
            world_size = 2

    pool_b = new_pool()
    set_adoption(path, True, False)
    sched = new_scheduler(pool_b)
    sched.vllm_config = TwoRank
    sched.parallel_config = TwoRank.parallel_config
    sleep_wake(sched)

    assert not any(b.block_hash for b in pool_b.blocks)


def test_a_missing_rank_answer_refuses(path):
    """Losing the signal is a miss, never a guess."""
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    populate(pool_a)
    assert not os.path.exists(f"{path}.rank0"), "consumed by the takeover"

    # nobody publishes an answer for this wake
    pool_b = new_pool()
    sleep_wake(new_scheduler(pool_b))
    assert not any(b.block_hash for b in pool_b.blocks)


def test_a_different_ruler_refuses(path):
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    populate(pool_a)

    class Changed(_Cfg):
        class cache_config(_Cfg.cache_config):
            block_size = BLOCK_SIZE * 2

    pool_b = new_pool()
    set_adoption(path, True)
    sched = new_scheduler(pool_b)
    sched.vllm_config = Changed
    sleep_wake(sched)

    assert not any(b.block_hash for b in pool_b.blocks)


def test_a_different_block_count_refuses(path):
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    populate(pool_a)

    pool_b = BlockPool(N_BLOCKS * 2, True, BLOCK_SIZE)
    set_adoption(path, True)
    sleep_wake(new_scheduler(pool_b))
    assert not any(b.block_hash for b in pool_b.blocks)


def test_known_gap_a_non_participating_engine_is_invisible(path):
    """DOCUMENTS A HOLE. Asserts behaviour we want to change.

    An engine with persist-KV on but this feature off adopts the pages,
    overwrites blocks, and leaves nothing observable. Closing it needs a writer
    epoch on the GMS handshake; when that lands this test SHOULD fail.
    """
    pool_a = new_pool()
    set_adoption(path, False)
    sleep_wake(new_scheduler(pool_a))
    blocks = populate(pool_a)
    stale = {b.block_id: b.block_hash for b in blocks}

    pool_c = new_pool()
    set_adoption(path, True)
    sleep_wake(new_scheduler(pool_c))

    got = {b.block_id: b.block_hash for b in pool_c.blocks if b.block_hash}
    assert got == stale, "if this now differs, the gap may be closed -- rewrite this"


def test_disabled_without_the_env_var(tmp_path, monkeypatch):
    monkeypatch.delenv("GMS_KV_INDEX_PATH", raising=False)
    pool = new_pool()
    sleep_wake(new_scheduler(pool))
    assert not hasattr(pool, "_dyn_mirror")


def test_a_stale_rank_answer_is_not_reused(path):
    """The sentinel is consumed, so a rank that fails to publish reads as "no".

    This is the failure that would turn a miss into a wrong answer: engine A
    adopts and says so; engine B builds a FRESH pool but never publishes its own
    answer. If A's file survived, B would replay A's labels onto pages B
    allocated itself.
    """
    pool_a = new_pool()
    set_adoption(path, True)
    sleep_wake(new_scheduler(pool_a))
    populate(pool_a)
    assert not os.path.exists(f"{path}.rank0"), "the answer must be consumed"

    # engine B wakes having built its own pool, and publishes nothing at all.
    pool_b = new_pool()
    sleep_wake(new_scheduler(pool_b))

    assert not any(
        b.block_hash for b in pool_b.blocks
    ), "a stale adoption answer was reused"
