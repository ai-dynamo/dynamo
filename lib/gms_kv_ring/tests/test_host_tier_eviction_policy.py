# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest


@pytest.fixture
def fake_host_tier(monkeypatch):
    from gms_kv_ring.daemon import host_tier as host_mod

    next_ptr = [0x1000]
    freed: list[int] = []

    def fake_alloc(size: int) -> int:
        ptr = next_ptr[0]
        next_ptr[0] += max(1, int(size)) + 0x100
        return ptr

    monkeypatch.setattr(host_mod, "_alloc_host", fake_alloc)
    monkeypatch.setattr(host_mod, "_free_host", lambda ptr: freed.append(ptr))
    return host_mod, freed


def _ready(ht, offset: int, *, size: int = 10, engine: str = "eng") -> int:
    ptr = ht.put(engine, 0, offset, size)
    assert ht.mark_ready(engine, 0, offset, crc=offset + 1)
    return ptr


def test_lru_policy_matches_existing_access_order(fake_host_tier):
    host_mod, freed = fake_host_tier
    ht = host_mod.HostTier(eviction_policy="lru")

    ptr_a = _ready(ht, 0)
    ptr_b = _ready(ht, 10)
    _ready(ht, 20)

    assert ht.get("eng", 0, 0) is not None

    evicted = ht.evict_until_under(20)

    assert evicted == [("eng", 0, 10)]
    assert ht.total_bytes() == 20
    assert ptr_b in freed
    assert ptr_a not in freed


def test_lfu_policy_evicts_coldest_ready_slot(fake_host_tier):
    host_mod, _freed = fake_host_tier
    ht = host_mod.HostTier(eviction_policy="lfu")

    _ready(ht, 0)
    _ready(ht, 10)
    _ready(ht, 20)
    assert ht.get("eng", 0, 0) is not None
    assert ht.get("eng", 0, 0) is not None
    assert ht.get("eng", 0, 10) is not None

    evicted = ht.evict_until_under(20)

    assert evicted == [("eng", 0, 20)]
    assert ht.get("eng", 0, 20) is None


def test_tiny_lfu_policy_preserves_hot_slots(fake_host_tier):
    host_mod, _freed = fake_host_tier
    ht = host_mod.HostTier(eviction_policy="tiny-lfu")

    _ready(ht, 0)
    _ready(ht, 10)
    _ready(ht, 20)
    for _ in range(4):
        assert ht.get("eng", 0, 0) is not None
    assert ht.get("eng", 0, 10) is not None

    evicted = ht.evict_until_under(20)

    assert ht.eviction_policy == "tiny_lfu"
    assert evicted == [("eng", 0, 20)]
    assert ht.get("eng", 0, 0) is not None


def test_eviction_skips_protected_not_ready_and_pinned(fake_host_tier):
    host_mod, freed = fake_host_tier
    ht = host_mod.HostTier(eviction_policy="lru")

    ptr_a = _ready(ht, 0)
    ptr_b = _ready(ht, 10)
    ptr_c = ht.put("eng", 0, 20, 10)
    lease = ht.pin("eng", 0, 10)
    assert lease is not None

    evicted = ht.evict_until_under(0, protected_keys={("eng", 0, 0)})

    assert evicted == []
    assert ht.total_bytes() == 30
    assert freed == []

    lease.release()
    evicted = ht.evict_until_under(10, protected_keys={("eng", 0, 0)})

    assert evicted == [("eng", 0, 10)]
    assert ht.total_bytes() == 20
    assert ptr_b in freed
    assert ptr_a not in freed
    assert ptr_c not in freed


def test_release_defers_free_until_pin_released(fake_host_tier):
    host_mod, freed = fake_host_tier
    ht = host_mod.HostTier()

    ptr = _ready(ht, 0)
    lease = ht.pin("eng", 0, 0)
    assert lease is not None

    assert ht.release_slot("eng", 0, 0) is True
    assert ht.total_bytes() == 0
    assert ht.get("eng", 0, 0) is None
    assert freed == []

    lease.release()

    assert freed == [ptr]


def test_invalid_policy_is_rejected(fake_host_tier):
    host_mod, _freed = fake_host_tier

    with pytest.raises(ValueError):
        host_mod.HostTier(eviction_policy="clock-pro")


def test_stale_write_cannot_publish_over_replacement(fake_host_tier):
    host_mod, freed = fake_host_tier
    ht = host_mod.HostTier()

    first = ht.reserve("eng", 0, 0, 16)
    replacement = ht.reserve("eng", 0, 0, 16)

    assert first.host_ptr != replacement.host_ptr
    assert ht.commit(first, crc=1) is False
    assert first.host_ptr in freed
    assert ht.get("eng", 0, 0) is None

    assert ht.commit(replacement, crc=2) is True
    assert ht.get("eng", 0, 0).crc == 2


def test_stale_abort_does_not_remove_replacement(fake_host_tier):
    host_mod, freed = fake_host_tier
    ht = host_mod.HostTier()

    first = ht.reserve("eng", 0, 0, 16)
    replacement = ht.reserve("eng", 0, 0, 16)

    assert ht.abort(first) is False
    assert first.host_ptr in freed
    assert ht.total_bytes() == 16
    assert ht.commit(replacement, crc=7) is True
    assert ht.get("eng", 0, 0).crc == 7


def test_stale_lease_release_does_not_remove_replacement(fake_host_tier):
    host_mod, freed = fake_host_tier
    ht = host_mod.HostTier()

    old_ptr = _ready(ht, 0, size=16)
    lease = ht.pin("eng", 0, 0)
    assert lease is not None
    replacement = ht.reserve("eng", 0, 0, 16)

    lease.release()
    assert old_ptr in freed
    assert ht.release_if_current(lease) is False
    assert ht.commit(replacement, crc=9) is True
    assert ht.get("eng", 0, 0).crc == 9
