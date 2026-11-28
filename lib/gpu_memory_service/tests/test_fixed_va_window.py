# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Placement inside a fixed-VA window (bookkeeping only, no driver calls)."""

from __future__ import annotations

import random

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.client.memory_manager import _FixedVAWindow

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
    pytest.mark.timeout(10),
]

P = 4096
BASE = 0x1000000


def _assert_disjoint(live):
    spans = sorted(live.items())
    for (a, n), (b, _) in zip(spans, spans[1:]):
        assert a + n <= b, (hex(a), n, hex(b))


def test_client_ranges_grow_down_and_reuse_released_space():
    w = _FixedVAWindow(BASE, 16 * P)
    a = w.carve_top_down(4 * P, P)
    b = w.carve_top_down(4 * P, P)
    assert (a, b) == (BASE + 12 * P, BASE + 8 * P)
    w.release(a)
    assert w.carve_top_down(2 * P, P) == BASE + 14 * P  # top of the hole
    w.release(b)
    assert w.carve_top_down(6 * P, P) == BASE + 8 * P  # coalesced hole
    with pytest.raises(RuntimeError, match="no free"):
        w.carve_top_down(9 * P, P)


def test_slot_imports_and_client_ranges_never_overlap():
    w = _FixedVAWindow(BASE, 16 * P)
    w.carve_top_down(8 * P, P)
    with pytest.raises(RuntimeError, match="do not fit"):
        w.carve(BASE + 6 * P, 4 * P, P)
    assert w.carve(BASE, 8 * P, P) == BASE
    assert w.imports == 1
    with pytest.raises(RuntimeError, match="do not fit"):
        w.carve(BASE + 4 * P, P, P)  # on top of the first import
    with pytest.raises(RuntimeError, match="no free"):
        w.carve_top_down(P, P)


def test_alignment_gap_below_the_client_ranges_never_covers_an_import():
    w = _FixedVAWindow(BASE, 16 * P)
    w.carve(BASE + 13 * P, P, P)
    # Aligned down to 8 pages, this range would leave [9, 16) as free client
    # space with the import inside it.
    with pytest.raises(RuntimeError, match="no free"):
        w.carve_top_down(P, 8 * P)


def test_random_carve_and_release_keep_ranges_disjoint():
    rng = random.Random(20261114)
    w = _FixedVAWindow(BASE, 512 * P)
    live = {}
    for _ in range(3000):
        if live and rng.random() < 0.45:
            va = rng.choice(list(live))
            w.release(va)
            del live[va]
        else:
            size, align = rng.choice([1, 2, 3, 8]) * P, rng.choice([P, 2 * P])
            try:
                if rng.random() < 0.2:
                    va = w.carve(BASE + rng.randrange(64) * align, size, align)
                else:
                    va = w.carve_top_down(size, align)
            except RuntimeError:
                continue
            assert va % align == 0 and w.base <= va and va + size <= w.end
            live[va] = size
        _assert_disjoint(live)
        # Client space [floor, end) is exactly the client ranges plus free holes.
        client = {v: n for v, n in live.items() if v >= w._floor}
        _assert_disjoint({**client, **w._holes})
        assert sum(client.values()) + sum(w._holes.values()) == w.end - w._floor
        assert w.imports == len(live) - len(client)
    for va in list(live):
        w.release(va)
    assert w.is_empty and w.imports == 0
    assert w.carve_top_down(512 * P, P) == w.base  # all space came back
