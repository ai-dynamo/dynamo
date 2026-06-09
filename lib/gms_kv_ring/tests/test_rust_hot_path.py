# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Rust hot path — correctness + speedup vs pure-Python fallback.

Both paths write the same on-disk bytes (verified by reading via
the Python fallback after Rust pushes, and vice versa). The Rust
path is ~3× faster on push/pop.
"""

from __future__ import annotations

import pytest

pytest.importorskip("gms_rust_ring")


def test_rust_extension_loaded():
    """Sanity: the native crate is importable and matches our layout."""
    import gms_rust_ring as r

    assert r.RECORD_SIZE == 512
    assert r.HEADER_SIZE == 64
    assert r.MAX_BLOCK_PAIRS == 54
    assert r.ENGINE_ID_MAX_LEN == 48


def test_rust_path_active_in_restore_ring():
    """Python wrapper auto-selects Rust when available."""
    from gms_kv_ring.common.restore_ring import is_rust_accelerated

    assert is_rust_accelerated() is True


def test_rust_python_layout_parity(tmp_path):
    """Bytes written by Rust must be byte-identical to Python — both
    paths share the same on-disk format. Verified by pushing via Rust
    + popping via Python's fallback (or vice versa)."""
    from gms_kv_ring.common import restore_ring as rr

    path = str(tmp_path / "parity.ring")
    w = rr.create_ring(path, capacity=8)
    r = rr.attach_reader(path)

    # Push via the active path (Rust if loaded).
    pairs = [(1, 101), (2, 102), (3, 103), (4, 104)]
    assert w.push(
        src_engine_id="parity-test",
        block_pairs=pairs,
        counter_slot=5,
        counter_target=7,
    )

    # Pop via PYTHON path explicitly (force the fallback).
    rec = r._try_pop_python()
    assert rec is not None, "Python pop saw nothing — layout mismatch"
    assert rec["src_engine_id"] == "parity-test"
    assert rec["counter_slot"] == 5
    assert rec["counter_target"] == 7
    assert rec["block_pairs"] == pairs

    # Now push via PYTHON path (force fallback) and pop via the
    # active path (Rust if loaded). Bytes must round-trip the
    # other direction too.
    w._push_python(
        src_engine_id="parity-2",
        block_pairs=[(9, 99)],
        counter_slot=2,
        counter_target=3,
    )
    rec = r.try_pop()
    assert rec["src_engine_id"] == "parity-2"
    assert rec["counter_slot"] == 2
    assert rec["counter_target"] == 3
    assert rec["block_pairs"] == [(9, 99)]

    w.close()
    r.close()


def test_rust_speedup_vs_python(tmp_path):
    """Microbench: Rust push/pop must be ≥2× faster than Python.
    Observational — prints numbers; asserts only on direction."""
    import time

    from gms_kv_ring.common import restore_ring as rr
    from gms_kv_ring.common.restore_ring import is_rust_accelerated

    if not is_rust_accelerated():
        pytest.skip("Rust extension not loaded; nothing to compare")

    path_rs = str(tmp_path / "rs.ring")
    path_py = str(tmp_path / "py.ring")
    w_rs = rr.create_ring(path_rs, capacity=2048)
    r_rs = rr.attach_reader(path_rs)
    w_py = rr.create_ring(path_py, capacity=2048)
    r_py = rr.attach_reader(path_py)

    ITERS = 1000
    pairs = [(i, i + 100) for i in range(8)]

    # Rust path
    t0 = time.perf_counter()
    for _ in range(ITERS):
        w_rs.push(
            src_engine_id="x",
            block_pairs=pairs,
            counter_slot=0,
            counter_target=1,
        )
        r_rs.try_pop()
    rs_us = (time.perf_counter() - t0) / ITERS * 1e6

    # Python path
    t0 = time.perf_counter()
    for _ in range(ITERS):
        w_py._push_python(
            src_engine_id="x",
            block_pairs=pairs,
            counter_slot=0,
            counter_target=1,
        )
        r_py._try_pop_python()
    py_us = (time.perf_counter() - t0) / ITERS * 1e6

    speedup = py_us / max(rs_us, 1e-6)
    print(
        f"\n[rust hot-path bench] push+pop\n"
        f"  Rust:   {rs_us:6.2f} µs/iter\n"
        f"  Python: {py_us:6.2f} µs/iter\n"
        f"  Speedup: {speedup:.2f}×"
    )
    assert speedup >= 2.0, f"Rust path should be ≥2× faster; got {speedup:.2f}×"

    w_rs.close()
    r_rs.close()
    w_py.close()
    r_py.close()
