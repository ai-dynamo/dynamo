# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import time

from .support import (
    _admission_stats,
    _register_worker,
    _release,
    _renew,
    _reserve,
    _reserve_request,
)


__all__ = ("_exercise_admission_expiry_heap",)

def _exercise_admission_expiry_heap(port: int) -> None:
    """Exercise expiry ordering, renewal repositioning, and non-root release."""
    key = b"admission-expiry-heap-index"
    domain = b"prefill"
    worker = 920
    candidates = [(worker, 0, 16)]
    assert _register_worker(port, key, worker, 0) == b"OK"

    _, first = _reserve(port, key, _reserve_request(domain, 1, 1, 80, [], candidates))
    _, renewed = _reserve(
        port, key, _reserve_request(domain, 2, 2, 100, [], candidates)
    )
    _, long_lived = _reserve(
        port, key, _reserve_request(domain, 3, 3, 500, [], candidates)
    )
    assert first is not None and renewed is not None and long_lived is not None
    assert _admission_stats(port, key) == 3

    # Move the middle entry past the long-lived one. The next mutation must
    # reap only the first, expired lease, proving renewal updated the index.
    time.sleep(0.03)
    _, renewed = _renew(
        port,
        key,
        domain,
        renewed["client"],
        renewed["request"],
        renewed["expires"],
        900,
    )
    time.sleep(0.10)
    _, trigger = _reserve(
        port, key, _reserve_request(domain, 4, 4, 1_000, [], candidates)
    )
    assert trigger is not None
    assert _admission_stats(port, key) == 3
    assert (
        _release(port, key, domain, first["client"], first["request"], first["expires"])
        == 0
    )

    # These removals hit arbitrary heap positions, not only the earliest
    # deadline, before the final reservation is released.
    assert (
        _release(
            port,
            key,
            domain,
            long_lived["client"],
            long_lived["request"],
            long_lived["expires"],
        )
        == 1
    )
    assert (
        _release(
            port, key, domain, renewed["client"], renewed["request"], renewed["expires"]
        )
        == 1
    )
    assert (
        _release(
            port, key, domain, trigger["client"], trigger["request"], trigger["expires"]
        )
        == 1
    )
    assert _admission_stats(port, key) == 0

    # Expiry storms are drained cooperatively. SELECT_RESERVE performs one
    # fixed 64-item pass before planning rather than monopolizing Valkey's
    # command thread for the whole heap.
    burst_key = b"admission-expiry-burst"
    burst_worker = 921
    burst_count = 512
    burst_candidates = [(burst_worker, 0, 1024)]
    assert _register_worker(port, burst_key, burst_worker, 0) == b"OK"
    for nonce in range(burst_count):
        _, reservation = _reserve(
            port,
            burst_key,
            _reserve_request(
                domain,
                10_000 + nonce,
                20_000 + nonce,
                # Keep every lease alive while the burst is assembled. A
                # sanitizer build can take longer than the old 25 ms lease,
                # which turned this into a test of command-loop speed instead
                # of the bounded 64-item expiry pass below.
                1_000,
                [],
                burst_candidates,
            ),
        )
        assert reservation is not None
    time.sleep(1.05)

    started = time.monotonic()
    _, trigger = _reserve(
        port,
        burst_key,
        _reserve_request(domain, 30_000, 40_000, 60_000, [], burst_candidates),
    )
    assert trigger is not None
    assert time.monotonic() - started < 0.5
    assert _admission_stats(port, burst_key) == burst_count - 64 + 1
