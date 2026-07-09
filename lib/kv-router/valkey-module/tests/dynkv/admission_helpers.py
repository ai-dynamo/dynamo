# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import threading
import time

from .support import (
    ADMISSION_NO_CAPACITY,
    ADMISSION_RESERVED,
    ADMISSION_VERSION,
    STORE,
    _admission_stats,
    _admission_workers,
    _command,
    _event,
    _rank_generation,
    _register_worker,
    _release,
    _renew,
    _replace_rank_if_generation,
    _reserve,
    _reserve_request,
)


__all__ = ("_exercise_admission",)

def _exercise_admission(port: int) -> tuple[bytes, bytes, dict[str, int], int]:
    """Exercise authoritative capacity and lease semantics on one primary."""
    key = b"admission-index"
    domain = b"prefill"
    candidates = [(810, 0, 2), (811, 0, 1)]
    _admission_workers(port, key)
    assert _register_worker(port, key, 810, 0) == b"NOOP"

    # A KV event creates index ownership, but cannot itself make a rank
    # eligible for authoritative admission.
    assert (
        _command(
            port,
            b"DYNKV.APPLY",
            key,
            _event(STORE, 812, 0, 1, blocks=[(8101, 81), (8102, 82)]),
        )
        == b"OK"
    )
    gate_payload = _reserve_request(
        b"registration-gate", 901, 902, 60_000, [81, 82], [(812, 0, 1)]
    )
    gate_raw, gate = _reserve(port, key, gate_payload)
    assert gate is None
    assert gate_raw == bytes((ADMISSION_VERSION, ADMISSION_NO_CAPACITY))
    assert _register_worker(port, key, 812, 0) == b"OK"
    gate_raw, gate = _reserve(port, key, gate_payload)
    assert gate_raw[1] == ADMISSION_RESERVED and gate is not None
    assert _release(port, key, b"registration-gate", 901, 902, gate["expires"]) == 1

    # One stale discovery candidate must not poison a healthy rank from the
    # same frontend snapshot.
    stale_raw, stale = _reserve(
        port,
        key,
        _reserve_request(
            b"stale-candidate", 903, 904, 60_000, [81, 82], [(999, 0, 1), (810, 0, 2)]
        ),
    )
    assert stale_raw[1] == ADMISSION_RESERVED and stale is not None
    assert (stale["worker"], stale["dp_rank"]) == (810, 0)
    assert (
        _release(
            port,
            key,
            b"stale-candidate",
            stale["client"],
            stale["request"],
            stale["expires"],
        )
        == 1
    )
    try:
        _command(port, b"DYNKV.ADMIT_APPLY", key, b"not-internal")
    except RuntimeError as error:
        assert "DYNKV_ADMIT_APPLY_INTERNAL_ONLY" in str(error)
    else:
        raise AssertionError("internal admission apply must reject client calls")

    # Three frontend-like clients reserve concurrently. Valkey serializes the
    # bookings, so the module-owned capacities (2 + 1) admit exactly three.
    barrier = threading.Barrier(3)
    responses: list[tuple[bytes, dict[str, int] | None] | None] = [None, None, None]
    failures: list[BaseException] = []

    def reserve_from_frontend(slot: int) -> None:
        try:
            payload = _reserve_request(
                domain, 1000 + slot, 2000 + slot, 60_000, [81, 82], candidates
            )
            barrier.wait(timeout=2)
            responses[slot] = _reserve(port, key, payload)
        except BaseException as error:  # pragma: no cover - asserted below
            failures.append(error)

    threads = [
        threading.Thread(target=reserve_from_frontend, args=(slot,))
        for slot in range(3)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert not failures, failures
    assert all(response is not None for response in responses)
    reservations = [response[1] for response in responses if response is not None]
    assert all(reservation is not None for reservation in reservations)
    admitted = [reservation for reservation in reservations if reservation is not None]
    assert sorted(
        (reservation["worker"], reservation["dp_rank"]) for reservation in admitted
    ) == [
        (810, 0),
        (810, 0),
        (811, 0),
    ]
    assert sorted(
        reservation["active"]
        for reservation in admitted
        if reservation["worker"] == 810
    ) == [1, 2]
    assert all(reservation["matched"] == 2 for reservation in admitted)
    assert _admission_stats(port, key) == 3

    for invalid_candidates in ([(810, 0, 3)], [(810, 0, 0)]):
        try:
            _reserve(
                port,
                key,
                _reserve_request(
                    domain, 1010, 2010, 60_000, [81, 82], invalid_candidates
                ),
            )
        except RuntimeError as error:
            assert "DYNKV_INVALID_CANDIDATE" in str(error)
        else:
            raise AssertionError("invalid or capacity-conflicting rank must fail")
    unknown_raw, unknown = _reserve(
        port,
        key,
        _reserve_request(domain, 1010, 2010, 60_000, [81, 82], [(999, 0, 1)]),
    )
    assert unknown is None
    assert unknown_raw == bytes((ADMISSION_VERSION, ADMISSION_NO_CAPACITY))
    try:
        _reserve(
            port,
            key,
            _reserve_request(
                b"max-capacity",
                1011,
                2011,
                60_000,
                [81, 82],
                [(810, 0, (1 << 32) - 1)],
            ),
        )
    except RuntimeError as error:
        assert "DYNKV_INVALID_CANDIDATE" in str(error)
    else:
        raise AssertionError("reserved legacy capacity sentinel must be rejected")

    no_capacity_payload = _reserve_request(
        domain, 1004, 2004, 60_000, [81, 82], candidates
    )
    no_capacity_raw, no_capacity = _reserve(port, key, no_capacity_payload)
    assert no_capacity is None and no_capacity_raw == bytes(
        (ADMISSION_VERSION, ADMISSION_NO_CAPACITY)
    )

    # Same nonce + exact request returns the original admission without taking
    # another slot; a conflicting replay is rejected.
    duplicate_raw, duplicate = _reserve(
        port,
        key,
        _reserve_request(domain, 1000, 2000, 60_000, [81, 82], candidates),
    )
    assert duplicate_raw == responses[0][0]
    assert duplicate == admitted[0]
    assert _admission_stats(port, key) == 3
    try:
        _reserve(
            port,
            key,
            _reserve_request(domain, 1000, 2000, 59_999, [81, 82], candidates),
        )
    except RuntimeError as error:
        assert "DYNKV_REQUEST_CONFLICT" in str(error)
    else:
        raise AssertionError("same nonce with a conflicting request must fail")

    # Release frees one slot and is idempotent. Renew changes the lease token;
    # a stale release must not release the renewed reservation.
    released = admitted[0]
    assert (
        _release(
            port,
            key,
            domain,
            released["client"],
            released["request"],
            released["expires"],
        )
        == 1
    )
    assert (
        _release(
            port,
            key,
            domain,
            released["client"],
            released["request"],
            released["expires"],
        )
        == 0
    )
    assert _admission_stats(port, key) == 2
    replacement_raw, replacement = _reserve(port, key, no_capacity_payload)
    assert replacement_raw[1] == ADMISSION_RESERVED and replacement is not None
    renewed_raw, renewed = _renew(
        port,
        key,
        domain,
        replacement["client"],
        replacement["request"],
        replacement["expires"],
        90_000,
    )
    assert renewed_raw[1] == ADMISSION_RESERVED
    assert renewed["matched"] == 2
    assert renewed["expires"] != replacement["expires"]
    renewed_retry_raw, renewed_retry = _renew(
        port,
        key,
        domain,
        replacement["client"],
        replacement["request"],
        replacement["expires"],
        90_000,
    )
    assert renewed_retry_raw == renewed_raw
    assert renewed_retry == renewed
    try:
        _release(
            port,
            key,
            domain,
            replacement["client"],
            replacement["request"],
            replacement["expires"],
        )
    except RuntimeError as error:
        assert "DYNKV_RESERVATION_EXPIRED" in str(error)
    else:
        raise AssertionError("stale lease token must not release a renewed reservation")

    # Semantic domain separates prefill/decode request identities even when
    # nonce values are the same.
    for reservation in admitted[1:]:
        assert (
            _release(
                port,
                key,
                domain,
                reservation["client"],
                reservation["request"],
                reservation["expires"],
            )
            == 1
        )
    assert (
        _release(
            port, key, domain, renewed["client"], renewed["request"], renewed["expires"]
        )
        == 1
    )
    assert _admission_stats(port, key) == 0
    prefill_payload = _reserve_request(domain, 3000, 4000, 60_000, [81, 82], candidates)
    decode_payload = _reserve_request(
        b"decode", 3000, 4000, 60_000, [81, 82], candidates
    )
    prefill_raw, prefill = _reserve(port, key, prefill_payload)
    decode_raw, decode = _reserve(port, key, decode_payload)
    assert prefill is not None and decode is not None
    # Capacity and load are domain scoped: neither request consumes the
    # other domain's first slot, even when the client/request nonces match.
    assert prefill["worker"] == decode["worker"] == 810
    assert prefill["active"] == decode["active"] == 1
    assert _admission_stats(port, key) == 2
    assert _release(port, key, domain, 3000, 4000, prefill["expires"]) == 1
    assert _release(port, key, b"decode", 3000, 4000, decode["expires"]) == 1

    # Expiry is reaped by the next authoritative mutation, then the same
    # request identity is safe to admit again.
    short_payload = _reserve_request(domain, 5000, 6000, 20, [81, 82], candidates)
    short_raw, short = _reserve(port, key, short_payload)
    assert short_raw[1] == ADMISSION_RESERVED and short is not None
    time.sleep(0.06)
    expiry_trigger_payload = _reserve_request(
        domain, 5001, 6001, 60_000, [81, 82], candidates
    )
    _, expiry_trigger = _reserve(port, key, expiry_trigger_payload)
    assert expiry_trigger is not None
    assert _admission_stats(port, key) == 1
    assert (
        _release(port, key, domain, short["client"], short["request"], short["expires"])
        == 0
    )
    assert (
        _release(
            port,
            key,
            domain,
            expiry_trigger["client"],
            expiry_trigger["request"],
            expiry_trigger["expires"],
        )
        == 1
    )

    # An exact retry is recoverable even if an unrelated candidate was
    # retired after the original grant.
    retry_payload = _reserve_request(domain, 6100, 6200, 60_000, [81, 82], candidates)
    retry_raw, retry = _reserve(port, key, retry_payload)
    assert retry is not None and retry["worker"] == 810
    assert (
        _command(
            port,
            b"DYNKV.REMOVE_WORKER",
            key,
            struct.pack("!Q", 811),
            struct.pack("!I", 0),
        )
        == b"OK"
    )
    retry_replay_raw, retry_replay = _reserve(port, key, retry_payload)
    assert retry_replay_raw == retry_raw
    assert retry_replay == retry
    assert (
        _command(
            port,
            b"DYNKV.RESET_WORKER",
            key,
            struct.pack("!Q", 811),
            struct.pack("!I", 0),
        )
        == b"OK"
    )
    assert _register_worker(port, key, 811, 0) == b"OK"
    assert (
        _release(port, key, domain, retry["client"], retry["request"], retry["expires"])
        == 1
    )

    # Retirement revokes live reservations and prevents stale register or
    # admission attempts. An all-ranks retirement also tombstones unseen ranks.
    active_raw, active = _reserve(
        port, key, _reserve_request(domain, 7000, 8000, 60_000, [81, 82], candidates)
    )
    assert active_raw[1] == ADMISSION_RESERVED and active is not None
    assert (
        _command(
            port,
            b"DYNKV.REMOVE_WORKER",
            key,
            struct.pack("!Q", active["worker"]),
            struct.pack("!I", active["dp_rank"]),
        )
        == b"OK"
    )
    assert _admission_stats(port, key) == 0
    retired_raw, retired = _reserve(
        port,
        key,
        _reserve_request(
            domain, 7001, 8001, 60_000, [81, 82], [(active["worker"], 0, 2)]
        ),
    )
    assert retired is None
    assert retired_raw == bytes((ADMISSION_VERSION, ADMISSION_NO_CAPACITY))
    try:
        _register_worker(port, key, active["worker"], active["dp_rank"])
    except RuntimeError as error:
        assert "DYNKV_WORKER_RETIRED" in str(error)
    else:
        raise AssertionError("retired rank registration must fail")
    # RESET also revokes every domain-scoped lease for the rank before it
    # permits the recovered rank to accept new admissions.
    assert (
        _command(
            port,
            b"DYNKV.RESET_WORKER",
            key,
            struct.pack("!Q", active["worker"]),
            struct.pack("!I", active["dp_rank"]),
        )
        == b"OK"
    )
    assert _register_worker(port, key, active["worker"], active["dp_rank"]) == b"OK"
    reset_capacity = 2 if active["worker"] == 810 else 1
    reset_live_raw, reset_live = _reserve(
        port,
        key,
        _reserve_request(
            b"reset-domain",
            7100,
            8100,
            60_000,
            [81, 82],
            [(active["worker"], active["dp_rank"], reset_capacity)],
        ),
    )
    assert reset_live_raw[1] == ADMISSION_RESERVED and reset_live is not None
    assert (
        _command(
            port,
            b"DYNKV.RESET_WORKER",
            key,
            struct.pack("!Q", active["worker"]),
            struct.pack("!I", active["dp_rank"]),
        )
        == b"OK"
    )
    assert _admission_stats(port, key) == 0

    # A generation-fenced tree replacement has the same lease-revocation
    # behavior as a reset for every admission domain on the replaced rank.
    replace_live_raw, replace_live = _reserve(
        port,
        key,
        _reserve_request(
            b"replace-domain",
            7150,
            8150,
            60_000,
            [81, 82],
            [(active["worker"], active["dp_rank"], reset_capacity)],
        ),
    )
    assert replace_live_raw[1] == ADMISSION_RESERVED and replace_live is not None
    replace_generation = _rank_generation(
        port, key, active["worker"], active["dp_rank"]
    )
    assert (
        _replace_rank_if_generation(
            port,
            key,
            active["worker"],
            active["dp_rank"],
            replace_generation,
            [
                _event(
                    STORE,
                    active["worker"],
                    active["dp_rank"],
                    1,
                    blocks=[(8101, 81), (8102, 82)],
                )
            ],
        )
        > replace_generation
    )
    assert _admission_stats(port, key) == 0

    # An all-ranks retirement revokes a live reservation and then tombstones
    # ranks that have not been registered yet.
    all_retired_worker = 811
    all_retire_raw, all_retire = _reserve(
        port,
        key,
        _reserve_request(b"all-retire", 7200, 8200, 60_000, [81, 82], [(811, 0, 1)]),
    )
    assert all_retire_raw[1] == ADMISSION_RESERVED and all_retire is not None
    assert (
        _command(
            port, b"DYNKV.REMOVE_WORKER_ALL", key, struct.pack("!Q", all_retired_worker)
        )
        == b"OK"
    )
    assert _admission_stats(port, key) == 0
    try:
        _register_worker(port, key, all_retired_worker, 7)
    except RuntimeError as error:
        assert "DYNKV_WORKER_RETIRED" in str(error)
    else:
        raise AssertionError("all-ranks retirement must block an unseen rank")

    # Return one durable admission payload/response for restart assertions.
    assert (
        _command(
            port,
            b"DYNKV.RESET_WORKER",
            key,
            struct.pack("!Q", 810),
            struct.pack("!I", 0),
        )
        == b"OK"
    )
    persistent_payload = _reserve_request(
        domain, 9000, 9001, 60_000, [81, 82], [(810, 0, 2)]
    )
    persistent_raw, persistent = _reserve(port, key, persistent_payload)
    assert persistent is not None
    persistent_initial_expires = persistent["expires"]
    persistent_raw, persistent = _renew(
        port,
        key,
        domain,
        persistent["client"],
        persistent["request"],
        persistent_initial_expires,
        120_000,
    )
    return persistent_payload, persistent_raw, persistent, persistent_initial_expires
