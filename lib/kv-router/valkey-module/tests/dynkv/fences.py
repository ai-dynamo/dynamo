# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import struct
import time

from .support import (
    CLEAR,
    REMOVE,
    STORE,
    WIRE_VERSION,
    _command,
    _event,
    _match,
    _rank_generation,
    _replace_rank_if_generation,
)


__all__ = (
    "_assert_state",
    "_assert_full_worker_two",
    "_wait_for_replica",
    "_wait_for_aof_rewrite",
    "_assert_generation_fence_state",
    "_assert_worker_wide_fence_state",
    "_exercise_generation_fence",
    "_assert_stale_replace",
    "_exercise_worker_wide_fences",
)

def _assert_state(port: int) -> None:
    request = struct.pack("!BIQQ", WIRE_VERSION, 2, 11, 12)
    result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
    assert result == {(2, 0): (1, 101)}, result


def _assert_full_worker_two(port: int) -> None:
    request = struct.pack("!BIQQ", WIRE_VERSION, 2, 11, 12)
    result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
    assert result == {(2, 0): (2, 102)}, result


def _wait_for_replica(port: int) -> None:
    for _ in range(100):
        try:
            info = _command(port, b"INFO", b"replication").decode()
        except (OSError, RuntimeError):
            time.sleep(0.05)
            continue
        if "master_link_status:up" in info:
            return
        time.sleep(0.05)
    raise RuntimeError("Valkey replica did not finish initial synchronization")


def _wait_for_aof_rewrite(port: int) -> None:
    """Wait for the module AOF callback's rewritten base file to be installed."""
    response = _command(port, b"BGREWRITEAOF")
    assert response.startswith(b"Background append only file rewriting")
    # The background child can finish before the first INFO sample. Waiting a
    # short interval avoids treating the command-acceptance window as a finish.
    time.sleep(0.05)
    for _ in range(100):
        info = _command(port, b"INFO", b"persistence").decode()
        if "aof_rewrite_in_progress:0" in info and "aof_rewrite_scheduled:0" in info:
            return
        time.sleep(0.05)
    raise RuntimeError("Valkey AOF rewrite did not finish")


def _assert_generation_fence_state(port: int) -> None:
    key = b"generation-fence-index"
    request = struct.pack("!BIQQ", WIRE_VERSION, 2, 71, 72)
    assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {(77, 3): (1, 701)}
    assert _rank_generation(port, key, 77, 3) == 3


def _assert_worker_wide_fence_state(port: int) -> None:
    worker = 88
    assert _rank_generation(port, b"clear-fence-index", worker, 1) == 1
    assert _rank_generation(port, b"remove-fence-index", worker, 2) == 1
    assert (
        _command(
            port, b"DYNKV.APPLY", b"clear-fence-index", _event(CLEAR, worker, 0, 0)
        )
        == b"NOOP"
    )

    # The clear emitter's dedupe watermark survives persistence and prevents a
    # delayed duplicate from erasing the restored rank snapshot.
    duplicate_key = b"duplicate-clear-fence-index"
    assert (
        _command(port, b"DYNKV.APPLY", duplicate_key, _event(CLEAR, worker, 0, 2))
        == b"NOOP"
    )
    request = struct.pack("!BIQ", WIRE_VERSION, 1, 82)
    assert _match(_command(port, b"DYNKV.MATCH", duplicate_key, request)) == {
        (worker, 0): (1, 882)
    }


def _exercise_generation_fence(port: int) -> None:
    """A dump captured before a direct APPLY cannot erase that live update."""
    key = b"generation-fence-index"
    worker = 77
    dp_rank = 3
    root = _event(STORE, worker, dp_rank, 1, blocks=[(701, 71)])
    child = _event(STORE, worker, dp_rank, 2, parent=701, blocks=[(702, 72)])

    assert _command(port, b"DYNKV.APPLY", key, root) == b"OK"
    captured_generation = _rank_generation(port, key, worker, dp_rank)
    assert captured_generation == 1

    # The worker returns this dump while a newer direct worker event reaches
    # Valkey. A legacy RESET_WORKER + replay would erase block 702.
    stale_dump = [root]
    assert _command(port, b"DYNKV.APPLY", key, child) == b"OK"
    assert _rank_generation(port, key, worker, dp_rank) == 2
    try:
        _replace_rank_if_generation(
            port, key, worker, dp_rank, captured_generation, stale_dump
        )
    except RuntimeError as error:
        assert "DYNKV_STALE_GENERATION" in str(error)
    else:
        raise AssertionError("stale tree dump must be fenced")

    request = struct.pack("!BIQQ", WIRE_VERSION, 2, 71, 72)
    assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
        (worker, dp_rank): (2, 702)
    }

    # A fresh dump may replace the rank atomically. The replacement itself is
    # a new fence generation, so older retrying dumps remain stale.
    assert _replace_rank_if_generation(port, key, worker, dp_rank, 2, stale_dump) == 3
    assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
        (worker, dp_rank): (1, 701)
    }
    assert _rank_generation(port, key, worker, dp_rank) == 3

    # The command validates all embedded events before reset. A wrong-rank
    # event or a clear barrier must not alter either the rank or its fence.
    for invalid_events in (
        [_event(STORE, worker, dp_rank + 1, 3, blocks=[(703, 73)])],
        [_event(CLEAR, worker, dp_rank, 3)],
    ):
        try:
            _replace_rank_if_generation(port, key, worker, dp_rank, 3, invalid_events)
        except RuntimeError as error:
            assert "DYNKV_INVALID_REPLACE" in str(error)
        else:
            raise AssertionError("invalid replacement dump must fail")
        assert _rank_generation(port, key, worker, dp_rank) == 3
        assert _match(_command(port, b"DYNKV.MATCH", key, request)) == {
            (worker, dp_rank): (1, 701)
        }


def _assert_stale_replace(
    port: int,
    key: bytes,
    worker: int,
    dp_rank: int,
    expected_generation: int,
    events: list[bytes],
) -> None:
    try:
        _replace_rank_if_generation(
            port, key, worker, dp_rank, expected_generation, events
        )
    except RuntimeError as error:
        assert "DYNKV_STALE_GENERATION" in str(error)
    else:
        raise AssertionError("replacement must be fenced")


def _exercise_worker_wide_fences(port: int) -> None:
    """CLEAR and no-owner REMOVE must also fence recovery snapshots."""
    worker = 88
    root = _event(STORE, worker, 1, 1, blocks=[(881, 81)])

    # A worker-wide CLEAR applies to an unseen sibling rank too. Its opaque
    # token must change even though that rank has no local WorkerState yet.
    clear_key = b"clear-fence-index"
    assert _rank_generation(port, clear_key, worker, 1) == 0
    # Event IDs begin at zero in the ZMQ publisher, so zero is a real clear
    # barrier rather than an "unseen" sentinel.
    assert (
        _command(port, b"DYNKV.APPLY", clear_key, _event(CLEAR, worker, 0, 0)) == b"OK"
    )
    assert _rank_generation(port, clear_key, worker, 1) == 1
    _assert_stale_replace(port, clear_key, worker, 1, 0, [root])

    # Reset/replacement must retain the clear event's duplicate barrier. A
    # delayed mirrored CLEAR is a NOOP rather than wiping recovered state.
    duplicate_key = b"duplicate-clear-fence-index"
    rank = 0
    duplicate_root = _event(STORE, worker, rank, 1, blocks=[(882, 82)])
    clear = _event(CLEAR, worker, rank, 2)
    assert _command(port, b"DYNKV.APPLY", duplicate_key, duplicate_root) == b"OK"
    assert _command(port, b"DYNKV.APPLY", duplicate_key, clear) == b"OK"
    assert (
        _replace_rank_if_generation(
            port, duplicate_key, worker, rank, 2, [duplicate_root]
        )
        == 3
    )
    assert _command(port, b"DYNKV.APPLY", duplicate_key, clear) == b"NOOP"
    request = struct.pack("!BIQ", WIRE_VERSION, 1, 82)
    assert _match(_command(port, b"DYNKV.MATCH", duplicate_key, request)) == {
        (worker, rank): (1, 882)
    }

    # A remove can reach Valkey before the matching store. It has no prefix
    # mutation to make locally, but it still must invalidate an older dump.
    remove_key = b"remove-fence-index"
    assert _rank_generation(port, remove_key, worker, 2) == 0
    assert (
        _command(
            port,
            b"DYNKV.APPLY",
            remove_key,
            _event(REMOVE, worker, 2, 1, blocks=[(883, 0)]),
        )
        == b"OK"
    )
    assert _rank_generation(port, remove_key, worker, 2) == 1
    _assert_stale_replace(
        port,
        remove_key,
        worker,
        2,
        0,
        [_event(STORE, worker, 2, 1, blocks=[(883, 83)])],
    )

    # Converter-produced empty removes are intentional no-ops and do not need
    # to churn a replicated fence ticket.
    zero_remove_key = b"zero-remove-fence-index"
    assert (
        _command(
            port,
            b"DYNKV.APPLY",
            zero_remove_key,
            _event(REMOVE, worker, 3, 0, blocks=[]),
        )
        == b"NOOP"
    )
    assert _rank_generation(port, zero_remove_key, worker, 3) == 0
