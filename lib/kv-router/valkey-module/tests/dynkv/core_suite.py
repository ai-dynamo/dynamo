# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import struct
import tempfile
from pathlib import Path

from .fences import (
    _assert_full_worker_two,
    _assert_generation_fence_state,
    _assert_state,
    _assert_worker_wide_fence_state,
    _exercise_generation_fence,
    _exercise_worker_wide_fences,
)
from .support import (
    CLEAR,
    REMOVE,
    STORE,
    WIRE_VERSION,
    _command,
    _event,
    _find_free_port,
    _match,
    _select,
    _select_request,
    _start,
    _stop,
)

def run(server: str, module: str) -> None:
    with tempfile.TemporaryDirectory(prefix="dynkv-module-") as path:
        directory = Path(path)
        port = _find_free_port()
        # Run the RDB durability path without an AOF, so restart necessarily
        # exercises the module type's version-10 rdb_load callback.
        process = _start(server, module, directory, port, appendonly=False)
        try:
            try:
                _command(
                    port, b"DYNKV.RESTORE", b"router-index", b"not-an-aof-snapshot"
                )
            except RuntimeError as error:
                assert "DYNKV_RESTORE_LOADING_ONLY" in str(error)
            else:
                raise AssertionError(
                    "DYNKV.RESTORE must not be a client rollback command"
                )
            malformed = _event(STORE, 9, 0, 1, blocks=[(201, 21), (202, 22)])[:-1]
            try:
                _command(port, b"DYNKV.APPLY", b"invalid-event-index", malformed)
            except RuntimeError as error:
                assert "DYNKV_INVALID_EVENT" in str(error)
            else:
                raise AssertionError("truncated store event must fail")
            assert (
                _match(
                    _command(
                        port,
                        b"DYNKV.MATCH",
                        b"invalid-event-index",
                        struct.pack("!BIQ", WIRE_VERSION, 1, 21),
                    )
                )
                == {}
            )

            for malformed in (
                _event(CLEAR, 9, 0, 1) + b"trailing",
                _event(REMOVE, 9, 0, 1, blocks=[(201, 0)]) + b"trailing",
                _event(99, 9, 0, 1),
            ):
                try:
                    _command(port, b"DYNKV.APPLY", b"invalid-event-index", malformed)
                except RuntimeError as error:
                    assert "DYNKV_INVALID_EVENT" in str(error)
                else:
                    raise AssertionError("malformed event must fail")
            assert (
                _match(
                    _command(
                        port,
                        b"DYNKV.MATCH",
                        b"invalid-event-index",
                        struct.pack("!BIQ", WIRE_VERSION, 1, 21),
                    )
                )
                == {}
            )

            root = [(101, 11), (102, 12)]
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 1, 0, 1, blocks=root),
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 2, 0, 1, blocks=root),
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 1, 1, 100, blocks=root),
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 1, 0, 1, blocks=root),
                )
                == b"NOOP"
            )

            request = struct.pack("!BIQQ", WIRE_VERSION, 2, 11, 12)
            result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
            assert result == {(1, 0): (2, 102), (1, 1): (2, 102), (2, 0): (2, 102)}, (
                result
            )

            # The module-side selector accepts the frontend-filtered candidate
            # set and ranks longest prefix overlap first, then lower load.
            selected = _select(
                _command(
                    port,
                    b"DYNKV.SELECT",
                    b"router-index",
                    _select_request([11, 12], [(1, 0, 100), (1, 1, 10), (2, 0, 0)]),
                )
            )
            assert selected == (2, 0, 2, 102)
            selected = _select(
                _command(
                    port,
                    b"DYNKV.SELECT",
                    b"router-index",
                    _select_request([11, 12], [(1, 1, 10), (3, 0, 0)]),
                )
            )
            assert selected == (1, 1, 2, 102)

            try:
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 3, 0, 1, parent=101, blocks=[(103, 13)]),
                )
            except RuntimeError as error:
                assert "DYNKV_MISSING_PARENT" in str(error)
            else:
                raise AssertionError("store without an active worker parent must fail")

            assert (
                _command(port, b"DYNKV.APPLY", b"router-index", _event(CLEAR, 1, 0, 2))
                == b"OK"
            )
            result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
            assert result == {(2, 0): (2, 102)}, result

            # A direct-worker clear and the frontend's mirrored copy can be
            # delivered after a sibling DP rank has recovered. The duplicate
            # clear must not wipe that newer sibling state.
            assert (
                _command(
                    port,
                    b"DYNKV.RESET_WORKER",
                    b"router-index",
                    struct.pack("!Q", 1),
                    struct.pack("!I", 1),
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 1, 1, 1, blocks=root),
                )
                == b"OK"
            )
            result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
            assert result == {(1, 1): (2, 102), (2, 0): (2, 102)}, result
            assert (
                _command(port, b"DYNKV.APPLY", b"router-index", _event(CLEAR, 1, 0, 2))
                == b"NOOP"
            )
            result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
            assert result == {(1, 1): (2, 102), (2, 0): (2, 102)}, result
            assert (
                _command(port, b"DYNKV.APPLY", b"router-index", _event(CLEAR, 1, 0, 3))
                == b"OK"
            )
            result = _match(_command(port, b"DYNKV.MATCH", b"router-index", request))
            assert result == {(2, 0): (2, 102)}, result

            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(REMOVE, 2, 0, 2, blocks=[(102, 0)]),
                )
                == b"OK"
            )
            _assert_state(port)

            # A retired rank must ignore delayed events until its next worker
            # tree dump explicitly resets it.
            assert (
                _command(
                    port,
                    b"DYNKV.REMOVE_WORKER",
                    b"router-index",
                    struct.pack("!Q", 2),
                    struct.pack("!I", 0),
                )
                == b"OK"
            )
            assert (
                _match(_command(port, b"DYNKV.MATCH", b"router-index", request)) == {}
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 2, 0, 1, blocks=root),
                )
                == b"OK"
            )
            assert (
                _match(_command(port, b"DYNKV.MATCH", b"router-index", request)) == {}
            )

            # Worker tree recovery resets the rank before replaying a dump
            # whose event IDs can be lower than prior live events.
            assert (
                _command(
                    port,
                    b"DYNKV.RESET_WORKER",
                    b"router-index",
                    struct.pack("!Q", 2),
                    struct.pack("!I", 0),
                )
                == b"OK"
            )
            assert (
                _command(
                    port,
                    b"DYNKV.APPLY",
                    b"router-index",
                    _event(STORE, 2, 0, 1, blocks=root),
                )
                == b"OK"
            )
            _assert_full_worker_two(port)
            _exercise_generation_fence(port)
            _exercise_worker_wide_fences(port)
            assert _command(port, b"SAVE") == b"OK"
        finally:
            _stop(process, port)

        process = _start(server, module, directory, port, appendonly=False)
        try:
            _assert_full_worker_two(port)
            _assert_generation_fence_state(port)
            _assert_worker_wide_fence_state(port)
        finally:
            _stop(process, port)
