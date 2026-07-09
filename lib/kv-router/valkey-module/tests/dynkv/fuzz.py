# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic malformed-wire fuzz coverage for public module parsers."""

from __future__ import annotations

import random
import struct
import tempfile
from pathlib import Path

from .support import (
    ADMISSION_VERSION,
    CLEAR,
    REGISTRATION_VERSION,
    REMOVE,
    STORE,
    WIRE_VERSION,
    WORKER_LEASE_CONTROL_VERSION,
    _command,
    _event,
    _find_free_port,
    _registration_ranks_payload,
    _release_request,
    _renew_request,
    _replace_snapshot,
    _reserve_request,
    _select_request,
    _start,
    _stop,
    _worker_lease_control_payload,
)


def _expect_rejected(port: int, command: bytes, key: bytes, *arguments: bytes) -> None:
    try:
        _command(port, command, key, *arguments)
    except RuntimeError:
        return
    raise AssertionError(f"malformed input unexpectedly accepted by {command!r}")


def run_malformed_wire_fuzz(server: str, module: str) -> None:
    rng = random.Random(0xD1A0_CAFE)
    commands = (
        (b"DYNKV.APPLY", ()),
        (b"DYNKV.MATCH", ()),
        (b"DYNKV.SELECT", ()),
        (b"DYNKV.SELECT_RESERVE", ()),
        (b"DYNKV.RELEASE", ()),
        (b"DYNKV.RENEW", ()),
        (b"DYNKV.REGISTER_WORKER_RANKS", ((1).to_bytes(8, "big"),)),
    )
    valid_version_commands = (
        (b"DYNKV.APPLY", (), WIRE_VERSION),
        (b"DYNKV.MATCH", (), WIRE_VERSION),
        (b"DYNKV.SELECT", (), WIRE_VERSION),
        (b"DYNKV.SELECT_RESERVE", (), ADMISSION_VERSION),
        (b"DYNKV.RELEASE", (), ADMISSION_VERSION),
        (b"DYNKV.RENEW", (), ADMISSION_VERSION),
        (
            b"DYNKV.REGISTER_WORKER_RANKS",
            (struct.pack("!Q", 1),),
            REGISTRATION_VERSION,
        ),
        (b"DYNKV.RENEW_WORKER_LEASE", (), WORKER_LEASE_CONTROL_VERSION),
        (b"DYNKV.UNREGISTER_WORKER", (), WORKER_LEASE_CONTROL_VERSION),
        (
            b"DYNKV.REPLACE_RANK_IF_GENERATION",
            (struct.pack("!Q", 1), struct.pack("!I", 0), struct.pack("!Q", 0)),
            WIRE_VERSION,
        ),
    )
    valid_envelopes = (
        (b"DYNKV.APPLY", (), _event(STORE, 1, 0, 1, blocks=[(11, 12)])),
        (b"DYNKV.MATCH", (), struct.pack("!BIQ", WIRE_VERSION, 1, 11)),
        (b"DYNKV.SELECT", (), _select_request([11], [(1, 0, 4)])),
        (
            b"DYNKV.SELECT_RESERVE",
            (),
            _reserve_request(b"fuzz", 1, 2, 30_000, [11], [(1, 0, 4)]),
        ),
        (b"DYNKV.RELEASE", (), _release_request(b"fuzz", 1, 2, 3)),
        (b"DYNKV.RENEW", (), _renew_request(b"fuzz", 1, 2, 3, 30_000)),
        (
            b"DYNKV.REGISTER_WORKER_RANKS",
            (struct.pack("!Q", 1),),
            _registration_ranks_payload([0, 1]),
        ),
        (
            b"DYNKV.RENEW_WORKER_LEASE",
            (),
            _worker_lease_control_payload(1, 2, 30_000),
        ),
        (
            b"DYNKV.UNREGISTER_WORKER",
            (),
            _worker_lease_control_payload(1, 2),
        ),
        (
            b"DYNKV.REPLACE_RANK_IF_GENERATION",
            (struct.pack("!Q", 1), struct.pack("!I", 0), struct.pack("!Q", 0)),
            _replace_snapshot([_event(STORE, 1, 0, 1, blocks=[(11, 12)])]),
        ),
    )

    with tempfile.TemporaryDirectory(prefix="dynkv-fuzz-") as path:
        port = _find_free_port()
        process = _start(server, module, Path(path), port, appendonly=False)
        key = b"malformed-fuzz-index"
        try:
            for iteration in range(512):
                command, prefix = commands[rng.randrange(len(commands))]
                invalid_version = bytes((0 if iteration % 2 == 0 else 255,))
                payload = invalid_version + rng.randbytes(rng.randrange(256))
                _expect_rejected(port, command, key, *prefix, payload)
                if iteration % 32 == 0:
                    assert _command(port, b"PING") == b"PONG"

            # Start from complete, structurally valid envelopes and cut at
            # every byte boundary. This reaches nested identities, rank lists,
            # candidate lists, lease fields, and recovery snapshots instead of
            # stopping in the version/header prefix.
            for command, prefix, valid in valid_envelopes:
                for cut in range(1, len(valid)):
                    _expect_rejected(port, command, key, *prefix, valid[:cut])
                for suffix_length in range(1, 17):
                    _expect_rejected(
                        port,
                        command,
                        key,
                        *prefix,
                        valid + rng.randbytes(suffix_length),
                    )
                assert _command(port, b"PING") == b"PONG"

            # Invalid-version inputs only exercise the first byte. Keep the
            # public wire version valid and truncate each command before its
            # minimum header completes so admission, registration, lease,
            # recovery, matching, and mutation parsers all run deeper.
            for iteration in range(512):
                command, prefix, version = valid_version_commands[
                    rng.randrange(len(valid_version_commands))
                ]
                payload = bytes((version,)) + rng.randbytes(rng.randrange(4))
                _expect_rejected(port, command, key, *prefix, payload)
                if iteration % 32 == 0:
                    assert _command(port, b"PING") == b"PONG"

            # GC is not versioned. Exercise its public watermark and budget
            # decoders with incorrect widths rather than the internal-only
            # replicated apply command.
            for width in (0, 1, 3, 5, 7, 9, 16):
                _expect_rejected(
                    port,
                    b"DYNKV.GC",
                    key,
                    rng.randbytes(width),
                    struct.pack("!I", 1),
                )
                _expect_rejected(
                    port,
                    b"DYNKV.GC",
                    key,
                    b"CURRENT",
                    rng.randbytes(width),
                )

            event_kinds = (STORE, REMOVE, CLEAR)
            for iteration in range(256):
                kind = event_kinds[iteration % len(event_kinds)]
                blocks = [] if kind == CLEAR else [(iteration + 1, iteration + 17)]
                valid = _event(kind, 1, 0, iteration + 1, blocks=blocks)
                if iteration % 2 == 0:
                    malformed = valid[: rng.randrange(len(valid))]
                else:
                    malformed = valid + rng.randbytes(rng.randrange(1, 17))
                _expect_rejected(port, b"DYNKV.APPLY", key, malformed)

            assert _command(port, b"EXISTS", key) == b"0"
            assert _command(port, b"PING") == b"PONG"
        finally:
            _stop(process, port)
