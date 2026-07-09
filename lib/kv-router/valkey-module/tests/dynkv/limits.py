# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the canonical C wire limits at their max and max-plus-one edges."""

import struct
import tempfile
from pathlib import Path

from .support import (
    ADMISSION_NO_CAPACITY,
    ADMISSION_VERSION,
    WIRE_VERSION,
    _command,
    _find_free_port,
    _match,
    _module_limit,
    _reserve_request,
    _select,
    _select_request,
    _start,
    _stop,
)


def _expect_error(port: int, command: bytes, key: bytes, payload: bytes, marker: str) -> None:
    try:
        _command(port, command, key, payload)
    except RuntimeError as error:
        assert marker in str(error)
    else:
        raise AssertionError(f"{command.decode()} accepted a max-plus-one payload")


def run_wire_limits(server: str, module: str) -> None:
    max_hashes = _module_limit("DYNKV_MAX_MATCH_HASHES")
    max_candidates = _module_limit("DYNKV_MAX_ADMISSION_CANDIDATES")
    assert max_candidates == _module_limit("DYNKV_MAX_SELECT_CANDIDATES")
    max_domain = _module_limit("DYNKV_MAX_ADMISSION_DOMAIN_LENGTH")
    max_lease_ms = _module_limit("DYNKV_MAX_ADMISSION_LEASE_MS")

    with tempfile.TemporaryDirectory(prefix="dynkv-wire-limits-") as path:
        port = _find_free_port()
        process = _start(server, module, Path(path), port, appendonly=False)
        key = b"wire-limit-index"
        try:
            match_payload = struct.pack("!BI", WIRE_VERSION, max_hashes) + (
                struct.pack("!Q", 0) * max_hashes
            )
            assert _match(_command(port, b"DYNKV.MATCH", key, match_payload)) == {}
            _expect_error(
                port,
                b"DYNKV.MATCH",
                key,
                struct.pack("!BI", WIRE_VERSION, max_hashes + 1),
                "DYNKV_INVALID_MATCH",
            )

            candidates = [(worker, 0, 1) for worker in range(1, max_candidates + 1)]
            selected = _select(
                _command(port, b"DYNKV.SELECT", key, _select_request([], candidates))
            )
            assert selected is not None and selected[:3] == (1, 0, 0)
            _expect_error(
                port,
                b"DYNKV.SELECT",
                key,
                struct.pack("!BII", WIRE_VERSION, 0, max_candidates + 1),
                "DYNKV_INVALID_SELECT",
            )

            admission = _reserve_request(
                b"d" * max_domain,
                1,
                1,
                max_lease_ms,
                list(range(max_hashes)),
                candidates,
            )
            assert _command(port, b"DYNKV.SELECT_RESERVE", key, admission) == bytes(
                (ADMISSION_VERSION, ADMISSION_NO_CAPACITY)
            )

            identity = (
                struct.pack("!BI", ADMISSION_VERSION, max_domain)
                + b"d" * max_domain
                + struct.pack("!QQ", 2, 2)
            )
            _expect_error(
                port,
                b"DYNKV.SELECT_RESERVE",
                key,
                identity + struct.pack("!QI", max_lease_ms, max_hashes + 1),
                "DYNKV_INVALID_RESERVE",
            )
            _expect_error(
                port,
                b"DYNKV.SELECT_RESERVE",
                key,
                identity + struct.pack("!QII", max_lease_ms, 0, max_candidates + 1),
                "DYNKV_INVALID_RESERVE",
            )
            oversized_domain = (
                struct.pack("!BI", ADMISSION_VERSION, max_domain + 1)
                + b"d" * (max_domain + 1)
                + struct.pack("!QQQII", 3, 3, max_lease_ms, 0, 0)
            )
            _expect_error(
                port,
                b"DYNKV.SELECT_RESERVE",
                key,
                oversized_domain,
                "DYNKV_INVALID_RESERVE",
            )
            _expect_error(
                port,
                b"DYNKV.SELECT_RESERVE",
                key,
                _reserve_request(b"d", 4, 4, max_lease_ms + 1, [], []),
                "DYNKV_INVALID_RESERVE",
            )
        finally:
            _stop(process, port)
