# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Black-box durability and prefix-match test for the dynkv Valkey module."""

from __future__ import annotations

import socket
import struct
import subprocess
import tempfile
import threading
import time
from pathlib import Path
import re

WIRE_VERSION = 1
STORE = 1
REMOVE = 2
CLEAR = 3
ROOT_PARENT = (1 << 64) - 1
ADMISSION_VERSION = 1
ADMISSION_NO_CAPACITY = 0
ADMISSION_RESERVED = 1
REGISTRATION_VERSION = 1
LEGACY_LEASED_REGISTRATION_VERSION = 2
LEASED_REGISTRATION_VERSION = 3
WORKER_LEASE_CONTROL_VERSION = 1


def _module_limit(name: str) -> int:
    header = Path(__file__).resolve().parents[2] / "dynkv_limits.h"
    match = re.search(rf"^#define {re.escape(name)} (\d+)$", header.read_text(), re.MULTILINE)
    if match is None:
        raise RuntimeError(f"missing canonical module limit {name}")
    return int(match.group(1))


MAX_REGISTRATION_RANKS = _module_limit("DYNKV_MAX_REGISTRATION_RANKS")

__all__ = (
    "socket",
    "struct",
    "subprocess",
    "tempfile",
    "threading",
    "time",
    "Path",
    "WIRE_VERSION",
    "STORE",
    "REMOVE",
    "CLEAR",
    "ROOT_PARENT",
    "ADMISSION_VERSION",
    "ADMISSION_NO_CAPACITY",
    "ADMISSION_RESERVED",
    "REGISTRATION_VERSION",
    "MAX_REGISTRATION_RANKS",
    "LEGACY_LEASED_REGISTRATION_VERSION",
    "LEASED_REGISTRATION_VERSION",
    "WORKER_LEASE_CONTROL_VERSION",
    "_module_limit",
    "_find_free_port",
    "_encode_command",
    "_read_line",
    "_read_response",
    "_command",
    "_command_with_timeout",
    "_integer_array_command",
    "_integer_array_command_with_timeout",
    "_command_and_wait",
    "_integer_array_command_and_wait",
    "_start",
    "_stop",
    "_event",
    "_large_store_event",
    "_rank_generation",
    "_replace_snapshot",
    "_replace_rank_if_generation",
    "_match",
    "_select_request",
    "_select",
    "_register_worker",
    "_registration_ranks_payload",
    "_register_worker_ranks",
    "_leased_registration_payload",
    "_registration_generation",
    "_register_worker_ranks_leased",
    "_worker_lease_control_payload",
    "_renew_worker_lease",
    "_unregister_worker",
    "_apply_owned",
    "_stats",
    "_memory_usage",
    "_lifecycle_stats",
    "_gc_stats",
    "_gc",
    "_gc_current",
    "_drain_gc",
    "_advance_to_partial_lease_cleanup",
    "_admission_identity",
    "_reserve_request",
    "_parse_reserve",
    "_reserve",
    "_release_request",
    "_release",
    "_renew_request",
    "_renew",
    "_admission_stats",
    "_admission_workers",
)


def _find_free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _encode_command(*parts: bytes) -> bytes:
    wire = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        wire.extend(f"${len(part)}\r\n".encode())
        wire.extend(part)
        wire.extend(b"\r\n")
    return bytes(wire)


def _read_line(sock: socket.socket) -> bytes:
    line = bytearray()
    while not line.endswith(b"\r\n"):
        byte = sock.recv(1)
        if not byte:
            raise RuntimeError("unexpected EOF reading Valkey reply")
        line.extend(byte)
    return bytes(line[:-2])


def _read_response(sock: socket.socket) -> bytes:
    marker = sock.recv(1)
    if marker == b"+":
        return _read_line(sock)
    if marker == b"-":
        raise RuntimeError(_read_line(sock).decode())
    if marker == b"$":
        length = int(_read_line(sock))
        if length == -1:
            return b""
        result = bytearray()
        while len(result) < length:
            result.extend(sock.recv(length - len(result)))
        if sock.recv(2) != b"\r\n":
            raise RuntimeError("bulk reply was not CRLF terminated")
        return bytes(result)
    if marker == b":":
        return _read_line(sock)
    raise RuntimeError(f"unsupported Valkey reply marker {marker!r}")


def _command(port: int, *parts: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
        sock.sendall(_encode_command(*parts))
        return _read_response(sock)


def _command_with_timeout(port: int, timeout: float, *parts: bytes) -> bytes:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(_encode_command(*parts))
        return _read_response(sock)


def _integer_array_command(port: int, *parts: bytes) -> tuple[int, ...]:
    with socket.create_connection(("127.0.0.1", port), timeout=2) as sock:
        sock.sendall(_encode_command(*parts))
        marker = sock.recv(1)
        if marker == b"-":
            raise RuntimeError(_read_line(sock).decode())
        if marker != b"*":
            raise RuntimeError(f"expected array reply, got marker {marker!r}")
        count = int(_read_line(sock))
        values = []
        for _ in range(count):
            if sock.recv(1) != b":":
                raise RuntimeError("expected integer array entry")
            values.append(int(_read_line(sock)))
        return tuple(values)


def _integer_array_command_with_timeout(
    port: int, timeout: float, *parts: bytes
) -> tuple[int, ...]:
    with socket.create_connection(("127.0.0.1", port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(_encode_command(*parts))
        marker = sock.recv(1)
        if marker == b"-":
            raise RuntimeError(_read_line(sock).decode())
        if marker != b"*":
            raise RuntimeError(f"expected array reply, got marker {marker!r}")
        count = int(_read_line(sock))
        values = []
        for _ in range(count):
            if sock.recv(1) != b":":
                raise RuntimeError("expected integer array entry")
            values.append(int(_read_line(sock)))
        return tuple(values)


def _command_and_wait(port: int, *parts: bytes) -> tuple[bytes, bytes]:
    """Run a mutation and WAIT on one connection, as Valkey requires."""
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(_encode_command(*parts))
        mutation = _read_response(sock)
        sock.sendall(_encode_command(b"WAIT", b"1", b"3000"))
        return mutation, _read_response(sock)


def _integer_array_command_and_wait(
    port: int, *parts: bytes
) -> tuple[tuple[int, ...], bytes]:
    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(_encode_command(*parts))
        marker = sock.recv(1)
        if marker == b"-":
            raise RuntimeError(_read_line(sock).decode())
        if marker != b"*":
            raise RuntimeError(f"expected array reply, got marker {marker!r}")
        count = int(_read_line(sock))
        values = []
        for _ in range(count):
            if sock.recv(1) != b":":
                raise RuntimeError("expected integer array entry")
            values.append(int(_read_line(sock)))
        sock.sendall(_encode_command(b"WAIT", b"1", b"3000"))
        return tuple(values), _read_response(sock)


def _start(
    server: str,
    module: str,
    directory: Path,
    port: int,
    *,
    primary_port: int | None = None,
    strict_replication: bool = False,
    appendonly: bool = True,
) -> subprocess.Popen[bytes]:
    directory.mkdir(parents=True, exist_ok=True)
    log = (directory / "server.log").open("ab")
    command = [
        server,
        "--port",
        str(port),
        "--bind",
        "127.0.0.1",
        "--dir",
        str(directory),
        "--appendonly",
        "yes" if appendonly else "no",
        "--appendfsync",
        "everysec",
        "--repl-diskless-sync-delay",
        "0",
        "--loadmodule",
        module,
    ]
    if strict_replication:
        command.extend(["--min-replicas-to-write", "1", "--min-replicas-max-lag", "5"])
    if primary_port is not None:
        command.extend(
            [
                "--replicaof",
                "127.0.0.1",
                str(primary_port),
                "--replica-read-only",
                "yes",
                "--replica-serve-stale-data",
                "no",
            ]
        )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError((directory / "server.log").read_text())
        try:
            if _command(port, b"PING") == b"PONG":
                return process
        except (OSError, RuntimeError):
            time.sleep(0.05)
    process.terminate()
    raise RuntimeError("Valkey did not become ready")


def _stop(process: subprocess.Popen[bytes], port: int) -> None:
    if process.poll() is not None:
        return
    try:
        _command(port, b"SHUTDOWN", b"NOSAVE")
    except (OSError, RuntimeError):
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _event(
    kind: int,
    worker: int,
    dp_rank: int,
    event_id: int,
    parent: int = ROOT_PARENT,
    blocks: list[tuple[int, int]] | None = None,
) -> bytes:
    payload = bytearray(
        struct.pack("!BBQIQ", WIRE_VERSION, kind, worker, dp_rank, event_id)
    )
    if kind == STORE:
        assert blocks
        payload.extend(struct.pack("!QI", parent, len(blocks)))
        for external_hash, local_hash in blocks:
            payload.extend(struct.pack("!QQ", external_hash, local_hash))
    elif kind == REMOVE:
        assert blocks is not None
        payload.extend(struct.pack("!I", len(blocks)))
        for external_hash, _ in blocks:
            payload.extend(struct.pack("!Q", external_hash))
    return bytes(payload)


def _large_store_event(
    worker: int, dp_rank: int, event_id: int, block_count: int
) -> bytes:
    payload = bytearray(
        struct.pack("!BBQIQ", WIRE_VERSION, STORE, worker, dp_rank, event_id)
    )
    payload.extend(struct.pack("!QI", ROOT_PARENT, block_count))
    offset = len(payload)
    payload.extend(b"\x00" * (block_count * 16))
    base = 1 << 50
    pair = struct.Struct("!QQ")
    for index in range(block_count):
        pair.pack_into(payload, offset + index * 16, base + index, index + 1)
    return bytes(payload)


def _rank_generation(port: int, key: bytes, worker: int, dp_rank: int) -> int:
    response = _command(
        port,
        b"DYNKV.RANK_GENERATION",
        key,
        struct.pack("!Q", worker),
        struct.pack("!I", dp_rank),
    )
    assert len(response) == 8
    return struct.unpack("!Q", response)[0]


def _replace_snapshot(events: list[bytes]) -> bytes:
    """Encode the store-only rank snapshot accepted by conditional replace."""
    payload = bytearray(struct.pack("!BI", WIRE_VERSION, len(events)))
    for event in events:
        payload.extend(struct.pack("!I", len(event)))
        payload.extend(event)
    return bytes(payload)


def _replace_rank_if_generation(
    port: int,
    key: bytes,
    worker: int,
    dp_rank: int,
    expected_generation: int,
    events: list[bytes],
) -> int:
    response = _command(
        port,
        b"DYNKV.REPLACE_RANK_IF_GENERATION",
        key,
        struct.pack("!Q", worker),
        struct.pack("!I", dp_rank),
        struct.pack("!Q", expected_generation),
        _replace_snapshot(events),
    )
    assert len(response) == 8
    return struct.unpack("!Q", response)[0]


def _match(payload: bytes) -> dict[tuple[int, int], tuple[int, int]]:
    assert payload[0] == WIRE_VERSION
    count = struct.unpack("!I", payload[1:5])[0]
    assert len(payload) == 5 + count * 24
    result = {}
    for offset in range(5, len(payload), 24):
        worker, dp_rank, blocks, last_hash = struct.unpack(
            "!QIIQ", payload[offset : offset + 24]
        )
        result[(worker, dp_rank)] = (blocks, last_hash)
    return result


def _select_request(
    local_hashes: list[int], candidates: list[tuple[int, int, int]]
) -> bytes:
    payload = bytearray(struct.pack("!BI", WIRE_VERSION, len(local_hashes)))
    for local_hash in local_hashes:
        payload.extend(struct.pack("!Q", local_hash))
    payload.extend(struct.pack("!I", len(candidates)))
    for worker, dp_rank, load in candidates:
        payload.extend(struct.pack("!QIQ", worker, dp_rank, load))
    return bytes(payload)


def _select(payload: bytes) -> tuple[int, int, int, int] | None:
    assert payload[:2] == bytes((WIRE_VERSION, 0)) or payload[:2] == bytes(
        (WIRE_VERSION, 1)
    )
    if payload[1] == 0:
        assert len(payload) == 2
        return None
    assert len(payload) == 26
    return struct.unpack("!QIIQ", payload[2:])


def _register_worker(port: int, key: bytes, worker: int, dp_rank: int) -> bytes:
    return _command(
        port,
        b"DYNKV.REGISTER_WORKER",
        key,
        struct.pack("!Q", worker),
        struct.pack("!I", dp_rank),
    )


def _registration_ranks_payload(dp_ranks: list[int]) -> bytes:
    payload = bytearray(struct.pack("!BI", REGISTRATION_VERSION, len(dp_ranks)))
    for dp_rank in dp_ranks:
        payload.extend(struct.pack("!I", dp_rank))
    return bytes(payload)


def _register_worker_ranks(
    port: int, key: bytes, worker: int, dp_ranks: list[int]
) -> bytes:
    return _command(
        port,
        b"DYNKV.REGISTER_WORKER_RANKS",
        key,
        struct.pack("!Q", worker),
        _registration_ranks_payload(dp_ranks),
    )


def _leased_registration_payload(
    owner_nonce: int,
    lease_ms: int,
    dp_ranks: list[int],
    expected_generation: int = 0,
    *,
    version: int = LEASED_REGISTRATION_VERSION,
) -> bytes:
    if version == LEASED_REGISTRATION_VERSION:
        payload = bytearray(
            struct.pack(
                "!BQQQI",
                version,
                owner_nonce,
                lease_ms,
                expected_generation,
                len(dp_ranks),
            )
        )
    else:
        payload = bytearray(
            struct.pack("!BQQI", version, owner_nonce, lease_ms, len(dp_ranks))
        )
    for dp_rank in dp_ranks:
        payload.extend(struct.pack("!I", dp_rank))
    return bytes(payload)


def _registration_generation(port: int, key: bytes, worker: int) -> int:
    response = _command(
        port,
        b"DYNKV.REGISTRATION_GENERATION",
        key,
        struct.pack("!Q", worker),
    )
    assert len(response) == 8
    return struct.unpack("!Q", response)[0]


def _register_worker_ranks_leased(
    port: int,
    key: bytes,
    worker: int,
    owner_nonce: int,
    lease_ms: int,
    dp_ranks: list[int],
) -> bytes:
    return _command(
        port,
        b"DYNKV.REGISTER_WORKER_RANKS",
        key,
        struct.pack("!Q", worker),
        _leased_registration_payload(
            owner_nonce,
            lease_ms,
            dp_ranks,
            _registration_generation(port, key, worker),
        ),
    )


def _worker_lease_control_payload(
    worker: int, owner_nonce: int, lease_ms: int | None = None
) -> bytes:
    payload = bytearray(
        struct.pack("!BQQ", WORKER_LEASE_CONTROL_VERSION, worker, owner_nonce)
    )
    if lease_ms is not None:
        payload.extend(struct.pack("!Q", lease_ms))
    return bytes(payload)


def _renew_worker_lease(
    port: int, key: bytes, worker: int, owner_nonce: int, lease_ms: int
) -> bytes:
    return _command(
        port,
        b"DYNKV.RENEW_WORKER_LEASE",
        key,
        _worker_lease_control_payload(worker, owner_nonce, lease_ms),
    )


def _unregister_worker(port: int, key: bytes, worker: int, owner_nonce: int) -> bytes:
    return _command(
        port,
        b"DYNKV.UNREGISTER_WORKER",
        key,
        _worker_lease_control_payload(worker, owner_nonce),
    )


def _apply_owned(port: int, key: bytes, owner_nonce: int, event: bytes) -> bytes:
    return _command(
        port,
        b"DYNKV.APPLY_OWNED",
        key,
        struct.pack("!Q", owner_nonce),
        event,
    )


def _stats(port: int, key: bytes) -> tuple[int, int, int]:
    response = _integer_array_command(port, b"DYNKV.STATS", key)
    assert len(response) == 3
    return response


def _memory_usage(port: int, key: bytes) -> int:
    return int(_command(port, b"MEMORY", b"USAGE", key))


def _lifecycle_stats(port: int, key: bytes) -> tuple[int, int, int]:
    response = _integer_array_command(port, b"DYNKV.LIFECYCLE_STATS", key)
    assert len(response) == 3
    return response


def _gc_stats(port: int, key: bytes) -> tuple[int, ...]:
    response = _integer_array_command(port, b"DYNKV.GC_STATS", key)
    assert len(response) == 8
    return response


def _gc(port: int, key: bytes, watermark: int, budget: int) -> tuple[int, ...]:
    response = _integer_array_command(
        port,
        b"DYNKV.GC",
        key,
        struct.pack("!Q", watermark),
        struct.pack("!I", budget),
    )
    assert len(response) == 8
    assert response[0] <= budget
    assert response[1] <= response[0]
    return response


def _gc_current(port: int, key: bytes, budget: int) -> tuple[int, ...]:
    response = _integer_array_command(
        port,
        b"DYNKV.GC",
        key,
        b"CURRENT",
        struct.pack("!I", budget),
    )
    assert len(response) == 8
    assert response[0] <= budget and response[1] <= response[0]
    return response


def _drain_gc(port: int, key: bytes, *, budget: int = 64) -> None:
    idle = 0
    for _ in range(256):
        result = _gc_current(port, key, budget)
        stats = _gc_stats(port, key)
        if stats[1] == 0 and stats[3] == 0 and stats[6] == 0:
            return
        idle = idle + 1 if result[1] == 0 else 0
        assert idle < 16, (result, stats)
    raise AssertionError(f"GC did not converge: {_gc_stats(port, key)}")


def _advance_to_partial_lease_cleanup(
    port: int, key: bytes, *, replicated: bool = False
) -> None:
    """Commit BEGIN and one owner chunk, leaving the owner tuple fenced."""
    saw_begin = False
    for _ in range(256):
        if replicated:
            result, acknowledged = _integer_array_command_and_wait(
                port,
                b"DYNKV.GC",
                key,
                b"CURRENT",
                struct.pack("!I", 1),
            )
            if result[1] != 0:
                assert int(acknowledged) >= 1
        else:
            result = _gc_current(port, key, 1)
        assert result[0] <= 1 and result[1] <= 1
        if result[6] != 0:
            saw_begin = True
        if saw_begin and result[2] != 0:
            return
    raise AssertionError(
        f"lease cleanup did not reach an owner chunk: {_gc_stats(port, key)}"
    )


def _admission_identity(
    domain: bytes, client_nonce: int, request_nonce: int
) -> bytearray:
    assert 0 < len(domain) <= 128
    payload = bytearray(struct.pack("!BI", ADMISSION_VERSION, len(domain)))
    payload.extend(domain)
    payload.extend(struct.pack("!QQ", client_nonce, request_nonce))
    return payload


def _reserve_request(
    domain: bytes,
    client_nonce: int,
    request_nonce: int,
    lease_ms: int,
    local_hashes: list[int],
    candidates: list[tuple[int, int, int]],
) -> bytes:
    payload = _admission_identity(domain, client_nonce, request_nonce)
    payload.extend(struct.pack("!QI", lease_ms, len(local_hashes)))
    for local_hash in local_hashes:
        payload.extend(struct.pack("!Q", local_hash))
    payload.extend(struct.pack("!I", len(candidates)))
    for worker, dp_rank, capacity in candidates:
        payload.extend(struct.pack("!QII", worker, dp_rank, capacity))
    return bytes(payload)


def _parse_reserve(response: bytes) -> dict[str, int] | None:
    assert len(response) >= 2
    assert response[0] == ADMISSION_VERSION
    if response[1] == ADMISSION_NO_CAPACITY:
        assert len(response) == 2
        return None
    assert response[1] == ADMISSION_RESERVED
    assert len(response) == 46
    client, request, worker, dp_rank, expires, matched, active = struct.unpack(
        "!QQQIQII", response[2:]
    )
    return {
        "client": client,
        "request": request,
        "worker": worker,
        "dp_rank": dp_rank,
        "expires": expires,
        "matched": matched,
        "active": active,
    }


def _reserve(
    port: int, key: bytes, payload: bytes
) -> tuple[bytes, dict[str, int] | None]:
    response = _command(port, b"DYNKV.SELECT_RESERVE", key, payload)
    return response, _parse_reserve(response)


def _release_request(
    domain: bytes, client_nonce: int, request_nonce: int, expected_expires: int
) -> bytes:
    payload = _admission_identity(domain, client_nonce, request_nonce)
    payload.extend(struct.pack("!Q", expected_expires))
    return bytes(payload)


def _release(
    port: int,
    key: bytes,
    domain: bytes,
    client_nonce: int,
    request_nonce: int,
    expected_expires: int,
) -> int:
    response = _command(
        port,
        b"DYNKV.RELEASE",
        key,
        _release_request(domain, client_nonce, request_nonce, expected_expires),
    )
    assert len(response) == 2 and response[0] == ADMISSION_VERSION
    return response[1]


def _renew_request(
    domain: bytes,
    client_nonce: int,
    request_nonce: int,
    expected_expires: int,
    lease_ms: int,
) -> bytes:
    payload = _admission_identity(domain, client_nonce, request_nonce)
    payload.extend(struct.pack("!QQ", expected_expires, lease_ms))
    return bytes(payload)


def _renew(
    port: int,
    key: bytes,
    domain: bytes,
    client_nonce: int,
    request_nonce: int,
    expected_expires: int,
    lease_ms: int,
) -> tuple[bytes, dict[str, int]]:
    response = _command(
        port,
        b"DYNKV.RENEW",
        key,
        _renew_request(domain, client_nonce, request_nonce, expected_expires, lease_ms),
    )
    reservation = _parse_reserve(response)
    assert reservation is not None
    return response, reservation


def _admission_stats(port: int, key: bytes) -> int:
    return int(_command(port, b"DYNKV.ADMISSION_STATS", key))


def _admission_workers(port: int, key: bytes) -> None:
    for worker in (810, 811):
        assert _register_worker(port, key, worker, 0) == b"OK"
        assert (
            _command(
                port,
                b"DYNKV.APPLY",
                key,
                _event(STORE, worker, 0, 1, blocks=[(8101, 81), (8102, 82)]),
            )
            == b"OK"
        )
