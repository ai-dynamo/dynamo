# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small RESP and configuration helpers for Valkey router E2E tests."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest


def router_valkey_config(
    urls: Sequence[str],
    index_scope: str,
    *,
    authoritative_admission: bool = False,
    worker_lease_ms: int = 30_000,
    sentinel_urls: Sequence[str] = (),
    sentinel_master_name: str | None = None,
    tokenizer_sentinel_master_name: str | None = None,
    tokenizer_scope: str = "e2e",
    allow_degraded_writes: bool = False,
) -> str:
    config: dict[str, Any] = {
        "allow_insecure_plaintext": True,
        "urls": list(urls),
        "index_scope": index_scope,
        "connection_pool_size": 4,
        "required_replica_acks": 1,
        "worker_events": True,
        "authoritative_admission": authoritative_admission,
        "admission_lease_ms": 30_000,
        "worker_lease_ms": worker_lease_ms,
    }
    if sentinel_urls:
        if sentinel_master_name is None:
            raise ValueError("sentinel_master_name is required with sentinel_urls")
        config["sentinel"] = {
            "urls": list(sentinel_urls),
            "master_name": sentinel_master_name,
            "quorum": len(sentinel_urls) // 2 + 1,
        }
        config["allow_degraded_writes"] = allow_degraded_writes
    if tokenizer_sentinel_master_name is not None:
        if not sentinel_urls:
            raise ValueError("tokenizer Sentinel cache requires sentinel_urls")
        config["tokenizer_cache"] = {
            "enabled": True,
            "sentinel_master_name": tokenizer_sentinel_master_name,
            "scope": tokenizer_scope,
            # Force cross-frontend L2 lookups in the HA test instead of
            # allowing a common chat-template prefix to remain in local L1.
            "l1_bytes": 1,
            "timeout_ms": 1_000,
            "connection_pool_size": 4,
        }
    return json.dumps(config, separators=(",", ":"), sort_keys=True)


def _valkey_module_test_paths() -> tuple[Path, Path]:
    server = Path(
        os.environ.get(
            "VALKEY_SERVER",
            shutil.which("valkey-server") or "/nonexistent/valkey-server",
        )
    )
    module = Path(
        os.environ.get(
            "DYNKV_MODULE",
            str(
                Path(__file__).resolve().parents[2]
                / "lib/kv-router/valkey-module/dynkv.so"
            ),
        )
    )
    if not server.is_file() or not module.is_file():
        pytest.skip(
            "Valkey router E2E requires VALKEY_SERVER and DYNKV_MODULE "
            "(build with make -C lib/kv-router/valkey-module first)"
        )
    return server, module


def _valkey_integer_command(url: str, *parts: bytes) -> int:
    """Issue one integer-returning module command without a redis client dependency."""

    endpoint = url.removeprefix("valkey://").removeprefix("redis://")
    host, port = endpoint.rstrip("/").rsplit(":", 1)
    request = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        request.extend(f"${len(part)}\r\n".encode())
        request.extend(part)
        request.extend(b"\r\n")

    def read_line(sock: socket.socket) -> bytes:
        line = bytearray()
        while not line.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                raise RuntimeError("unexpected EOF reading Valkey reply")
            line.extend(chunk)
        return bytes(line[:-2])

    with socket.create_connection((host, int(port)), timeout=2) as sock:
        sock.sendall(request)
        marker = sock.recv(1)
        if marker == b"-":
            raise RuntimeError(read_line(sock).decode(errors="replace"))
        if marker != b":":
            raise RuntimeError(f"expected integer Valkey reply, got {marker!r}")
        return int(read_line(sock))


def _valkey_integer_array_command(url: str, *parts: bytes) -> tuple[int, ...]:
    """Issue one small integer-array module command without redis-py."""

    endpoint = url.removeprefix("valkey://").removeprefix("redis://")
    host, port = endpoint.rstrip("/").rsplit(":", 1)
    request = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        request.extend(f"${len(part)}\r\n".encode())
        request.extend(part)
        request.extend(b"\r\n")

    def read_line(sock: socket.socket) -> bytes:
        line = bytearray()
        while not line.endswith(b"\r\n"):
            chunk = sock.recv(1)
            if not chunk:
                raise RuntimeError("unexpected EOF reading Valkey reply")
            line.extend(chunk)
        return bytes(line[:-2])

    with socket.create_connection((host, int(port)), timeout=2) as sock:
        sock.sendall(request)
        marker = sock.recv(1)
        if marker == b"-":
            raise RuntimeError(read_line(sock).decode(errors="replace"))
        if marker != b"*":
            raise RuntimeError(f"expected array Valkey reply, got {marker!r}")
        count = int(read_line(sock))
        values = []
        for _ in range(count):
            if sock.recv(1) != b":":
                raise RuntimeError("expected integer in Valkey array reply")
            values.append(int(read_line(sock)))
        return tuple(values)


def _valkey_resp_command_on_socket(sock: socket.socket, *parts: bytes) -> Any:
    """Issue one RESP2 command while preserving the caller-owned connection."""

    request = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        request.extend(f"${len(part)}\r\n".encode())
        request.extend(part)
        request.extend(b"\r\n")
    sock.sendall(request)

    def read_exact(length: int) -> bytes:
        payload = bytearray()
        while len(payload) < length:
            chunk = sock.recv(length - len(payload))
            if not chunk:
                raise RuntimeError("unexpected EOF reading Valkey reply")
            payload.extend(chunk)
        return bytes(payload)

    def read_line() -> bytes:
        line = bytearray()
        while not line.endswith(b"\r\n"):
            line.extend(read_exact(1))
        return bytes(line[:-2])

    def read_response() -> Any:
        marker = read_exact(1)
        if marker == b"+":
            return read_line().decode(errors="replace")
        if marker == b"-":
            raise RuntimeError(read_line().decode(errors="replace"))
        if marker == b":":
            return int(read_line())
        if marker == b"$":
            length = int(read_line())
            if length == -1:
                return None
            payload = read_exact(length)
            if read_exact(2) != b"\r\n":
                raise RuntimeError("invalid Valkey bulk reply terminator")
            return payload
        if marker == b"*":
            count = int(read_line())
            if count == -1:
                return None
            return tuple(read_response() for _ in range(count))
        raise RuntimeError(f"unsupported Valkey RESP marker {marker!r}")

    return read_response()


def _valkey_cli_command(cli: Path, port: int, *parts: str) -> str:
    result = subprocess.run(
        [str(cli), "--raw", "-h", "127.0.0.1", "-p", str(port), *parts],
        check=False,
        capture_output=True,
        text=True,
        timeout=3,
    )
    output = (result.stdout + result.stderr).strip()
    if result.returncode != 0 or output.startswith("ERR"):
        raise RuntimeError(f"Valkey command {' '.join(parts)} failed: {output}")
    return result.stdout.strip()


def _valkey_barrier_and_wait(url: str, key: bytes) -> int:
    endpoint = url.removeprefix("valkey://").removeprefix("redis://")
    host, port = endpoint.rstrip("/").rsplit(":", 1)
    with socket.create_connection((host, int(port)), timeout=5) as sock:
        assert _valkey_resp_command_on_socket(sock, b"DYNKV.BARRIER", key) == "OK"
        acknowledged = _valkey_resp_command_on_socket(sock, b"WAIT", b"1", b"3000")
    assert isinstance(acknowledged, int)
    return acknowledged


async def _wait_for_replication_roles(
    cli: Path,
    expected_primary: int,
    expected_replica: int,
    timeout: float = 20,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_primary = ""
    last_replica = ""
    while asyncio.get_running_loop().time() < deadline:
        last_primary, last_replica = await asyncio.gather(
            asyncio.to_thread(
                _valkey_cli_command,
                cli,
                expected_primary,
                "INFO",
                "replication",
            ),
            asyncio.to_thread(
                _valkey_cli_command,
                cli,
                expected_replica,
                "INFO",
                "replication",
            ),
        )
        primary_ready = (
            "role:master" in last_primary and "connected_slaves:1" in last_primary
        )
        replica_ready = (
            "role:slave" in last_replica
            and f"master_port:{expected_primary}" in last_replica
            and "master_link_status:up" in last_replica
        )
        if primary_ready and replica_ready:
            return
        await asyncio.sleep(0.1)
    raise AssertionError(
        "Valkey roles did not converge after failover; "
        f"primary={last_primary!r}, replica={last_replica!r}"
    )


async def _wait_for_sentinel_primary(
    sentinels: Sequence[Any],
    expected_port: int,
    *,
    master_name: str | None = None,
    timeout: float = 20,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_addresses: list[tuple[str, int]] = []
    while asyncio.get_running_loop().time() < deadline:
        last_addresses = await asyncio.gather(
            *(
                asyncio.to_thread(
                    sentinel.get_master_addr,
                    master_name=master_name,
                )
                for sentinel in sentinels
            )
        )
        if all(address == ("127.0.0.1", expected_port) for address in last_addresses):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"Sentinels did not converge on port {expected_port}: {last_addresses}"
    )


async def _wait_for_module_convergence(
    direct_urls: Sequence[str],
    index_key: bytes,
    worker_count: int,
    timeout: float = 20,
) -> None:
    deadline = asyncio.get_running_loop().time() + timeout
    last_state: list[tuple[tuple[int, ...], tuple[int, ...], int]] = []
    while asyncio.get_running_loop().time() < deadline:
        last_state = []
        for url in direct_urls:
            stats, lifecycle, admissions = await asyncio.gather(
                asyncio.to_thread(
                    _valkey_integer_array_command,
                    url,
                    b"DYNKV.STATS",
                    index_key,
                ),
                asyncio.to_thread(
                    _valkey_integer_array_command,
                    url,
                    b"DYNKV.LIFECYCLE_STATS",
                    index_key,
                ),
                asyncio.to_thread(
                    _valkey_integer_command,
                    url,
                    b"DYNKV.ADMISSION_STATS",
                    index_key,
                ),
            )
            last_state.append((stats, lifecycle, admissions))
        if (
            last_state[0] == last_state[1]
            and last_state[0][0][0] > 0
            and last_state[0][0][1] == worker_count
            and last_state[0][1][0] > 0
            and last_state[0][2] == 0
        ):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(
        f"module state did not converge across primary/replica: {last_state}"
    )
