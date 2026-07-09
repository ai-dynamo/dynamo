#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: F401

"""Measure the saturation knee of one local dynkv Valkey server.

The simple count-based CLI remains compatible with the original tool::

  python benchmarks/router/valkey_module_saturation.py \
    --server "$VALKEY_REPO/src/valkey-server" \
    --module lib/kv-router/valkey-module/dynkv.so \
    --mode apply --events 400000 --connections 64 --pipeline 128

Duration, repetition, and concurrency/pipeline sweep options turn the same
runner into a saturation campaign. Results distinguish logical iterations
(events, queries, or reservation lifecycles) from module command throughput
and include client latency, server command statistics, CPU, memory, wire bytes,
and AOF facts.

``select_reserve`` measures complete reserve/release lifecycles; ``renew``
measures reserve/renew/release lifecycles and reports each command's latency
separately. ``churn_owned`` alternates owner-fenced STORE/REMOVE pairs over a
bounded ring of prefix identities and verifies that neither radix nodes nor
inactive owner records accumulate. ``--preset dynamo`` models four leased
workers and three logical frontend connection pools by default. A sweep writes
CSV plus PNG/SVG throughput and p99-latency plots.

This is a single-server metadata benchmark. It does not measure replica WAIT,
frontend tokenization, GPU KV transfer, or NIXL.
"""

from __future__ import annotations

import argparse
import asyncio
from array import array
import contextlib
import csv
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Sequence


WIRE_VERSION = 1
EVENT_STORE = 1
EVENT_REMOVE = 2
ROOT_PARENT = (1 << 64) - 1
ADMISSION_VERSION = 1
ADMISSION_RESERVED = 1
LEASED_REGISTRATION_VERSION = 3
XOR_MASK = 0xA5A5_A5A5_A5A5_A5A5
MAX_BLOCKS = 1_048_576
MAX_LEASE_MS = 600_000
HASH_NAMESPACE_SIZE = 1 << 48
MAX_HASH_NAMESPACES = 1 << 16

MODE_ALIASES = {
    "reserve_release": "select_reserve",
    "select_reserve_release": "select_reserve",
}
MODES = (
    "apply",
    "apply_owned",
    "churn_owned",
    "match",
    "select",
    "select_reserve",
    "renew",
    "mixed",
    *MODE_ALIASES,
)
QUERY_MODES = {"match", "select", "mixed"}
LIFECYCLE_MODES = {"select_reserve", "renew"}
OWNED_MODES = {"apply_owned", "churn_owned", *LIFECYCLE_MODES}

DYNKV_PROTOCOL = {
    "event_wire_version": WIRE_VERSION,
    "leased_registration_version": LEASED_REGISTRATION_VERSION,
}


def file_provenance(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as artifact:
        while chunk := artifact.read(4 * 1024 * 1024):
            digest.update(chunk)
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "sha256": digest.hexdigest(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def campaign_provenance(args: argparse.Namespace) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None
    return {
        "server": file_provenance(args.server),
        "module": file_provenance(args.module),
        "harness": file_provenance(Path(__file__)),
        "python": {
            "executable": os.path.realpath(sys.executable),
            "version": sys.version,
        },
        "git_revision": revision,
        "dynkv_protocol": DYNKV_PROTOCOL,
        "runner_cpu_affinity": (
            sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None
        ),
    }


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def resp_command(*parts: bytes) -> bytes:
    encoded = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        encoded.extend(f"${len(part)}\r\n".encode())
        encoded.extend(part)
        encoded.extend(b"\r\n")
    return bytes(encoded)


@dataclass(frozen=True)
class Resp:
    marker: bytes
    value: Any
    wire_bytes: int


class RespCommandError(RuntimeError):
    def __init__(self, message: str, wire_bytes: int) -> None:
        super().__init__(message)
        self.wire_bytes = wire_bytes


async def read_resp(reader: asyncio.StreamReader) -> Resp:
    marker = await reader.readexactly(1)
    if marker in (b"+", b"-", b":"):
        line = await reader.readuntil(b"\r\n")
        wire_bytes = 1 + len(line)
        payload = line[:-2]
        if marker == b"-":
            raise RespCommandError(payload.decode(errors="replace"), wire_bytes)
        value: bytes | int = int(payload) if marker == b":" else payload
        return Resp(marker, value, wire_bytes)
    if marker == b"$":
        header = await reader.readuntil(b"\r\n")
        length = int(header[:-2])
        wire_bytes = 1 + len(header)
        if length < 0:
            return Resp(marker, None, wire_bytes)
        payload = await reader.readexactly(length)
        trailer = await reader.readexactly(2)
        if trailer != b"\r\n":
            raise RuntimeError("bulk response lacks CRLF")
        return Resp(marker, payload, wire_bytes + length + 2)
    if marker == b"*":
        header = await reader.readuntil(b"\r\n")
        count = int(header[:-2])
        wire_bytes = 1 + len(header)
        if count < 0:
            return Resp(marker, None, wire_bytes)
        values: list[Resp] = []
        for _ in range(count):
            item = await read_resp(reader)
            values.append(item)
            wire_bytes += item.wire_bytes
        return Resp(marker, values, wire_bytes)
    raise RuntimeError(f"unsupported RESP marker {marker!r}")


async def read_response(reader: asyncio.StreamReader) -> bytes:
    """Compatibility wrapper used by the startup and setup paths."""

    response = await read_resp(reader)
    if isinstance(response.value, bytes):
        return response.value
    if isinstance(response.value, int):
        return str(response.value).encode()
    raise RuntimeError(f"expected scalar RESP response, got {response.marker!r}")


def store_event(worker: int, event_id: int, first_hash: int, blocks: int) -> bytes:
    payload = bytearray(
        struct.pack("!BBQIQ", WIRE_VERSION, EVENT_STORE, worker, 0, event_id)
    )
    payload.extend(struct.pack("!QI", ROOT_PARENT, blocks))
    for index in range(blocks):
        block_hash = first_hash + index
        payload.extend(struct.pack("!QQ", block_hash, block_hash ^ XOR_MASK))
    return bytes(payload)


def remove_event(worker: int, event_id: int, block_hashes: Sequence[int]) -> bytes:
    if not block_hashes or len(block_hashes) > MAX_BLOCKS:
        raise ValueError(f"REMOVE block count must be in [1, {MAX_BLOCKS}]")
    payload = bytearray(
        struct.pack(
            "!BBQIQI",
            WIRE_VERSION,
            EVENT_REMOVE,
            worker,
            0,
            event_id,
            len(block_hashes),
        )
    )
    for block_hash in block_hashes:
        payload.extend(struct.pack("!Q", block_hash))
    return bytes(payload)


def event_hash_start(namespace: int, sequence: int, blocks: int) -> int:
    """Return a non-overlapping hash range for one generated STORE.

    The original generator shifted the sequence by eight bits, which made
    adjacent events overlap when ``blocks > 256``. Each connection/phase now
    owns a 48-bit namespace and advances by the actual event width.
    """

    if not 0 <= namespace < MAX_HASH_NAMESPACES:
        raise ValueError("event hash namespace exceeds 16-bit allocation")
    if sequence < 0 or blocks <= 0:
        raise ValueError("event sequence and block count must be positive")
    offset = sequence * blocks
    if offset + blocks > HASH_NAMESPACE_SIZE:
        raise ValueError("event hash namespace exhausted")
    return (namespace << 48) + offset


def match_request(block_hashes: Sequence[int]) -> bytes:
    payload = bytearray(struct.pack("!BI", WIRE_VERSION, len(block_hashes)))
    for block_hash in block_hashes:
        payload.extend(struct.pack("!Q", block_hash ^ XOR_MASK))
    return bytes(payload)


def select_request(
    block_hashes: Sequence[int], candidates: Sequence[tuple[int, int, int]]
) -> bytes:
    payload = bytearray(match_request(block_hashes))
    payload.extend(struct.pack("!I", len(candidates)))
    for worker, dp_rank, load in candidates:
        payload.extend(struct.pack("!QIQ", worker, dp_rank, load))
    return bytes(payload)


def admission_identity(
    domain: bytes, client_nonce: int, request_nonce: int
) -> bytearray:
    if not 0 < len(domain) <= 128:
        raise ValueError("admission domain must be 1 through 128 bytes")
    payload = bytearray(struct.pack("!BI", ADMISSION_VERSION, len(domain)))
    payload.extend(domain)
    payload.extend(struct.pack("!QQ", client_nonce, request_nonce))
    return payload


def reserve_request(
    domain: bytes,
    client_nonce: int,
    request_nonce: int,
    lease_ms: int,
    local_hashes: Sequence[int],
    candidates: Sequence[tuple[int, int, int]],
) -> bytes:
    payload = admission_identity(domain, client_nonce, request_nonce)
    payload.extend(struct.pack("!QI", lease_ms, len(local_hashes)))
    for local_hash in local_hashes:
        payload.extend(struct.pack("!Q", local_hash))
    payload.extend(struct.pack("!I", len(candidates)))
    for worker, dp_rank, capacity in candidates:
        payload.extend(struct.pack("!QII", worker, dp_rank, capacity))
    return bytes(payload)


def release_request(
    domain: bytes, client_nonce: int, request_nonce: int, expires_at_ms: int
) -> bytes:
    payload = admission_identity(domain, client_nonce, request_nonce)
    payload.extend(struct.pack("!Q", expires_at_ms))
    return bytes(payload)


def renew_request(
    domain: bytes,
    client_nonce: int,
    request_nonce: int,
    expires_at_ms: int,
    lease_ms: int,
) -> bytes:
    payload = admission_identity(domain, client_nonce, request_nonce)
    payload.extend(struct.pack("!QQ", expires_at_ms, lease_ms))
    return bytes(payload)


def leased_registration_payload(
    owner_nonce: int, lease_ms: int, expected_generation: int
) -> bytes:
    return struct.pack(
        "!BQQQII",
        LEASED_REGISTRATION_VERSION,
        owner_nonce,
        lease_ms,
        expected_generation,
        1,
        0,
    )


def scalar_bytes(response: Resp) -> bytes:
    if not isinstance(response.value, bytes):
        raise ValueError(f"expected scalar bytes, got RESP {response.marker!r}")
    return response.value


def expect_simple(response: Resp, expected: bytes) -> bytes:
    value = scalar_bytes(response)
    if response.marker != b"+" or value != expected:
        raise ValueError(
            f"expected simple response {expected!r}, got {response.marker!r} {value!r}"
        )
    return value


def parse_match_response(payload: bytes) -> list[tuple[int, int, int, int]]:
    if len(payload) < 5 or payload[0] != WIRE_VERSION:
        raise ValueError("invalid MATCH response header")
    count = struct.unpack_from("!I", payload, 1)[0]
    if len(payload) != 5 + count * 24:
        raise ValueError("invalid MATCH response length")
    result = []
    offset = 5
    for _ in range(count):
        result.append(struct.unpack_from("!QIIQ", payload, offset))
        offset += 24
    return result


def parse_select_response(payload: bytes) -> tuple[int, int, int, int] | None:
    if len(payload) < 2 or payload[0] != WIRE_VERSION:
        raise ValueError("invalid SELECT response header")
    if payload[1] == 0:
        if len(payload) != 2:
            raise ValueError("invalid empty SELECT response")
        return None
    if payload[1] != 1 or len(payload) != 26:
        raise ValueError("invalid selected SELECT response")
    return struct.unpack("!QIIQ", payload[2:])


@dataclass(frozen=True)
class Reservation:
    client_nonce: int
    request_nonce: int
    worker: int
    dp_rank: int
    expires_at_ms: int
    matched_blocks: int
    active_at_grant: int


def parse_reservation_response(payload: bytes) -> Reservation:
    if len(payload) != 46 or payload[:2] != bytes(
        (ADMISSION_VERSION, ADMISSION_RESERVED)
    ):
        raise ValueError("SELECT_RESERVE did not grant a reservation")
    values = struct.unpack("!QQQIQII", payload[2:])
    return Reservation(*values)


def parse_release_response(payload: bytes) -> int:
    if len(payload) != 2 or payload[0] != ADMISSION_VERSION:
        raise ValueError("invalid RELEASE response")
    if payload[1] not in (0, 1):
        raise ValueError("invalid RELEASE status")
    return payload[1]


def linear_percentile(values: Sequence[int | float], quantile: float) -> float | None:
    if not values:
        return None
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return float(ordered[low] + (ordered[high] - ordered[low]) * (position - low))


@dataclass
class LatencySeries:
    limit: int
    observations: int = 0
    stride: int = 1
    samples_ns: array[int] = field(default_factory=lambda: array("Q"))

    def record(self, latency_ns: int) -> None:
        self.observations += 1
        if self.observations % self.stride:
            return
        if len(self.samples_ns) >= self.limit:
            self.samples_ns = array("Q", self.samples_ns[1::2])
            self.stride *= 2
            if self.observations % self.stride:
                return
        self.samples_ns.append(latency_ns)

    def summary(self) -> dict[str, int | float | None]:
        result: dict[str, int | float | None] = {
            "observations": self.observations,
            "samples": len(self.samples_ns),
            "sampling_stride": self.stride,
        }
        for label, quantile in (
            ("p50_ms", 0.50),
            ("p90_ms", 0.90),
            ("p95_ms", 0.95),
            ("p99_ms", 0.99),
            ("p99_9_ms", 0.999),
        ):
            value = linear_percentile(self.samples_ns, quantile)
            result[label] = None if value is None else value / 1_000_000
        result["max_ms"] = max(self.samples_ns) / 1_000_000 if self.samples_ns else None
        return result


@dataclass
class LatencyRecorder:
    limit_per_command: int
    series: dict[str, LatencySeries] = field(default_factory=dict)

    def record(self, command: str, latency_ns: int) -> None:
        self.series.setdefault(
            command, LatencySeries(limit=self.limit_per_command)
        ).record(latency_ns)

    def summary(self) -> dict[str, dict[str, int | float | None]]:
        return {name: value.summary() for name, value in sorted(self.series.items())}


@dataclass
class Counters:
    commands: int = 0
    iterations: int = 0
    events: int = 0
    blocks: int = 0
    queries: int = 0
    selections: int = 0
    reservation_cycles: int = 0
    event_bytes: int = 0
    logical_payload_bytes: int = 0
    request_wire_bytes: int = 0
    response_wire_bytes: int = 0
    commands_by_kind: dict[str, int] = field(default_factory=dict)
    events_by_kind: dict[str, int] = field(default_factory=dict)

    def merge(self, other: Counters) -> None:
        for name in (
            "commands",
            "iterations",
            "events",
            "blocks",
            "queries",
            "selections",
            "reservation_cycles",
            "event_bytes",
            "logical_payload_bytes",
            "request_wire_bytes",
            "response_wire_bytes",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for command, count in other.commands_by_kind.items():
            self.commands_by_kind[command] = (
                self.commands_by_kind.get(command, 0) + count
            )
        for event, count in other.events_by_kind.items():
            self.events_by_kind[event] = self.events_by_kind.get(event, 0) + count


Validator = Callable[[Resp], Any]


@dataclass(frozen=True)
class WorkCommand:
    kind: str
    encoded: bytes
    validator: Validator
    logical_payload_bytes: int = 0
    event_bytes: int = 0
    blocks: int = 0
    query: bool = False
    selection: bool = False
    latency_kind: str | None = None
    event_kind: str | None = None
