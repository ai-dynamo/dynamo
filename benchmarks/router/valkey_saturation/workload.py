# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import contextlib
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .protocol import (
    Counters,
    LIFECYCLE_MODES,
    OWNED_MODES,
    XOR_MASK,
    LatencyRecorder,
    Reservation,
    Resp,
    RespCommandError,
    Validator,
    WorkCommand,
    event_hash_start,
    expect_simple,
    leased_registration_payload,
    match_request,
    parse_match_response,
    parse_release_response,
    parse_reservation_response,
    parse_select_response,
    read_resp,
    release_request,
    remove_event,
    renew_request,
    reserve_request,
    resp_command,
    scalar_bytes,
    select_request,
    store_event,
)
from .setup import send_batch

@dataclass(frozen=True)
class Topology:
    preset: str
    workers: tuple[int, ...]
    writer_workers: tuple[int, ...]
    frontends: int
    owners: tuple[int, ...]
    leased: bool
    owner_nonces: dict[int, int]

    def worker_for_connection(self, connection_id: int) -> int:
        return self.writer_workers[connection_id % len(self.writer_workers)]


@dataclass(frozen=True)
class WorkloadSetup:
    key: bytes
    topology: Topology
    block_hashes: tuple[int, ...]
    local_hashes: tuple[int, ...]
    match_payload: bytes
    select_payload: bytes
    admission_candidates: tuple[tuple[int, int, int], ...]
    domain: bytes
    lease_ms: int


def owner_nonce(worker: int) -> int:
    return 0xD100_0000_0000_0000 + worker


def registration_validator(response: Resp) -> bytes:
    value = scalar_bytes(response)
    if response.marker != b"+" or value not in {b"OK", b"NOOP"}:
        raise ValueError(f"registration returned {response.marker!r} {value!r}")
    return value


def build_topology(args: argparse.Namespace, connections: int, mode: str) -> Topology:
    if args.preset in {"dynamo", "worker-scale"}:
        worker_count = args.workers or (1024 if args.preset == "worker-scale" else 4)
        frontend_count = args.frontends or 3
        owners = args.owners or worker_count
        writer_workers = tuple(range(1, worker_count + 1))
    else:
        worker_count = args.workers or connections + 1
        frontend_count = args.frontends or 1
        owners = args.owners or 1
        writer_workers = (
            tuple(range(2, connections + 2))
            if args.workers is None
            else tuple(range(1, worker_count + 1))
        )
    if owners > worker_count:
        raise ValueError("prefix owner count cannot exceed worker count")
    workers = tuple(range(1, worker_count + 1))
    leased = args.preset in {"dynamo", "worker-scale"} or mode in OWNED_MODES
    return Topology(
        preset=args.preset,
        workers=workers,
        writer_workers=writer_workers,
        frontends=frontend_count,
        owners=workers[:owners],
        leased=leased,
        owner_nonces={worker: owner_nonce(worker) for worker in workers},
    )


async def execute(port: int, *parts: bytes) -> Resp:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(resp_command(*parts))
        await writer.drain()
        return await read_resp(reader)
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()


def registration_generation_validator(response: Resp) -> int:
    payload = scalar_bytes(response)
    if response.marker != b"$" or len(payload) != 8:
        raise ValueError("REGISTRATION_GENERATION must return one 8-byte bulk value")
    return struct.unpack("!Q", payload)[0]


async def register_leased_worker(
    port: int,
    key: bytes,
    worker: int,
    nonce: int,
    lease_ms: int,
    *,
    max_attempts: int = 8,
    base_retry_delay_s: float = 0.005,
) -> bytes:
    """Claim one worker using v3 lifecycle CAS, retrying only explicit staleness.

    Transport failures and malformed replies are deliberately not retried: an
    ambiguous registration result must fail the sample rather than risk
    benchmarking an unknown worker incarnation.
    """

    if max_attempts <= 0:
        raise ValueError("registration max_attempts must be positive")
    for attempt in range(max_attempts):
        generation = registration_generation_validator(
            await execute(
                port,
                b"DYNKV.REGISTRATION_GENERATION",
                key,
                struct.pack("!Q", worker),
            )
        )
        try:
            response = await execute(
                port,
                b"DYNKV.REGISTER_WORKER_RANKS",
                key,
                struct.pack("!Q", worker),
                leased_registration_payload(nonce, lease_ms, generation),
            )
        except RespCommandError as error:
            if (
                "DYNKV_STALE_REGISTRATION_GENERATION" not in str(error)
                or attempt + 1 == max_attempts
            ):
                raise RuntimeError(f"leased registration failed: {error}") from error
            await asyncio.sleep(min(base_retry_delay_s * (2**attempt), 0.25))
            continue
        return registration_validator(response)
    raise AssertionError("registration retry loop exhausted unexpectedly")


async def setup_workload(
    port: int,
    key: bytes,
    topology: Topology,
    query_blocks: int,
    capacity: int,
    lease_ms: int,
    worker_lease_ms: int,
) -> WorkloadSetup:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        registration_commands = []
        for worker in topology.workers:
            if topology.leased:
                await register_leased_worker(
                    port,
                    key,
                    worker,
                    topology.owner_nonces[worker],
                    worker_lease_ms,
                )
                continue
            else:
                parts = (
                    b"DYNKV.REGISTER_WORKER",
                    key,
                    struct.pack("!Q", worker),
                    struct.pack("!I", 0),
                )
            registration_commands.append(
                WorkCommand(
                    "register",
                    resp_command(*parts),
                    registration_validator,
                )
            )
        if registration_commands:
            await send_batch(
                reader, writer, registration_commands, Counters(), latency=None
            )

        block_hashes = tuple(10_000 + index for index in range(query_blocks))
        for worker in topology.owners:
            event = store_event(worker, 1, block_hashes[0], query_blocks)
            if topology.leased:
                parts = (
                    b"DYNKV.APPLY_OWNED",
                    key,
                    struct.pack("!Q", topology.owner_nonces[worker]),
                    event,
                )
            else:
                parts = (b"DYNKV.APPLY", key, event)
            await send_batch(
                reader,
                writer,
                [
                    WorkCommand(
                        "preload",
                        resp_command(*parts),
                        lambda response: expect_simple(response, b"OK"),
                    )
                ],
                Counters(),
                latency=None,
            )
    finally:
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()

    local_hashes = tuple(block_hash ^ XOR_MASK for block_hash in block_hashes)
    stateless_candidates = tuple((worker, 0, 0) for worker in topology.workers)
    return WorkloadSetup(
        key=key,
        topology=topology,
        block_hashes=block_hashes,
        local_hashes=local_hashes,
        match_payload=match_request(block_hashes),
        select_payload=select_request(block_hashes, stateless_candidates),
        admission_candidates=tuple(
            (worker, 0, capacity) for worker in topology.workers
        ),
        domain=b"dynkv-saturation",
        lease_ms=lease_ms,
    )


def match_validator(setup: WorkloadSetup) -> Validator:
    expected_workers = set(setup.topology.owners)
    expected_depth = len(setup.local_hashes)

    def validate(response: Resp) -> list[tuple[int, int, int, int]]:
        parsed = parse_match_response(scalar_bytes(response))
        workers = {worker for worker, rank, _, _ in parsed if rank == 0}
        if workers != expected_workers or any(
            matched != expected_depth for _, _, matched, _ in parsed
        ):
            raise ValueError(
                f"MATCH returned workers/depth {parsed!r}, expected "
                f"{sorted(expected_workers)!r}/{expected_depth}"
            )
        return parsed

    return validate


def select_validator(setup: WorkloadSetup) -> Validator:
    expected_workers = set(setup.topology.owners)
    expected_depth = len(setup.local_hashes)

    def validate(response: Resp) -> tuple[int, int, int, int]:
        parsed = parse_select_response(scalar_bytes(response))
        if parsed is None:
            raise ValueError("SELECT returned no candidate")
        worker, rank, matched, _ = parsed
        if worker not in expected_workers or rank != 0 or matched != expected_depth:
            raise ValueError(f"SELECT returned unexpected choice {parsed!r}")
        return parsed

    return validate


def reserve_validator(
    client_nonce: int, request_nonce: int, setup: WorkloadSetup
) -> Validator:
    workers = set(setup.topology.workers)

    def validate(response: Resp) -> Reservation:
        reservation = parse_reservation_response(scalar_bytes(response))
        if (
            reservation.client_nonce != client_nonce
            or reservation.request_nonce != request_nonce
            or reservation.worker not in workers
            or reservation.dp_rank != 0
        ):
            raise ValueError(f"unexpected reservation {reservation!r}")
        return reservation

    return validate


@dataclass
class PhaseControl:
    start: asyncio.Event = field(default_factory=asyncio.Event)
    deadline: float | None = None


@dataclass(frozen=True)
class ConnectionResult:
    counters: Counters
    completed_at: float


def event_identity(
    phase_id: int,
    connections: int,
    connection_id: int,
    sequence: int,
    blocks: int,
) -> tuple[int, int]:
    # Namespace zero is unused so generated roots cannot collide with the
    # small fixed hashes used by the query preload.
    namespace = phase_id * connections + connection_id + 1
    first_hash = event_hash_start(namespace, sequence, blocks)
    event_id = ((phase_id + 1) << 56) | (sequence * connections + connection_id + 1)
    if event_id > (1 << 64) - 1:
        raise ValueError("event ID space exhausted")
    return event_id, first_hash


def churn_event_ids(
    phase_id: int, connections: int, connection_id: int, sequence: int
) -> tuple[int, int]:
    if not 0 <= phase_id < 255:
        raise ValueError("churn phase ID must be in [0, 254]")
    if connections <= 0 or not 0 <= connection_id < connections or sequence <= 0:
        raise ValueError("invalid churn connection or sequence")
    ordinal = (sequence - 1) * connections + connection_id
    store_low = ordinal * 2 + 1
    remove_low = store_low + 1
    if remove_low >= 1 << 56:
        raise ValueError("churn event ID space exhausted")
    phase = (phase_id + 1) << 56
    return phase | store_low, phase | remove_low


def churn_hash_start(
    phase_id: int,
    connections: int,
    connection_id: int,
    sequence: int,
    prefixes_per_connection: int,
    blocks: int,
) -> int:
    if prefixes_per_connection <= 0:
        raise ValueError("churn prefixes per connection must be positive")
    slot = (sequence - 1) % prefixes_per_connection
    namespace = phase_id * connections + connection_id + 1
    return event_hash_start(namespace, slot, blocks)


def churn_owned_commands(
    setup: WorkloadSetup,
    *,
    phase_id: int,
    connections: int,
    connection_id: int,
    sequence: int,
    prefixes_per_connection: int,
    blocks: int,
) -> tuple[WorkCommand, WorkCommand]:
    worker = setup.topology.worker_for_connection(connection_id)
    store_id, remove_id = churn_event_ids(
        phase_id, connections, connection_id, sequence
    )
    first_hash = churn_hash_start(
        phase_id,
        connections,
        connection_id,
        sequence,
        prefixes_per_connection,
        blocks,
    )
    block_hashes = tuple(first_hash + offset for offset in range(blocks))
    store = store_event(worker, store_id, first_hash, blocks)
    remove = remove_event(worker, remove_id, block_hashes)
    owner = struct.pack("!Q", setup.topology.owner_nonces[worker])

    def applied(response: Resp) -> bytes:
        return expect_simple(response, b"OK")

    return (
        WorkCommand(
            "apply_owned",
            resp_command(b"DYNKV.APPLY_OWNED", setup.key, owner, store),
            applied,
            logical_payload_bytes=len(store),
            event_bytes=len(store),
            blocks=blocks,
            latency_kind="apply_owned_store",
            event_kind="store",
        ),
        WorkCommand(
            "apply_owned",
            resp_command(b"DYNKV.APPLY_OWNED", setup.key, owner, remove),
            applied,
            logical_payload_bytes=len(remove),
            event_bytes=len(remove),
            blocks=blocks,
            latency_kind="apply_owned_remove",
            event_kind="remove",
        ),
    )


def request_nonce(
    phase_id: int, connections: int, connection_id: int, sequence: int
) -> int:
    value = ((phase_id + 1) << 56) | (sequence * connections + connection_id + 1)
    if value > (1 << 64) - 1:
        raise ValueError("request nonce space exhausted")
    return value


async def run_simple_connection(
    *,
    port: int,
    setup: WorkloadSetup,
    mode: str,
    connection_id: int,
    connections: int,
    count: int | None,
    blocks_per_event: int,
    churn_prefixes_per_connection: int,
    pipeline: int,
    phase_id: int,
    control: PhaseControl,
    ready: asyncio.Queue[None],
    latency: LatencyRecorder | None,
) -> ConnectionResult:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    ready.put_nowait(None)
    await control.start.wait()
    counters = Counters()
    sequence = 1
    remaining = count
    match_check = match_validator(setup)
    select_check = select_validator(setup)
    try:
        while remaining is None or remaining > 0:
            if control.deadline is not None and time.perf_counter() >= control.deadline:
                break
            current = pipeline if remaining is None else min(pipeline, remaining)
            commands: list[WorkCommand] = []
            for _ in range(current):
                if mode == "churn_owned":
                    commands.extend(
                        churn_owned_commands(
                            setup,
                            phase_id=phase_id,
                            connections=connections,
                            connection_id=connection_id,
                            sequence=sequence,
                            prefixes_per_connection=churn_prefixes_per_connection,
                            blocks=blocks_per_event,
                        )
                    )
                elif mode in {"apply", "apply_owned", "mixed"}:
                    worker = setup.topology.worker_for_connection(connection_id)
                    event_id, first_hash = event_identity(
                        phase_id,
                        connections,
                        connection_id,
                        sequence,
                        blocks_per_event,
                    )
                    event = store_event(worker, event_id, first_hash, blocks_per_event)
                    if mode == "apply_owned":
                        parts = (
                            b"DYNKV.APPLY_OWNED",
                            setup.key,
                            struct.pack("!Q", setup.topology.owner_nonces[worker]),
                            event,
                        )
                        kind = "apply_owned"
                    else:
                        parts = (b"DYNKV.APPLY", setup.key, event)
                        kind = "apply"
                    commands.append(
                        WorkCommand(
                            kind,
                            resp_command(*parts),
                            lambda response: expect_simple(response, b"OK"),
                            logical_payload_bytes=len(event),
                            event_bytes=len(event),
                            blocks=blocks_per_event,
                        )
                    )
                if mode in {"match", "mixed"}:
                    commands.append(
                        WorkCommand(
                            "match",
                            resp_command(
                                b"DYNKV.MATCH", setup.key, setup.match_payload
                            ),
                            match_check,
                            logical_payload_bytes=len(setup.match_payload),
                            query=True,
                        )
                    )
                elif mode == "select":
                    commands.append(
                        WorkCommand(
                            "select",
                            resp_command(
                                b"DYNKV.SELECT", setup.key, setup.select_payload
                            ),
                            select_check,
                            logical_payload_bytes=len(setup.select_payload),
                            query=True,
                            selection=True,
                        )
                    )
                sequence += 1
            await send_batch(reader, writer, commands, counters, latency)
            counters.iterations += current
            if remaining is not None:
                remaining -= current
    finally:
        completed_at = time.perf_counter()
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()
    return ConnectionResult(counters, completed_at)


async def run_lifecycle_connection(
    *,
    port: int,
    setup: WorkloadSetup,
    mode: str,
    connection_id: int,
    connections: int,
    count: int | None,
    pipeline: int,
    phase_id: int,
    control: PhaseControl,
    ready: asyncio.Queue[None],
    latency: LatencyRecorder | None,
) -> ConnectionResult:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    ready.put_nowait(None)
    await control.start.wait()
    counters = Counters()
    sequence = 1
    remaining = count
    client_nonce = connection_id + 1
    try:
        while remaining is None or remaining > 0:
            if control.deadline is not None and time.perf_counter() >= control.deadline:
                break
            current = pipeline if remaining is None else min(pipeline, remaining)
            reserve_commands = []
            for offset in range(current):
                nonce = request_nonce(
                    phase_id, connections, connection_id, sequence + offset
                )
                payload = reserve_request(
                    setup.domain,
                    client_nonce,
                    nonce,
                    setup.lease_ms,
                    setup.local_hashes,
                    setup.admission_candidates,
                )
                reserve_commands.append(
                    WorkCommand(
                        "select_reserve",
                        resp_command(b"DYNKV.SELECT_RESERVE", setup.key, payload),
                        reserve_validator(client_nonce, nonce, setup),
                        logical_payload_bytes=len(payload),
                        selection=True,
                    )
                )
            reservations = await send_batch(
                reader, writer, reserve_commands, counters, latency
            )

            if mode == "renew":
                renew_commands = []
                for reservation in reservations:
                    payload = renew_request(
                        setup.domain,
                        reservation.client_nonce,
                        reservation.request_nonce,
                        reservation.expires_at_ms,
                        setup.lease_ms,
                    )
                    renew_commands.append(
                        WorkCommand(
                            "renew",
                            resp_command(b"DYNKV.RENEW", setup.key, payload),
                            reserve_validator(
                                reservation.client_nonce,
                                reservation.request_nonce,
                                setup,
                            ),
                            logical_payload_bytes=len(payload),
                        )
                    )
                reservations = await send_batch(
                    reader, writer, renew_commands, counters, latency
                )

            release_commands = []
            for reservation in reservations:
                payload = release_request(
                    setup.domain,
                    reservation.client_nonce,
                    reservation.request_nonce,
                    reservation.expires_at_ms,
                )

                def release_check(response: Resp) -> int:
                    status = parse_release_response(scalar_bytes(response))
                    if status != 1:
                        raise ValueError("RELEASE did not remove its reservation")
                    return status

                release_commands.append(
                    WorkCommand(
                        "release",
                        resp_command(b"DYNKV.RELEASE", setup.key, payload),
                        release_check,
                        logical_payload_bytes=len(payload),
                    )
                )
            await send_batch(reader, writer, release_commands, counters, latency)
            counters.reservation_cycles += current
            counters.iterations += current
            sequence += current
            if remaining is not None:
                remaining -= current
    finally:
        completed_at = time.perf_counter()
        writer.close()
        with contextlib.suppress(ConnectionError):
            await writer.wait_closed()
    return ConnectionResult(counters, completed_at)


@dataclass(frozen=True)
class PhaseResult:
    counters: Counters
    elapsed_s: float


async def run_phase(
    *,
    port: int,
    setup: WorkloadSetup,
    mode: str,
    connections: int,
    pipeline: int,
    blocks_per_event: int,
    churn_prefixes_per_connection: int,
    count: int | None,
    duration_s: float | None,
    phase_id: int,
    latency: LatencyRecorder | None,
    on_ready: Callable[[], Any] | None = None,
) -> tuple[PhaseResult, Any]:
    ready: asyncio.Queue[None] = asyncio.Queue()
    control = PhaseControl()
    per_connection = remainder = 0
    if count is not None:
        per_connection, remainder = divmod(count, connections)
    tasks = []
    for connection_id in range(connections):
        connection_count = (
            None if count is None else per_connection + int(connection_id < remainder)
        )
        common = dict(
            port=port,
            setup=setup,
            mode=mode,
            connection_id=connection_id,
            connections=connections,
            count=connection_count,
            pipeline=pipeline,
            phase_id=phase_id,
            control=control,
            ready=ready,
            latency=latency,
        )
        if mode in LIFECYCLE_MODES:
            coroutine = run_lifecycle_connection(**common)
        else:
            coroutine = run_simple_connection(
                **common,
                blocks_per_event=blocks_per_event,
                churn_prefixes_per_connection=churn_prefixes_per_connection,
            )
        tasks.append(asyncio.create_task(coroutine))

    for _ in range(connections):
        await ready.get()
    ready_value = await on_ready() if on_ready is not None else None
    started = time.perf_counter()
    control.deadline = None if duration_s is None else started + duration_s
    control.start.set()
    results = await asyncio.gather(*tasks)
    completed_at = max((result.completed_at for result in results), default=started)
    counters = Counters()
    for result in results:
        counters.merge(result.counters)
    return PhaseResult(counters, max(completed_at - started, 1e-12)), ready_value
