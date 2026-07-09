# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import socket
import threading
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

from .common import VALKEY_TEARDOWN_FAILURE_MARKERS


def _read_line(sock: socket.socket) -> bytes:
    value = bytearray()
    while not value.endswith(b"\r\n"):
        chunk = sock.recv(1)
        if not chunk:
            raise RuntimeError("unexpected EOF reading Valkey response")
        value.extend(chunk)
    return bytes(value[:-2])


def _read_exact(sock: socket.socket, size: int) -> bytes:
    value = bytearray()
    while len(value) < size:
        chunk = sock.recv(size - len(value))
        if not chunk:
            raise RuntimeError("unexpected EOF reading Valkey response body")
        value.extend(chunk)
    return bytes(value)


def valkey_info(port: int, section: str) -> str:
    """Return a small INFO reply without depending on redis-py."""

    command = (b"INFO", section.encode())
    request = bytearray(f"*{len(command)}\r\n".encode())
    for part in command:
        request.extend(f"${len(part)}\r\n".encode())
        request.extend(part)
        request.extend(b"\r\n")

    with socket.create_connection(("127.0.0.1", port), timeout=1) as sock:
        sock.sendall(request)
        marker = _read_exact(sock, 1)
        if marker == b"-":
            raise RuntimeError(_read_line(sock).decode(errors="replace"))
        if marker != b"$":
            raise RuntimeError(f"unexpected INFO response marker {marker!r}")
        length = int(_read_line(sock))
        if length < 0:
            return ""
        response = _read_exact(sock, length)
        if _read_exact(sock, 2) != b"\r\n":
            raise RuntimeError("INFO response did not end in CRLF")
        return response.decode(errors="replace")


def valkey_integer_command(port: int, *parts: str) -> int:
    """Send a small RESP command which is expected to return an integer."""

    request = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        encoded = part.encode()
        request.extend(f"${len(encoded)}\r\n".encode())
        request.extend(encoded)
        request.extend(b"\r\n")

    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request)
        marker = _read_exact(sock, 1)
        if marker == b"-":
            raise RuntimeError(_read_line(sock).decode(errors="replace"))
        if marker != b":":
            raise RuntimeError(f"unexpected integer response marker {marker!r}")
        return int(_read_line(sock))


def valkey_integer_array_command(port: int, *parts: str) -> list[int]:
    """Send a small RESP command which is expected to return integer entries."""

    request = bytearray(f"*{len(parts)}\r\n".encode())
    for part in parts:
        encoded = part.encode()
        request.extend(f"${len(encoded)}\r\n".encode())
        request.extend(encoded)
        request.extend(b"\r\n")

    with socket.create_connection(("127.0.0.1", port), timeout=5) as sock:
        sock.sendall(request)
        marker = _read_exact(sock, 1)
        if marker == b"-":
            raise RuntimeError(_read_line(sock).decode(errors="replace"))
        if marker != b"*":
            raise RuntimeError(f"unexpected array response marker {marker!r}")
        count = int(_read_line(sock))
        values: list[int] = []
        for _ in range(count):
            if _read_exact(sock, 1) != b":":
                raise RuntimeError("expected integer entry in Valkey array response")
            values.append(int(_read_line(sock)))
        return values


def info_fields(info: str) -> dict[str, str]:
    return {
        line.split(":", 1)[0]: line.split(":", 1)[1]
        for line in info.splitlines()
        if ":" in line
    }


def sample_valkey_client_pressure(
    stop: threading.Event,
    ports: list[int],
    *,
    interval_seconds: float = 0.05,
    info_reader: Callable[[int, str], str] = valkey_info,
) -> dict[str, Any]:
    """Sample probe-inclusive Valkey client peaks until ``stop`` is set."""

    if interval_seconds < 0:
        raise ValueError("client pressure sample interval cannot be negative")
    unique_ports = tuple(dict.fromkeys(ports))
    pressure: dict[str, dict[str, int]] = {str(port): {} for port in unique_ports}
    sample_rounds = 0
    successful_reads = 0
    read_errors = 0
    while not stop.is_set():
        sample_rounds += 1
        for port in unique_ports:
            try:
                clients = info_fields(info_reader(port, "clients"))
                connected = int(clients["connected_clients"])
                blocked = int(clients["blocked_clients"])
                maxclients = int(clients["maxclients"])
                if connected < 0 or blocked < 0 or maxclients <= 0:
                    raise ValueError("Valkey client counters are out of range")
            except (KeyError, OSError, RuntimeError, ValueError):
                read_errors += 1
                continue
            successful_reads += 1
            port_pressure = pressure[str(port)]
            port_pressure["peak_connected_clients"] = max(
                connected, port_pressure.get("peak_connected_clients", 0)
            )
            port_pressure["peak_blocked_clients"] = max(
                blocked, port_pressure.get("peak_blocked_clients", 0)
            )
            port_pressure["maxclients"] = maxclients
        stop.wait(interval_seconds)
    return {
        "probe_inclusive": True,
        "sample_interval_seconds": interval_seconds,
        "sample_rounds": sample_rounds,
        "successful_reads": successful_reads,
        "read_errors": read_errors,
        "ports": pressure,
    }


@contextmanager
def observe_valkey_client_pressure(
    ports: list[int], *, interval_seconds: float = 0.05
) -> Iterator[dict[str, Any]]:
    """Run the bounded client-pressure sampler in the benchmark background."""

    stop = threading.Event()
    result: dict[str, Any] = {}

    def sample() -> None:
        try:
            result.update(
                sample_valkey_client_pressure(
                    stop, ports, interval_seconds=interval_seconds
                )
            )
        except Exception as error:  # Preserve the benchmark result for diagnosis.
            result.update(
                {
                    "error": f"{type(error).__name__}: {error}",
                    "ports": {},
                }
            )

    sampler = threading.Thread(
        target=sample,
        name="valkey-client-pressure-sampler",
        daemon=True,
    )
    sampler.start()
    try:
        yield result
    finally:
        stop.set()
        sampler.join(timeout=5.0)
        if sampler.is_alive():
            result.update(
                {
                    "error": "client pressure sampler did not stop within 5 seconds",
                    "ports": {},
                }
            )


def valkey_state(primary_port: int, replica_port: int) -> dict[str, Any]:
    """Capture HA and module command counters without needing a Valkey client."""

    primary_commandstats = info_fields(valkey_info(primary_port, "commandstats"))
    replica_commandstats = info_fields(valkey_info(replica_port, "commandstats"))

    def relevant_commandstats(stats: Mapping[str, str]) -> dict[str, str]:
        return {
            name.removeprefix("cmdstat_"): value
            for name, value in stats.items()
            if any(
                (
                    name.startswith("cmdstat_dynkv"),
                    name in {"cmdstat_wait", "cmdstat_waitaof"},
                )
            )
        }

    return {
        "primary_clients": info_fields(valkey_info(primary_port, "clients")),
        "replica_clients": info_fields(valkey_info(replica_port, "clients")),
        "primary_replication": info_fields(valkey_info(primary_port, "replication")),
        "replica_replication": info_fields(valkey_info(replica_port, "replication")),
        "primary_commandstats": relevant_commandstats(primary_commandstats),
        "replica_commandstats": relevant_commandstats(replica_commandstats),
    }


def valkey_singleton_state(port: int, index_key: str) -> dict[str, Any]:
    """Capture the promoted singleton after the old primary was killed."""

    commandstats = info_fields(valkey_info(port, "commandstats"))
    relevant = {
        name.removeprefix("cmdstat_"): value
        for name, value in commandstats.items()
        if name.startswith("cmdstat_dynkv")
        or name in {"cmdstat_wait", "cmdstat_waitaof"}
    }
    return {
        "port": port,
        "clients": info_fields(valkey_info(port, "clients")),
        "replication": info_fields(valkey_info(port, "replication")),
        "commandstats": relevant,
        "stats": valkey_integer_array_command(port, "DYNKV.STATS", index_key),
        "lifecycle_stats": valkey_integer_array_command(
            port, "DYNKV.LIFECYCLE_STATS", index_key
        ),
        "admission_stats": valkey_integer_command(
            port, "DYNKV.ADMISSION_STATS", index_key
        ),
    }


def commandstats_fields(value: Any) -> dict[str, str] | None:
    if not isinstance(value, str):
        return None
    fields: dict[str, str] = {}
    for item in value.split(","):
        key, separator, field_value = item.partition("=")
        if not separator or not key:
            return None
        fields[key] = field_value
    return fields


def integer_field(fields: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = fields.get(name)
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def valkey_ha_validation_errors(
    state: Any, *, authoritative_admission: bool
) -> list[str]:
    """Validate final replication health and module command outcomes."""

    errors: list[str] = []
    if not isinstance(state, Mapping):
        return ["final Valkey HA state is missing"]
    primary = state.get("primary_replication")
    replica = state.get("replica_replication")
    if not isinstance(primary, Mapping):
        errors.append("final Valkey primary replication state is missing")
    if not isinstance(replica, Mapping):
        errors.append("final Valkey replica replication state is missing")
    if isinstance(primary, Mapping) and isinstance(replica, Mapping):
        if primary.get("role") != "master":
            errors.append(f"final Valkey primary role is {primary.get('role')!r}")
        connected = integer_field(primary, "connected_slaves", "connected_replicas")
        if connected is None or connected < 1:
            errors.append("final Valkey primary has no connected replica")
        has_online_replica = any(
            str(key).startswith(("slave", "replica")) and "state=online" in str(value)
            for key, value in primary.items()
        )
        good = integer_field(primary, "min_slaves_good_slaves")
        if not ((good is not None and good >= 1) or has_online_replica):
            errors.append("final Valkey primary has no online/good replica")
        if replica.get("role") not in {"slave", "replica"}:
            errors.append(f"final Valkey replica role is {replica.get('role')!r}")
        if replica.get("master_link_status") != "up":
            errors.append("final Valkey replica link is not up")
        if replica.get("master_sync_in_progress") != "0":
            errors.append("final Valkey replica is still synchronizing")
        primary_replid = primary.get("master_replid")
        replica_replid = replica.get("master_replid")
        if not primary_replid or primary_replid != replica_replid:
            errors.append("final Valkey primary/replica replication IDs differ")

    commandstats_by_role: list[tuple[str, Mapping[str, Any]]] = []
    for role, key in (
        ("primary", "primary_commandstats"),
        ("replica", "replica_commandstats"),
    ):
        commandstats = state.get(key)
        if not isinstance(commandstats, Mapping):
            errors.append(f"final Valkey {role} commandstats are missing")
        else:
            commandstats_by_role.append((role, commandstats))
            for command, raw_fields in commandstats.items():
                fields = commandstats_fields(raw_fields)
                if fields is None:
                    errors.append(
                        f"could not parse {role} commandstats for {command}: {raw_fields!r}"
                    )
                    continue
                for failure_field in ("failed_calls", "rejected_calls"):
                    value = integer_field(fields, failure_field)
                    if value is None:
                        errors.append(
                            f"{role} commandstats for {command} has no valid {failure_field}"
                        )
                    elif value != 0:
                        errors.append(
                            f"{role} commandstats for {command} reports "
                            f"{failure_field}={value}"
                        )

    primary_commandstats = next(
        (stats for role, stats in commandstats_by_role if role == "primary"), {}
    )
    required_commands = {
        "dynkv.apply_owned",
        "dynkv.register_worker_ranks",
        "wait",
    }
    if authoritative_admission:
        required_commands.update(("dynkv.select_reserve", "dynkv.release"))
    missing_commands = sorted(required_commands - set(primary_commandstats))
    if missing_commands:
        errors.append(
            "final Valkey primary commandstats are missing required commands: "
            + ", ".join(missing_commands)
        )

    replica_commandstats = next(
        (stats for role, stats in commandstats_by_role if role == "replica"), {}
    )
    required_replica_commands = {
        "dynkv.apply_owned_at",
        "dynkv.worker_lease_apply",
    }
    if authoritative_admission:
        required_replica_commands.add("dynkv.admit_apply")
    missing_replica_commands = sorted(
        required_replica_commands - set(replica_commandstats)
    )
    if missing_replica_commands:
        errors.append(
            "final Valkey replica commandstats are missing required commands: "
            + ", ".join(missing_replica_commands)
        )
    return errors


def valkey_singleton_validation_errors(
    state: Any, *, expected_ranks: int, authoritative_admission: bool
) -> list[str]:
    """Validate state after a two-node failover deliberately enters ack-0 mode."""

    errors: list[str] = []
    if not isinstance(state, Mapping):
        return ["final promoted Valkey state is missing"]
    replication = state.get("replication")
    if not isinstance(replication, Mapping) or replication.get("role") != "master":
        errors.append("surviving Valkey node is not the promoted master")
    stats = state.get("stats")
    if not isinstance(stats, list) or len(stats) != 3:
        errors.append(f"promoted DYNKV.STATS reply is invalid: {stats!r}")
    elif stats[1] != expected_ranks:
        errors.append(
            f"promoted Valkey registered ranks={stats[1]}, expected={expected_ranks}"
        )
    if state.get("admission_stats") != 0:
        errors.append(
            f"promoted Valkey leaked admission reservations: {state.get('admission_stats')!r}"
        )

    commandstats = state.get("commandstats")
    if not isinstance(commandstats, Mapping):
        errors.append("promoted Valkey commandstats are missing")
        return errors
    for command, raw_fields in commandstats.items():
        fields = commandstats_fields(raw_fields)
        if fields is None:
            errors.append(
                f"could not parse promoted commandstats for {command}: {raw_fields!r}"
            )
            continue
        for failure_field in ("failed_calls", "rejected_calls"):
            value = integer_field(fields, failure_field)
            if value is not None and value != 0:
                errors.append(
                    f"promoted commandstats for {command} reports {failure_field}={value}"
                )
    required = {"dynkv.apply_owned", "wait"}
    if authoritative_admission:
        required.update(("dynkv.select_reserve", "dynkv.release"))
    missing = sorted(required - set(commandstats))
    if missing:
        errors.append(
            "promoted Valkey commandstats are missing post-failover commands: "
            + ", ".join(missing)
        )
    return errors


def scan_valkey_teardown_logs(logs_dir: Path) -> dict[str, Any]:
    """Collect owner-fencing/publisher failures emitted during teardown."""

    counts = {marker: 0 for marker in VALKEY_TEARDOWN_FAILURE_MARKERS}
    examples: list[dict[str, Any]] = []
    files_scanned = 0
    scan_errors = 0
    for path in sorted(logs_dir.rglob("*.log.txt")):
        files_scanned += 1
        try:
            lines = path.read_text(errors="replace").splitlines()
        except OSError as error:
            scan_errors += 1
            examples.append(
                {
                    "path": str(path),
                    "line": None,
                    "text": f"could not read teardown log: {error}",
                }
            )
            continue
        for line_number, line in enumerate(lines, start=1):
            for marker in VALKEY_TEARDOWN_FAILURE_MARKERS:
                if marker not in line:
                    continue
                counts[marker] += 1
                if len(examples) < 20:
                    examples.append(
                        {
                            "path": str(path),
                            "line": line_number,
                            "marker": marker,
                            "text": line,
                        }
                    )
    return {
        "files_scanned": files_scanned,
        "scan_errors": scan_errors,
        "failure_count": sum(counts.values()) + scan_errors,
        "counts": counts,
        "examples": examples,
    }


def wait_for_registered_workers(
    primary_port: int,
    index_key: str,
    expected_ranks: int,
    timeout_seconds: float,
) -> int:
    """Do not admit traffic until direct worker registration is visible.

    A worker rank writes ``DYNKV.REGISTER_WORKER`` before its first event.
    ``DYNKV.STATS`` counts `(worker_id, dp_rank)` records, not physical worker
    processes. Waiting for all ranks removes the startup race from a benchmark
    arm, so a measured 529 is a routing/admission result rather than a topology
    convergence artifact.
    """

    deadline = time.monotonic() + timeout_seconds
    last_error = "not queried"
    while time.monotonic() < deadline:
        try:
            stats = valkey_integer_array_command(primary_port, "DYNKV.STATS", index_key)
            if len(stats) != 3:
                last_error = f"unexpected DYNKV.STATS response {stats!r}"
            elif stats[1] >= expected_ranks:
                return stats[1]
            else:
                last_error = f"registered ranks={stats[1]}, expected={expected_ranks}"
        except (OSError, RuntimeError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.1)
    raise TimeoutError(
        f"Timed out waiting {timeout_seconds:.1f}s for direct Valkey worker-rank registration: "
        f"{last_error}"
    )


def wait_for_valkey_replica(
    primary_port: int, replica_port: int, timeout_seconds: float
) -> None:
    """Wait for an online replica and a successful primary ``WAIT 1`` barrier."""

    deadline = time.monotonic() + timeout_seconds
    last_error = "not queried"
    while time.monotonic() < deadline:
        try:
            fields = info_fields(valkey_info(primary_port, "replication"))
            replicas = int(
                fields.get("connected_slaves", fields.get("connected_replicas", "0"))
            )
            replica_fields = info_fields(valkey_info(replica_port, "replication"))
            link_status = replica_fields.get("master_link_status", "unknown")
            if replicas >= 1 and link_status == "up":
                acknowledged = valkey_integer_command(primary_port, "WAIT", "1", "3000")
                if acknowledged >= 1:
                    return
                last_error = f"WAIT acknowledged {acknowledged} replica(s)"
            else:
                last_error = (
                    f"primary replicas={replicas}, "
                    f"replica master_link_status={link_status}"
                )
        except (OSError, RuntimeError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.2)
    raise TimeoutError(
        f"Timed out waiting {timeout_seconds:.1f}s for Valkey replica: {last_error}"
    )


def wait_for_zero_admission_reservations(
    primary_port: int,
    replica_port: int,
    index_key: str,
    timeout_seconds: float,
) -> dict[str, int]:
    """Require all completed requests to release primary and replica leases."""

    deadline = time.monotonic() + timeout_seconds
    last_counts: dict[str, int] | None = None
    last_error = "not queried"
    while time.monotonic() < deadline:
        try:
            primary = valkey_integer_command(
                primary_port, "DYNKV.ADMISSION_STATS", index_key
            )
            replica = valkey_integer_command(
                replica_port, "DYNKV.ADMISSION_STATS", index_key
            )
            last_counts = {"primary": primary, "replica": replica}
            if primary == 0 and replica == 0:
                return last_counts
        except (OSError, RuntimeError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.1)
    if last_counts is not None:
        return last_counts
    raise TimeoutError(
        f"Could not query Valkey admission reservations before timeout: {last_error}"
    )


def wait_for_zero_singleton_admission_reservations(
    port: int, index_key: str, timeout_seconds: float
) -> int:
    deadline = time.monotonic() + timeout_seconds
    last_count: int | None = None
    last_error = "not queried"
    while time.monotonic() < deadline:
        try:
            last_count = valkey_integer_command(
                port, "DYNKV.ADMISSION_STATS", index_key
            )
            if last_count == 0:
                return 0
        except (OSError, RuntimeError, ValueError) as error:
            last_error = str(error)
        time.sleep(0.1)
    if last_count is not None:
        return last_count
    raise TimeoutError(
        f"Could not query promoted Valkey admission reservations: {last_error}"
    )
