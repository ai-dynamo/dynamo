# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import asyncio
import contextlib
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .protocol import Resp, resp_command, scalar_bytes
from .workload import execute

async def wait_ready(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            log_path = Path(process.dynkv_log_path)  # type: ignore[attr-defined]
            log = (
                log_path.read_text(errors="replace")
                if log_path.exists()
                else "<no server log>"
            )
            raise RuntimeError(f"Valkey exited during startup:\n{log}")
        try:
            response = await execute(port, b"PING")
            if scalar_bytes(response) == b"PONG":
                return
        except (ConnectionError, OSError, asyncio.IncompleteReadError):
            await asyncio.sleep(0.02)
    raise RuntimeError("Valkey did not become ready")


def start_server(
    server: Path,
    module: Path,
    directory: Path,
    port: int,
    appendonly: bool,
    appendfsync: str,
    auto_aof_rewrite_percentage: int,
) -> subprocess.Popen[bytes]:
    command = [
        str(server),
        "--port",
        str(port),
        "--bind",
        "127.0.0.1",
        "--dir",
        str(directory),
        "--appendonly",
        "yes" if appendonly else "no",
        "--appendfsync",
        appendfsync,
        "--auto-aof-rewrite-percentage",
        str(auto_aof_rewrite_percentage),
        "--save",
        "",
        "--loadmodule",
        str(module),
    ]
    log_path = directory / "valkey.log"
    with log_path.open("wb") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    process.dynkv_log_path = str(log_path)  # type: ignore[attr-defined]
    return process


async def shutdown(port: int, process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        with contextlib.suppress(ConnectionError, OSError, asyncio.IncompleteReadError):
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
            writer.write(resp_command(b"SHUTDOWN", b"NOSAVE"))
            await writer.drain()
            writer.close()
            await writer.wait_closed()
        with contextlib.suppress(subprocess.TimeoutExpired):
            process.wait(timeout=5)
    if process.poll() is None:
        process.kill()
        process.wait(timeout=5)


def parse_info(payload: bytes) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in payload.decode(errors="replace").splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition(":")
        if separator:
            fields[key] = value
    return fields


async def info(port: int, section: bytes) -> dict[str, str]:
    response = await execute(port, b"INFO", section)
    return parse_info(scalar_bytes(response))


def resp_array_values(response: Resp) -> list[Any]:
    if response.marker != b"*" or not isinstance(response.value, list):
        raise ValueError("expected RESP array")
    return [item.value for item in response.value]


async def config_facts(port: int) -> dict[str, str]:
    response = await execute(
        port,
        b"CONFIG",
        b"GET",
        b"appendonly",
        b"appendfsync",
        b"auto-aof-rewrite-percentage",
        b"dir",
    )
    values = resp_array_values(response)
    if len(values) % 2 or not all(isinstance(value, bytes) for value in values):
        raise ValueError("unexpected CONFIG GET response")
    return {
        values[index].decode(): values[index + 1].decode()
        for index in range(0, len(values), 2)
    }


def proc_cpu_seconds(pid: int) -> float | None:
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
        fields = text[text.rfind(")") + 2 :].split()
        ticks = int(fields[11]) + int(fields[12])
        return ticks / os.sysconf("SC_CLK_TCK")
    except (FileNotFoundError, IndexError, OSError, ValueError):
        return None


def proc_memory_kib(pid: int) -> dict[str, int]:
    result: dict[str, int] = {}
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            name, separator, value = line.partition(":")
            if separator and name in {"VmRSS", "VmHWM"}:
                result[name] = int(value.strip().split()[0])
    except (FileNotFoundError, OSError, ValueError):
        pass
    return result


def integer_response(response: Resp) -> int:
    if not isinstance(response.value, int):
        raise ValueError("expected integer response")
    return response.value


def integer_array(response: Resp) -> list[int]:
    values = resp_array_values(response)
    if not all(isinstance(value, int) for value in values):
        raise ValueError("expected integer array")
    return values


@dataclass(frozen=True)
class Telemetry:
    commandstats: dict[str, str]
    server: dict[str, str]
    memory: dict[str, str]
    persistence: dict[str, str]
    stats: dict[str, str]
    module_stats: list[int]
    admission_reservations: int
    lifecycle_stats: list[int]
    gc_stats: list[int] | None
    config: dict[str, str]
    process_cpu_s: float | None
    client_cpu_s: float
    process_memory_kib: dict[str, int]


async def capture_telemetry(
    port: int,
    process: subprocess.Popen[bytes],
    key: bytes,
    *,
    before: bool,
    include_gc_stats: bool = False,
) -> Telemetry:
    async def module_values() -> tuple[list[int], int, list[int], list[int] | None]:
        return (
            integer_array(await execute(port, b"DYNKV.STATS", key)),
            integer_response(await execute(port, b"DYNKV.ADMISSION_STATS", key)),
            integer_array(await execute(port, b"DYNKV.LIFECYCLE_STATS", key)),
            (
                integer_array(await execute(port, b"DYNKV.GC_STATS", key))
                if include_gc_stats
                else None
            ),
        )

    if before:
        module_stats, admission, lifecycle, gc_stats = await module_values()
        memory = await info(port, b"memory")
        persistence = await info(port, b"persistence")
        stats = await info(port, b"stats")
        server = await info(port, b"server")
        config = await config_facts(port)
        commandstats = await info(port, b"commandstats")
        process_cpu = proc_cpu_seconds(process.pid)
        client_cpu = time.process_time()
        process_memory = proc_memory_kib(process.pid)
    else:
        process_cpu = proc_cpu_seconds(process.pid)
        client_cpu = time.process_time()
        process_memory = proc_memory_kib(process.pid)
        commandstats = await info(port, b"commandstats")
        module_stats, admission, lifecycle, gc_stats = await module_values()
        memory = await info(port, b"memory")
        persistence = await info(port, b"persistence")
        stats = await info(port, b"stats")
        server = await info(port, b"server")
        config = await config_facts(port)
    return Telemetry(
        commandstats={
            key: value
            for key, value in commandstats.items()
            if key.startswith("cmdstat_dynkv.")
        },
        server=server,
        memory=memory,
        persistence=persistence,
        stats=stats,
        module_stats=module_stats,
        admission_reservations=admission,
        lifecycle_stats=lifecycle,
        gc_stats=gc_stats,
        config=config,
        process_cpu_s=process_cpu,
        client_cpu_s=client_cpu,
        process_memory_kib=process_memory,
    )


def parse_commandstat(value: str | None) -> dict[str, float]:
    result: dict[str, float] = {}
    if value is None:
        return result
    for part in value.split(","):
        key, separator, raw_value = part.partition("=")
        if not separator:
            continue
        with contextlib.suppress(ValueError):
            result[key] = float(raw_value)
    return result


def commandstats_delta(
    before: dict[str, str], after: dict[str, str]
) -> dict[str, dict[str, int | float]]:
    result: dict[str, dict[str, int | float]] = {}
    for command in sorted(set(before) | set(after)):
        earlier = parse_commandstat(before.get(command))
        later = parse_commandstat(after.get(command))
        fields: dict[str, int | float] = {}
        for field_name in ("calls", "usec", "rejected_calls", "failed_calls"):
            delta = later.get(field_name, 0) - earlier.get(field_name, 0)
            fields[field_name] = int(delta)
        calls = int(fields["calls"])
        fields["usec_per_call"] = fields["usec"] / calls if calls else 0.0
        result[command.removeprefix("cmdstat_")] = fields
    return result


def int_field(fields: dict[str, str], name: str) -> int | None:
    try:
        return int(fields[name])
    except (KeyError, ValueError):
        return None


def delta_or_none(
    after: int | float | None, before: int | float | None
) -> float | None:
    if after is None or before is None:
        return None
    return float(after - before)


def filesystem_facts(path_value: str | None) -> dict[str, int | str] | None:
    if path_value is None:
        return None
    path = Path(path_value)
    try:
        stat = path.stat()
        filesystem = os.statvfs(path)
    except OSError:
        return None
    return {
        "path": str(path.resolve()),
        "device": stat.st_dev,
        "block_size": filesystem.f_frsize,
        "total_bytes": filesystem.f_blocks * filesystem.f_frsize,
        "available_bytes": filesystem.f_bavail * filesystem.f_frsize,
    }


def telemetry_summary(
    before: Telemetry, after: Telemetry, elapsed_s: float
) -> dict[str, Any]:
    commandstats = commandstats_delta(before.commandstats, after.commandstats)
    failed = {
        command: fields
        for command, fields in commandstats.items()
        if fields.get("failed_calls", 0) or fields.get("rejected_calls", 0)
    }
    if failed:
        raise RuntimeError(f"Valkey commandstats reported failures: {failed!r}")
    server_cpu_delta = delta_or_none(after.process_cpu_s, before.process_cpu_s)
    client_cpu_delta = after.client_cpu_s - before.client_cpu_s
    aof_before = int_field(before.persistence, "aof_current_size")
    aof_after = int_field(after.persistence, "aof_current_size")
    return {
        "server": {
            "valkey_version": after.server.get("valkey_version"),
            "redis_version": after.server.get("redis_version"),
            "process_id": after.server.get("process_id"),
            "arch_bits": after.server.get("arch_bits"),
            "multiplexing_api": after.server.get("multiplexing_api"),
            "executable": after.server.get("executable"),
        },
        "commandstats": commandstats,
        "server_cpu_seconds": server_cpu_delta,
        "server_cpu_percent_of_one_core": (
            server_cpu_delta / elapsed_s * 100 if server_cpu_delta is not None else None
        ),
        "client_cpu_seconds": client_cpu_delta,
        "client_cpu_percent_of_one_core": client_cpu_delta / elapsed_s * 100,
        "memory": {
            "used_memory_before": int_field(before.memory, "used_memory"),
            "used_memory_after": int_field(after.memory, "used_memory"),
            "used_memory_peak_after": int_field(after.memory, "used_memory_peak"),
            "process_before_kib": before.process_memory_kib,
            "process_after_kib": after.process_memory_kib,
        },
        "aof": {
            "config": after.config,
            "enabled": after.persistence.get("aof_enabled"),
            "current_size_before": aof_before,
            "current_size_after": aof_after,
            "size_delta": (
                aof_after - aof_before
                if aof_after is not None and aof_before is not None
                else None
            ),
            "base_size_after": int_field(after.persistence, "aof_base_size"),
            "rewrite_in_progress_after": after.persistence.get(
                "aof_rewrite_in_progress"
            ),
            "rewrite_scheduled_after": after.persistence.get("aof_rewrite_scheduled"),
            "last_write_status": after.persistence.get("aof_last_write_status"),
            "filesystem": filesystem_facts(after.config.get("dir")),
        },
        "instantaneous_ops_per_sec_after": int_field(
            after.stats, "instantaneous_ops_per_sec"
        ),
        "module_stats_before": before.module_stats,
        "module_stats_after": after.module_stats,
        "admission_reservations_before": before.admission_reservations,
        "admission_reservations_after": after.admission_reservations,
        "lifecycle_stats_before": before.lifecycle_stats,
        "lifecycle_stats_after": after.lifecycle_stats,
        "gc_stats_before": before.gc_stats,
        "gc_stats_after": after.gc_stats,
    }
