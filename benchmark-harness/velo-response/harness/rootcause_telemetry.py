#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture packet-processing, NIC, IRQ, softirq, and TCP state once per second."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any


STOP = False


def stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def read_text(path: str | Path) -> str:
    try:
        return Path(path).read_text(errors="replace")
    except (FileNotFoundError, PermissionError, OSError):
        return ""


def command(argv: list[str], timeout: float = 5.0) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout, check=False
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"argv": argv, "error": f"{type(error).__name__}: {error}"}


def proc_stat() -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for line in read_text("/proc/stat").splitlines():
        fields = line.split()
        if fields and re.fullmatch(r"cpu\d*", fields[0]):
            result[fields[0]] = [int(value) for value in fields[1:]]
    return result


def named_matrix(path: str, names: set[str] | None = None) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for line in read_text(path).splitlines()[1:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        name = name.strip()
        if names is not None and name not in names:
            continue
        values: list[int] = []
        for field in raw.split():
            if not field.isdigit():
                break
            values.append(int(field))
        result[name] = values
    return result


def softnet() -> list[list[int]]:
    rows: list[list[int]] = []
    for line in read_text("/proc/net/softnet_stat").splitlines():
        try:
            rows.append([int(value, 16) for value in line.split()])
        except ValueError:
            continue
    return rows


def interrupts() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    lines = read_text("/proc/interrupts").splitlines()
    cpu_count = len(lines[0].split()) if lines else 0
    for line in lines[1:]:
        if ":" not in line:
            continue
        irq, raw = line.split(":", 1)
        fields = raw.split()
        if len(fields) < cpu_count:
            continue
        try:
            counts = [int(value) for value in fields[:cpu_count]]
        except ValueError:
            continue
        result[irq.strip()] = {
            "counts": counts,
            "description": " ".join(fields[cpu_count:]),
        }
    return result


def network() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    keys = (
        "rx_bytes",
        "rx_packets",
        "rx_errors",
        "rx_dropped",
        "rx_fifo",
        "rx_frame",
        "rx_compressed",
        "rx_multicast",
        "tx_bytes",
        "tx_packets",
        "tx_errors",
        "tx_dropped",
        "tx_fifo",
        "tx_collisions",
        "tx_carrier",
        "tx_compressed",
    )
    for line in read_text("/proc/net/dev").splitlines()[2:]:
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        values = [int(value) for value in raw.split()]
        result[name.strip()] = dict(zip(keys, values))
    return result


def protocol_counters(path: str) -> dict[str, dict[str, int]]:
    lines = read_text(path).splitlines()
    result: dict[str, dict[str, int]] = {}
    index = 0
    while index + 1 < len(lines):
        headers = lines[index].split()
        values = lines[index + 1].split()
        if headers and values and headers[0] == values[0]:
            try:
                result[headers[0].rstrip(":")] = {
                    name: int(value)
                    for name, value in zip(headers[1:], values[1:])
                }
            except ValueError:
                pass
            index += 2
        else:
            index += 1
    return result


def sockstat() -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for line in read_text("/proc/net/sockstat").splitlines():
        fields = line.replace(":", "").split()
        if not fields:
            continue
        values: dict[str, int] = {}
        for index in range(1, len(fields) - 1, 2):
            try:
                values[fields[index]] = int(fields[index + 1])
            except ValueError:
                continue
        result[fields[0]] = values
    return result


def ethtool_stats(interface: str) -> dict[str, int]:
    completed = command(["ethtool", "-S", interface], timeout=4.0)
    result: dict[str, int] = {}
    for line in str(completed.get("stdout", "")).splitlines():
        if ":" not in line:
            continue
        key, raw = line.strip().split(":", 1)
        try:
            result[key.strip()] = int(raw.strip(), 0)
        except ValueError:
            continue
    return result


def irq_affinity() -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for path in Path("/proc/irq").glob("[0-9]*"):
        result[path.name] = {
            "smp_affinity_list": read_text(path / "smp_affinity_list").strip(),
            "effective_affinity_list": read_text(
                path / "effective_affinity_list"
            ).strip(),
        }
    return result


def queue_configuration(interfaces: list[str]) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for interface in interfaces:
        queues: dict[str, dict[str, str]] = {}
        for path in Path(f"/sys/class/net/{interface}/queues").glob("*"):
            queue: dict[str, str] = {}
            for name in ("rps_cpus", "rps_flow_cnt", "xps_cpus", "xps_rxqs"):
                value = read_text(path / name).strip()
                if value:
                    queue[name] = value
            queues[path.name] = queue
        result[interface] = queues
    return result


def interface_irqs(interfaces: list[str]) -> dict[str, list[int]]:
    result: dict[str, list[int]] = {}
    for interface in interfaces:
        values: list[int] = []
        for path in Path(f"/sys/class/net/{interface}/device/msi_irqs").glob("[0-9]*"):
            try:
                values.append(int(path.name))
            except ValueError:
                continue
        result[interface] = sorted(values)
    return result


def rdma_snapshot() -> dict[str, Any]:
    result = {}
    for device in Path("/sys/class/infiniband").glob("*"):
        ports = {}
        for port in (device / "ports").glob("*"):
            counters = {}
            for group in ("counters", "hw_counters"):
                for counter in (port / group).glob("*"):
                    value = read_text(counter).strip()
                    if value.isdigit():
                        counters[f"{group}/{counter.name}"] = int(value)
            ports[port.name] = {
                "counters": counters,
                **{key: read_text(port / key).strip() for key in
                   ("state", "phys_state", "rate", "link_layer", "lid")},
            }
        result[device.name] = {
            "numa_node": read_text(device / "device/numa_node").strip(),
            "pci_device": str((device / "device").resolve()),
            "ports": ports,
        }
    return result


def static_snapshot(interfaces: list[str]) -> dict[str, Any]:
    return {
        "timestamp": timestamp(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "rdma": rdma_snapshot(),
        "rdma_devices": command(["ibv_devinfo", "-v"], timeout=10.0),
        "slurm": {
            key: os.environ.get(key)
            for key in (
                "SLURM_JOB_ID",
                "SLURM_NODEID",
                "SLURM_PROCID",
                "SLURMD_NODENAME",
            )
        },
        "interfaces": interfaces,
        "addresses": command(["ip", "-j", "address", "show"]),
        "routes": command(["ip", "-j", "route", "show", "table", "all"]),
        "links": command(["ip", "-j", "-details", "-statistics", "link", "show"]),
        "sysctl": command(
            [
                "sysctl",
                "net.core.netdev_budget",
                "net.core.netdev_budget_usecs",
                "net.core.netdev_max_backlog",
                "net.core.rps_sock_flow_entries",
                "net.ipv4.tcp_congestion_control",
                "net.ipv4.tcp_rmem",
                "net.ipv4.tcp_wmem",
            ]
        ),
        "irq_affinity": irq_affinity(),
        "interface_irqs": interface_irqs(interfaces),
        "queue_configuration": queue_configuration(interfaces),
        "interrupts": read_text("/proc/interrupts"),
        "softirqs": read_text("/proc/softirqs"),
        "softnet": read_text("/proc/net/softnet_stat"),
        "ethtool": {
            interface: {
                name: command(["ethtool", flag, interface], timeout=10.0)
                for name, flag in (
                    ("driver", "-i"),
                    ("rings", "-g"),
                    ("coalescing", "-c"),
                    ("channels", "-l"),
                    ("rss", "-x"),
                    ("features", "-k"),
                    ("pause", "-a"),
                )
            }
            for interface in interfaces
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--interface", action="append", default=[])
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--socket-interval", type=float, default=5.0)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    (args.output_dir / "static.json").write_text(
        json.dumps(static_snapshot(args.interface), indent=2, sort_keys=True) + "\n"
    )
    next_sample = time.monotonic()
    next_sockets = next_sample
    with (args.output_dir / "samples.jsonl").open("w") as samples, (
        args.output_dir / "sockets.jsonl"
    ).open("w") as sockets:
        while not STOP:
            now = time.monotonic()
            sample = {
                "timestamp": timestamp(),
                "monotonic_seconds": now,
                "host": socket.gethostname(),
                "clock_ticks_per_second": os.sysconf("SC_CLK_TCK"),
                "proc_stat": proc_stat(),
                "softirqs": named_matrix(
                    "/proc/softirqs", {"NET_RX", "NET_TX", "TIMER", "SCHED"}
                ),
                "softnet": softnet(),
                "interrupts": interrupts(),
                "network": network(),
                "ethtool": {
                    interface: ethtool_stats(interface)
                    for interface in args.interface
                },
                "rdma": rdma_snapshot(),
                "snmp": protocol_counters("/proc/net/snmp"),
                "netstat": protocol_counters("/proc/net/netstat"),
                "sockstat": sockstat(),
                "cpu_pressure": read_text("/proc/pressure/cpu").strip(),
                "ss_summary": command(["ss", "-s"], timeout=4.0),
            }
            samples.write(json.dumps(sample, sort_keys=True) + "\n")
            samples.flush()
            if now >= next_sockets:
                snapshot = command(
                    ["ss", "-tinmopH", "state", "established"], timeout=15.0
                )
                snapshot.update(
                    {
                        "timestamp": sample["timestamp"],
                        "monotonic_seconds": now,
                        "host": sample["host"],
                    }
                )
                sockets.write(json.dumps(snapshot, sort_keys=True) + "\n")
                sockets.flush()
                next_sockets += args.socket_interval
            next_sample += args.interval
            time.sleep(max(0.0, next_sample - time.monotonic()))
    (args.output_dir / "stopped-at.txt").write_text(timestamp() + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
