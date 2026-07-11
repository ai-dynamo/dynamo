#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Start one Valkey identity using Sentinel as the post-bootstrap authority."""

from __future__ import annotations

import argparse
import collections
import os
import pathlib
import signal
import socket
import subprocess
import time
from collections.abc import Callable, Sequence


Master = tuple[str, int]
SENTINELS = tuple(
    f"valkey-sweep-sentinel-{ordinal}.valkey-sweep-sentinel" for ordinal in range(3)
)


def sentinel_master(master_name: str, allowed_hosts: frozenset[str]) -> Master | None:
    votes: list[Master] = []
    for host in SENTINELS:
        try:
            result = subprocess.run(
                [
                    "/usr/local/bin/valkey-cli",
                    "--raw",
                    "-h",
                    host,
                    "-p",
                    "26379",
                    "SENTINEL",
                    "get-master-addr-by-name",
                    master_name,
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            values = [value for value in result.stdout.splitlines() if value]
            if (
                len(values) == 2
                and values[0] in allowed_hosts
                and int(values[1]) == 6379
            ):
                votes.append((values[0], int(values[1])))
        except (OSError, ValueError, subprocess.SubprocessError):
            continue
    if not votes:
        return None
    master, count = collections.Counter(votes).most_common(1)[0]
    return master if count >= 2 else None


def wait_for_master(
    master_name: str,
    *,
    allowed_hosts: frozenset[str],
    initialized: bool,
    bootstrap_timeout: float = 15,
    lookup: Callable[[str, frozenset[str]], Master | None] = sentinel_master,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> Master | None:
    """Wait forever after bootstrap; only a new identity has a fallback deadline."""
    deadline = None if initialized else monotonic() + bootstrap_timeout
    while True:
        master = lookup(master_name, allowed_hosts)
        if master is not None:
            return master
        if deadline is not None and monotonic() >= deadline:
            return None
        sleep(1)


def resolves_to_self(host: str, pod_ip: str, pod_dns: str) -> bool:
    if host in {pod_ip, pod_dns}:
        return True
    try:
        addresses = {
            result[4][0]
            for result in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
        }
    except socket.gaierror:
        return False
    return pod_ip in addresses


def role_arguments(
    master: Master | None,
    *,
    bootstrap_role: str,
    bootstrap_primary_host: str,
    pod_ip: str,
    pod_dns: str,
) -> list[str]:
    if master is not None:
        return (
            []
            if resolves_to_self(master[0], pod_ip, pod_dns)
            else ["--replicaof", master[0], str(master[1])]
        )
    if bootstrap_role == "replica":
        return ["--replicaof", bootstrap_primary_host, "6379"]
    if bootstrap_role == "primary":
        return []
    raise ValueError(f"invalid bootstrap role: {bootstrap_role}")


def mark_initialized(state_dir: pathlib.Path) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    marker = state_dir / "bootstrap-complete"
    if marker.is_file():
        return
    temporary = state_dir / "bootstrap-complete.tmp"
    temporary.write_text("initialized\n", encoding="utf-8")
    os.replace(temporary, marker)


def wait_for_server_ready(
    process: subprocess.Popen[str],
    *,
    timeout: float = 30,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Require a live local server before committing bootstrap state."""
    deadline = monotonic() + timeout
    while process.poll() is None and monotonic() < deadline:
        try:
            result = subprocess.run(
                ["/usr/local/bin/valkey-cli", "--raw", "PING"],
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            )
            if result.stdout.strip() == "PONG":
                return
        except (OSError, subprocess.SubprocessError):
            pass
        sleep(1)
    if process.poll() is not None:
        raise RuntimeError(f"Valkey exited during bootstrap: {process.returncode}")
    raise RuntimeError("Valkey did not become ready during bootstrap")


def terminate_child(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def wait_for_child(process: subprocess.Popen[str]) -> int:
    """Keep the launcher as PID 1 while forwarding normal termination signals."""

    def forward_signal(signum: int, _frame: object) -> None:
        if process.poll() is None:
            process.send_signal(signum)

    signals = (signal.SIGINT, signal.SIGTERM)
    previous = {signum: signal.signal(signum, forward_signal) for signum in signals}
    try:
        return process.wait()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def start_initial_server(
    command: list[str],
    *,
    state_dir: pathlib.Path,
    master_name: str,
    allowed_hosts: frozenset[str],
) -> int:
    """Start, verify, and adopt an initial server before marking it durable."""
    process = subprocess.Popen(command)
    try:
        wait_for_server_ready(process)
        if (
            wait_for_master(
                master_name,
                allowed_hosts=allowed_hosts,
                initialized=False,
                bootstrap_timeout=30,
            )
            is None
        ):
            raise RuntimeError("Sentinel did not adopt initial Valkey bootstrap")
        mark_initialized(state_dir)
        return wait_for_child(process)
    except BaseException:
        terminate_child(process)
        raise


def parse_args(
    argv: Sequence[str] | None = None,
) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-name", required=True)
    parser.add_argument(
        "--bootstrap-role", choices=("primary", "replica"), required=True
    )
    parser.add_argument("--bootstrap-primary-host", required=True)
    parser.add_argument("--allowed-master-host", action="append", required=True)
    args, valkey_args = parser.parse_known_args(argv)
    if valkey_args and valkey_args[0] == "--":
        valkey_args = valkey_args[1:]
    return args, valkey_args


def main(argv: Sequence[str] | None = None) -> None:
    args, valkey_args = parse_args(argv)
    state_dir = pathlib.Path(os.environ["STATE_DIR"])
    initialized = (state_dir / "bootstrap-complete").is_file()
    master = wait_for_master(
        args.master_name,
        allowed_hosts=frozenset(args.allowed_master_host),
        initialized=initialized,
    )
    roles = role_arguments(
        master,
        bootstrap_role=args.bootstrap_role,
        bootstrap_primary_host=args.bootstrap_primary_host,
        pod_ip=os.environ["POD_IP"],
        pod_dns=os.environ["POD_DNS"],
    )
    command = [
        "/usr/local/bin/valkey-server",
        "--bind",
        "0.0.0.0",
        "--protected-mode",
        "no",
        "--port",
        "6379",
        "--dir",
        "/data",
        "--replica-announce-ip",
        os.environ["POD_DNS"],
        "--replica-announce-port",
        "6379",
        *valkey_args,
        *roles,
    ]
    if initialized:
        os.execv(command[0], command)
    returncode = start_initial_server(
        command,
        state_dir=state_dir,
        master_name=args.master_name,
        allowed_hosts=frozenset(args.allowed_master_host),
    )
    if returncode:
        raise SystemExit(returncode)


if __name__ == "__main__":
    main()
