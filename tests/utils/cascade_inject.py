# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
cascade_inject — CLI for live fault injection against a long-running
Dynamo deployment driven by test_cascade_console.py.

The driver test (`test_cascade_console.py`) holds the deployment open and
exposes a Unix socket at /tmp/cascade-inject-<dgd>.sock. This CLI sends
JSON commands to that socket and prints the response.

Subcommands all dispatch to existing event classes in tests/fault_tolerance/
deploy/events.py — the CLI doesn't re-implement faults, just packages them.

Examples (run from anywhere with access to the socket file):

    cascade_inject stall VllmDecodeWorker --duration 30s
    cascade_inject unstall VllmDecodeWorker
    cascade_inject partition VllmDecodeWorker VllmPrefillWorker
    cascade_inject memhog vllm-disagg-...-vllmdecodeworker-XYZ --gb 20 --duration 60s
    cascade_inject kill vllm-disagg-...-vllmdecodeworker-XYZ
    cascade_inject load 64
    cascade_inject dump
    cascade_inject status
    cascade_inject quit

Each command also writes a line to /tmp/cascade_events.log (which the TUI
tails) and POSTs a Grafana annotation if GRAFANA_URL + GRAFANA_TOKEN are set.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_SOCKET_FMT = "/tmp/cascade-inject-{dgd}.sock"
DEFAULT_EVENT_LOG = Path("/tmp/cascade_events.log")


def _socket_path(dgd: str | None, override: str | None) -> Path:
    if override:
        return Path(override)
    if not dgd:
        # Look for the only socket matching the pattern
        candidates = list(Path("/tmp").glob("cascade-inject-*.sock"))
        if len(candidates) == 1:
            return candidates[0]
        if not candidates:
            raise SystemExit(
                "no cascade-inject sockets found in /tmp; pass --dgd or --socket"
            )
        raise SystemExit(
            f"multiple cascade-inject sockets ({[c.name for c in candidates]}); "
            "pass --dgd or --socket"
        )
    return Path(DEFAULT_SOCKET_FMT.format(dgd=dgd))


def send_command(sock_path: Path, payload: dict) -> dict:
    """Connect to the driver, send one JSON line, read one JSON line back."""
    if not sock_path.exists():
        raise SystemExit(
            f"socket {sock_path} not found — is test_cascade_console running?"
        )
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.connect(str(sock_path))
    try:
        s.sendall((json.dumps(payload) + "\n").encode())
        # Read until newline
        buf = bytearray()
        while True:
            chunk = s.recv(4096)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\n" in buf:
                break
        line, _, _ = buf.partition(b"\n")
        return json.loads(line.decode()) if line else {}
    finally:
        s.close()


def append_event_log(text: str, log_path: Path = DEFAULT_EVENT_LOG) -> None:
    ts = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())
    line = f"{ts}  {text}"
    try:
        with log_path.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass  # event log is best-effort


def post_grafana_annotation(
    text: str,
    tags: list[str],
    grafana_url: str | None,
    token: str | None,
) -> None:
    """Best-effort POST to /api/annotations. Silent on failure."""
    if not grafana_url or not token:
        return
    try:
        import urllib.request

        body = json.dumps(
            {
                "time": int(time.time() * 1000),
                "tags": tags,
                "text": text,
            }
        ).encode()
        req = urllib.request.Request(
            url=f"{grafana_url.rstrip('/')}/api/annotations",
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
        )
        urllib.request.urlopen(req, timeout=2.0).close()
    except Exception:
        pass  # annotations are best-effort; don't break the inject flow


def _parse_duration(s: str) -> float:
    """Parse '30s', '5m', '500ms', or bare seconds → float seconds."""
    s = s.strip()
    if s.endswith("ms"):
        return float(s[:-2]) / 1000.0
    if s.endswith("s"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) * 60.0
    return float(s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0])
    ap.add_argument("--dgd", help="DynamoGraphDeployment name (used to find socket)")
    ap.add_argument(
        "--socket",
        help="Override socket path (default: /tmp/cascade-inject-<dgd>.sock)",
    )
    ap.add_argument(
        "--event-log",
        type=Path,
        default=DEFAULT_EVENT_LOG,
        help="Append-only event log file (read by cascade_tui)",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)

    # stall
    p_stall = sub.add_parser("stall", help="SIGSTOP a worker (rank or whole pod)")
    p_stall.add_argument(
        "target", help="Service name (e.g. VllmDecodeWorker) or pod name"
    )
    p_stall.add_argument(
        "--duration",
        default="30s",
        help="Auto-unstall after this duration. Default 30s.",
    )
    p_stall.add_argument(
        "--process-name",
        default="dynamo.vllm",
        help="Substring of the process command to match (default 'dynamo.vllm' = the vllm engine).",
    )

    # unstall
    p_unstall = sub.add_parser("unstall", help="SIGCONT a previously stalled worker")
    p_unstall.add_argument("target", help="Service or pod name")

    # partition
    p_part = sub.add_parser("partition", help="NetworkPolicy drop between two services")
    p_part.add_argument("src", help="Source pod or service name")
    p_part.add_argument("dst", help="Destination pod or service name")
    p_part.add_argument("--duration", default="30s")

    # unpartition
    sub.add_parser("unpartition", help="Remove the partition NetworkPolicy")

    # memhog
    p_mem = sub.add_parser("memhog", help="GPU memory pressure on a pod's node")
    p_mem.add_argument("target_pod")
    p_mem.add_argument("--gb", type=float, default=20.0)
    p_mem.add_argument("--duration", default="60s")

    # kill
    p_kill = sub.add_parser("kill", help="Force-delete a worker pod")
    p_kill.add_argument("target_pod")

    # load
    p_load = sub.add_parser("load", help="Live concurrency change")
    p_load.add_argument("concurrency", type=int)

    # dump
    p_dump = sub.add_parser(
        "dump", help="Snapshot pod manifests + recent logs to log_dir/snapshots/<ts>/"
    )
    p_dump.add_argument(
        "--service",
        action="append",
        help="Filter to specific service(s); repeat for multiple",
    )

    # status
    sub.add_parser("status", help="Show DGD pod state + last metrics tick")

    # quit
    sub.add_parser("quit", help="Stop the driver and tear down the deployment")

    args = ap.parse_args()
    sock_path = _socket_path(args.dgd, args.socket)

    # Build the JSON payload
    cmd = args.cmd
    payload: dict[str, Any] = {"cmd": cmd}
    if cmd == "stall":
        payload.update(
            target=args.target,
            duration_s=_parse_duration(args.duration),
            process_name=args.process_name,
        )
        log_text = f"stall {args.target}/{args.process_name} for {args.duration}"
    elif cmd == "unstall":
        payload.update(target=args.target)
        log_text = f"unstall {args.target}"
    elif cmd == "partition":
        payload.update(
            src=args.src, dst=args.dst, duration_s=_parse_duration(args.duration)
        )
        log_text = f"partition {args.src} → {args.dst} for {args.duration}"
    elif cmd == "unpartition":
        log_text = "unpartition"
    elif cmd == "memhog":
        payload.update(
            target_pod=args.target_pod,
            gb=args.gb,
            duration_s=_parse_duration(args.duration),
        )
        log_text = f"memhog {args.target_pod} {args.gb}GB for {args.duration}"
    elif cmd == "kill":
        payload.update(target_pod=args.target_pod)
        log_text = f"kill {args.target_pod}"
    elif cmd == "load":
        payload.update(concurrency=args.concurrency)
        log_text = f"load → concurrency={args.concurrency}"
    elif cmd == "dump":
        payload.update(services=args.service or [])
        log_text = f"dump {args.service or 'all'}"
    elif cmd == "status":
        log_text = "status"
    elif cmd == "quit":
        log_text = "quit"
    else:
        log_text = cmd

    # Send + print response
    response = send_command(sock_path, payload)
    print(json.dumps(response, indent=2))

    # Side effects: event log + Grafana annotation
    if cmd not in ("status",):  # don't pollute the log with status checks
        append_event_log(log_text, args.event_log)
        post_grafana_annotation(
            text=log_text,
            tags=["cascade", cmd],
            grafana_url=os.environ.get("GRAFANA_URL"),
            token=os.environ.get("GRAFANA_TOKEN"),
        )

    return 0 if response.get("ok", True) else 1


if __name__ == "__main__":
    sys.exit(main())
