#!/usr/bin/env python3
"""Run one native harness scenario through a capture proxy and direct or SSH endpoint."""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def _free_loopback_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def _wait_for_tunnel(port: int, process: subprocess.Popen[bytes], timeout_s: float = 15) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"SSH tunnel exited before becoming ready (exit={process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError("SSH tunnel did not bind its local port")


def _start_proxy(
    script: Path,
    listen_port: int,
    upstream_url: str,
    artifacts: Path,
    inject_status: int | None,
    inject_at_request: int,
    truncate_sse_after_events: int | None,
    truncate_sse_at_request: int,
) -> subprocess.Popen[bytes]:
    command = [
        sys.executable,
        str(script),
        "--listen",
        f"127.0.0.1:{listen_port}",
        "--upstream",
        upstream_url.rstrip("/"),
        "--record",
        str(artifacts / "wire.jsonl"),
    ]
    if inject_status is not None:
        command.extend(["--inject-status", str(inject_status), "--inject-at-request", str(inject_at_request)])
    if truncate_sse_after_events is not None:
        command.extend(
            [
                "--truncate-sse-after-events",
                str(truncate_sse_after_events),
                "--truncate-sse-at-request",
                str(truncate_sse_at_request),
            ]
        )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None
    ready, _, _ = select.select([process.stdout], [], [], 10)
    if not ready:
        raise TimeoutError("capture proxy did not announce readiness")
    announcement = process.stdout.readline().decode(errors="replace")
    if not announcement.startswith("capture proxy:"):
        raise RuntimeError(f"capture proxy failed to start: {announcement.strip()}")
    return process


def _stop_process_group(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _copy_remote_artifacts(args: argparse.Namespace, artifacts: Path) -> None:
    destination = artifacts / "remote"
    destination.mkdir()
    command = ["scp", "-P", str(args.ssh_port), "-i", str(args.identity_file), "-o", "IdentitiesOnly=yes"]
    for name in ("endpoint.json", "frontend.log", "worker.log", "request-trace.jsonl", "dynamo.diff", "sglang.diff"):
        command.append(f"{args.remote_user}@{args.remote_host}:{args.remote_run_root}/{name}")
    command.append(str(destination))
    completed = subprocess.run(command, capture_output=True, text=True)
    (artifacts / "copy_remote.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise RuntimeError("failed to collect remote artifacts")


def _write_transport_summary(artifacts: Path) -> None:
    """Persist only transport outcomes, so injected-error runs have one oracle."""
    rows: list[dict[str, object]] = []
    wire = artifacts / "wire.jsonl"
    if wire.exists():
        for line in wire.read_text(encoding="utf-8").splitlines():
            with contextlib.suppress(json.JSONDecodeError):
                value = json.loads(line)
                if isinstance(value, dict):
                    rows.append(value)
    statuses: dict[str, int] = {}
    for row in rows:
        if row.get("kind") != "response":
            continue
        status = str(row.get("status"))
        statuses[status] = statuses.get(status, 0) + 1
    summary = {
        "request_count": sum(row.get("kind") == "request" for row in rows),
        "response_statuses": statuses,
        "injected_faults": [
            {
                key: row[key]
                for key in ("fault", "request_number", "status", "after_sse_events")
                if key in row
            }
            for row in rows
            if row.get("kind") == "fault_injected"
        ],
        "terminal_sse_events": sorted(
            {
                row.get("event")
                for row in rows
                if row.get("kind") == "sse_event" and row.get("event") in {"response.completed", "message_stop"}
            }
        ),
    }
    (artifacts / "transport.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _validate_model(upstream_url: str, model: str, artifacts: Path) -> None:
    """Fail before a native run if the live endpoint cannot serve the requested model."""
    with urlopen(f"{upstream_url.rstrip('/')}/v1/models", timeout=10) as response:
        payload = json.loads(response.read())
    data = payload.get("data") if isinstance(payload, dict) else None
    available = sorted(
        item["id"] for item in data if isinstance(item, dict) and isinstance(item.get("id"), str)
    ) if isinstance(data, list) else []
    preflight = {"requested_model": model, "available_models": available, "matched": model in available}
    (artifacts / "endpoint-preflight.json").write_text(json.dumps(preflight, indent=2) + "\n", encoding="utf-8")
    if not preflight["matched"]:
        raise RuntimeError(f"requested model {model!r} is absent from live endpoint models: {available}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", choices=("codex", "claude"), required=True)
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--artifacts", type=Path, help="Output directory; defaults under /tmp/dynamo-harness-compat")
    parser.add_argument("--remote-host", default="72.25.69.152")
    parser.add_argument("--remote-user", default="nvidia")
    parser.add_argument("--ssh-port", type=int, default=2222)
    parser.add_argument("--identity-file", type=Path, default=Path.home() / ".ssh" / "id_ed2551")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--endpoint-url", help="Direct Dynamo base URL, for example http://127.0.0.1:8000")
    target.add_argument("--remote-http-port", type=int, help="Dynamo loopback port on the remote host")
    parser.add_argument("--remote-run-root", help="Remote run directory to copy logs from; requires --remote-http-port")
    parser.add_argument("--codex", default="codex")
    parser.add_argument("--claude", default="claude")
    parser.add_argument("--interactive", action="store_true", help="Use the real Claude Code terminal UI driver.")
    parser.add_argument(
        "--result-timeout-s",
        type=float,
        help="Native harness result budget; Claude runs otherwise default to 900 seconds.",
    )
    parser.add_argument(
        "--turn-timeout-s",
        type=float,
        help="Bound one native Codex turn; use a shorter value for budgeted nightly sentinels.",
    )
    parser.add_argument(
        "--interactive-timeout-s",
        type=float,
        help="Bound each real Claude terminal interaction when --interactive is selected.",
    )
    parser.add_argument("--inject-status", type=int, choices=(400, 401, 403, 404, 409, 429, 500, 502, 503, 529))
    parser.add_argument("--inject-at-request", type=int, default=1)
    parser.add_argument("--truncate-sse-after-events", type=int)
    parser.add_argument("--truncate-sse-at-request", type=int, default=1)
    args = parser.parse_args()
    if args.remote_run_root and args.remote_http_port is None:
        parser.error("--remote-run-root requires --remote-http-port")

    if args.artifacts is None:
        run_name = f"{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{args.harness}-{args.scenario}"
        args.artifacts = Path("/tmp/dynamo-harness-compat") / run_name
    artifacts = args.artifacts.resolve()
    artifacts.mkdir(parents=True, exist_ok=False)
    (artifacts / "fault.json").write_text(
        json.dumps(
            {
                "inject_status": args.inject_status,
                "inject_at_request": args.inject_at_request if args.inject_status is not None else None,
                "truncate_sse_after_events": args.truncate_sse_after_events,
                "truncate_sse_at_request": (
                    args.truncate_sse_at_request if args.truncate_sse_after_events is not None else None
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    script_dir = Path(__file__).resolve().parent
    tunnel_port = _free_loopback_port()
    proxy_port = _free_loopback_port()
    tunnel: subprocess.Popen[bytes] | None = None
    proxy: subprocess.Popen[bytes] | None = None
    driver_process: subprocess.Popen[str] | None = None
    try:
        if args.endpoint_url is not None:
            upstream_url = args.endpoint_url
        else:
            tunnel = subprocess.Popen(
                [
                    "ssh",
                    "-N",
                    "-p",
                    str(args.ssh_port),
                    "-i",
                    str(args.identity_file),
                    "-o",
                    "IdentitiesOnly=yes",
                    "-o",
                    "ControlMaster=no",
                    "-o",
                    "ControlPath=none",
                    "-o",
                    "ExitOnForwardFailure=yes",
                    "-o",
                    "ServerAliveInterval=30",
                    "-L",
                    f"{tunnel_port}:127.0.0.1:{args.remote_http_port}",
                    f"{args.remote_user}@{args.remote_host}",
                ],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=(artifacts / "ssh-tunnel.stderr.log").open("wb"),
                start_new_session=True,
            )
            _wait_for_tunnel(tunnel_port, tunnel)
            upstream_url = f"http://127.0.0.1:{tunnel_port}"
        _validate_model(upstream_url, args.model, artifacts)
        proxy = _start_proxy(
            script_dir / "capture_proxy.py",
            proxy_port,
            upstream_url,
            artifacts,
            args.inject_status,
            args.inject_at_request,
            args.truncate_sse_after_events,
            args.truncate_sse_at_request,
        )
        driver = script_dir / ("claude_interactive_driver.py" if args.interactive else f"{args.harness}_driver.py")
        if args.interactive and args.harness != "claude":
            parser.error("--interactive is supported only with --harness claude")
        command = [
            sys.executable,
            str(driver),
            "--proxy-url",
            f"http://127.0.0.1:{proxy_port}",
            "--model",
            args.model,
            "--artifacts",
            str(artifacts),
            "--scenario",
            args.scenario,
            f"--{args.harness}",
            getattr(args, args.harness),
        ]
        if args.result_timeout_s is not None and args.harness == "claude":
            command.extend(["--result-timeout-s", str(args.result_timeout_s)])
        if args.turn_timeout_s is not None and args.harness == "codex":
            command.extend(["--turn-timeout-s", str(args.turn_timeout_s)])
        if args.interactive_timeout_s is not None and args.interactive:
            command.extend(["--request-timeout-s", str(args.interactive_timeout_s)])
        driver_process = subprocess.Popen(
            command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True
        )
        while driver_process.poll() is None:
            if tunnel is not None and tunnel.poll() is not None:
                _stop_process_group(driver_process)
                stdout, stderr = driver_process.communicate()
                (artifacts / "driver.stdout.log").write_text(stdout, encoding="utf-8")
                (artifacts / "driver.stderr.log").write_text(stderr, encoding="utf-8")
                _write_transport_summary(artifacts)
                if args.remote_run_root is not None:
                    _copy_remote_artifacts(args, artifacts)
                raise RuntimeError(f"SSH tunnel exited during native harness run (exit={tunnel.returncode})")
            time.sleep(0.2)
        stdout, stderr = driver_process.communicate()
        (artifacts / "driver.stdout.log").write_text(stdout, encoding="utf-8")
        (artifacts / "driver.stderr.log").write_text(stderr, encoding="utf-8")
        _write_transport_summary(artifacts)
        if args.remote_run_root is not None:
            _copy_remote_artifacts(args, artifacts)
        print(stdout.strip())
        return driver_process.returncode
    finally:
        _stop_process_group(driver_process)
        _stop_process_group(proxy)
        _stop_process_group(tunnel)
        if proxy is not None and proxy.stderr is not None:
            (artifacts / "capture-proxy.stderr.log").write_bytes(proxy.stderr.read())


if __name__ == "__main__":
    raise SystemExit(main())
