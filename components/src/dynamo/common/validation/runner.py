# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference check runner for worker validation gates.

Each ``POST /checks/<name>`` runs the command configured for ``<name>`` and
answers 200 if it exits 0, and otherwise 500 with the end of its stderr. Point
``DYN_PUBLISH_GATE`` at these URLs. Run it as a sidecar with its own image, or
as a helper process in the worker container.

The request's ``base_url``, the worker's loopback OpenAI endpoint, is passed
to the command as ``OPENAI_BASE_URL``. Everything else a check needs comes
from the runner's own environment and arguments. The runner keeps no state:
every request runs the check again, and the worker owns every verdict.

    python3 -m dynamo.common.validation.runner --port 9100 --timeout 600 \\
        --check smoke='python3 /checks/smoke.py' \\
        --check completion='python3 -m dynamo.common.validation.completion'

With ``--ready-checks``, the runner also runs those checks itself against
``--base-url`` (the worker's ``DYN_VALIDATION_ENDPOINT_PORT``) and serves
``GET /ready`` for a readiness probe: 200 only once every listed check has
passed against the worker answering now. The worker's endpoint names its
instance in the ``x-dynamo-instance-id`` header, so a restarted worker is
checked again rather than inheriting a pass. A check that fails holds that
worker instance; restart the pod to retry.
"""

import argparse
import json
import os
import shlex
import subprocess
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional

INSTANCE_ID_HEADER = "x-dynamo-instance-id"


def check_env(body: dict) -> dict[str, str]:
    """The command's environment: the runner's own, with ``OPENAI_BASE_URL``
    taken only from the worker's request, so a stale ambient value is never
    checked instead."""
    env = {key: value for key, value in os.environ.items() if key != "OPENAI_BASE_URL"}
    if body.get("base_url"):
        env["OPENAI_BASE_URL"] = str(body["base_url"])
    return env


def run_check(argv: list[str], body: dict, timeout: float) -> tuple[bool, str]:
    """Run one check once. Returns whether it passed, and why not."""
    try:
        result = subprocess.run(
            argv,
            env=check_env(body),
            stdin=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return False, f"no verdict within {timeout:g}s"
    except OSError as error:
        return False, f"could not run {argv[0]!r}: {error}"
    if result.returncode == 0:
        return True, "pass"
    tail = result.stderr.decode(errors="replace")[-1000:].strip()
    return False, f"exit status {result.returncode}: {tail}"


def instance_id(base_url: str) -> Optional[str]:
    """The id of the worker answering at ``base_url`` now, or None if none is."""
    try:
        with urllib.request.urlopen(f"{base_url}/models", timeout=5) as response:
            return response.headers.get(INSTANCE_ID_HEADER)
    except (urllib.error.URLError, OSError):
        return None


class Readiness:
    """Runs ``names`` against each worker instance once, and remembers which
    instance passed. A pass counts only for the instance that earned it."""

    def __init__(
        self,
        base_url: str,
        checks: dict[str, list[str]],
        names: list[str],
        timeout: float,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.checks = checks
        self.names = names
        self.timeout = timeout
        self.lock = threading.Lock()
        self.passed: Optional[str] = None
        self.checked: Optional[str] = None
        self.reason = "waiting for the worker's validation endpoint"

    def poll(self) -> None:
        """Check the instance answering now, unless it was already checked."""
        current = instance_id(self.base_url)
        with self.lock:
            if current is None or current == self.checked:
                return
            self.checked = current
            self.reason = f"checking instance {current}"
        body = {"base_url": self.base_url}
        with ThreadPoolExecutor(max_workers=len(self.names)) as pool:
            verdicts = list(
                pool.map(
                    lambda name: (
                        name,
                        *run_check(self.checks[name], body, self.timeout),
                    ),
                    self.names,
                )
            )
        failed = [f"{name}: {why}" for name, ok, why in verdicts if not ok]
        with self.lock:
            if self.checked != current:
                return
            if failed:
                self.reason = f"instance {current} failed " + "; ".join(failed)
            else:
                self.passed = current
                self.reason = "pass"

    def ready(self) -> tuple[bool, str]:
        """Whether the worker answering now is the one that passed."""
        current = instance_id(self.base_url)
        with self.lock:
            if current is not None and current == self.passed:
                return True, "pass"
            if current is None:
                return False, "the worker's validation endpoint is not answering"
            if current != self.checked:
                return False, f"instance {current} not checked yet"
            return False, self.reason

    def run_forever(self, interval: float) -> None:
        while True:
            try:
                self.poll()
            except Exception as error:  # noqa: BLE001 - keep polling
                with self.lock:
                    self.reason = f"poll failed: {error}"
            time.sleep(interval)


def make_server(
    port: int,
    checks: dict[str, list[str]],
    timeout: float,
    readiness: Optional[Readiness] = None,
) -> ThreadingHTTPServer:
    """Serve ``checks`` on loopback only: the runner runs commands."""

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path != "/ready" or readiness is None:
                return self.reply(404, "not found")
            ok, why = readiness.ready()
            self.reply(200 if ok else 503, why)

        def do_POST(self) -> None:
            name = self.path.removeprefix("/checks/")
            argv = checks.get(name)
            if argv is None:
                return self.reply(404, f"no check named {name!r}")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = {}
            if not isinstance(body, dict):
                body = {}
            ok, why = run_check(argv, body, timeout)
            self.reply(200 if ok else 500, why)

        def reply(self, status: int, text: str) -> None:
            data = text.encode()
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=9100)
    parser.add_argument(
        "--timeout",
        type=float,
        default=600,
        help="seconds a check may run before it fails (default 600)",
    )
    parser.add_argument(
        "--check", action="append", required=True, help="NAME=COMMAND; repeatable"
    )
    parser.add_argument(
        "--ready-checks",
        help="comma-separated check names to run against --base-url for GET /ready",
    )
    parser.add_argument(
        "--base-url",
        help="the worker's validation endpoint, e.g. http://127.0.0.1:8790/v1",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5,
        help="seconds between checks for a new worker instance (default 5)",
    )
    args = parser.parse_args()
    checks = {}
    for spec in args.check:
        name, _, command = spec.partition("=")
        if not name or not command.strip():
            parser.error(f"--check must be NAME=COMMAND, got {spec!r}")
        checks[name] = shlex.split(command)
    readiness = None
    if args.ready_checks:
        names = [name.strip() for name in args.ready_checks.split(",") if name.strip()]
        unknown = [name for name in names if name not in checks]
        if not names or unknown:
            parser.error(f"--ready-checks names unknown checks: {unknown or names}")
        if not args.base_url:
            parser.error("--ready-checks needs --base-url")
        readiness = Readiness(args.base_url, checks, names, args.timeout)
        threading.Thread(
            target=readiness.run_forever, args=(args.poll_interval,), daemon=True
        ).start()
    make_server(args.port, checks, args.timeout, readiness).serve_forever()


if __name__ == "__main__":
    main()
