# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference check runner for worker validation gates.

Each ``POST /checks/<name>`` runs the command configured for ``<name>`` and
answers 200 if it exits 0, and otherwise 500 with the end of its stderr. Point
``DYN_PUBLISH_GATE`` or ``DYN_READY_GATE`` at these URLs. Run it as a sidecar
with its own image, or as a helper process in the worker container.

The worker's request body is passed to the command as ``OPENAI_BASE_URL`` (the
worker's loopback OpenAI endpoint), ``DYN_VALIDATION_MODEL`` and
``DYN_VALIDATION_GATE`` (``publish`` or ``ready``). The runner keeps no
state: every request runs the check again, and the worker owns every verdict.

    python3 -m dynamo.validation.runner --port 9100 --timeout 600 \\
        --check smoke='python3 /checks/smoke.py' \\
        --check completion='python3 -m dynamo.validation.completion'
"""

import argparse
import json
import os
import shlex
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATE_ENV_KEYS = ("OPENAI_BASE_URL", "DYN_VALIDATION_MODEL", "DYN_VALIDATION_GATE")


def check_env(body: dict) -> dict[str, str]:
    """The command's environment: the runner's own, with the gate variables
    taken only from the worker's request, so a stale ambient value is never
    checked instead."""
    env = {key: value for key, value in os.environ.items() if key not in GATE_ENV_KEYS}
    models = body.get("models") or []
    for key, value in (
        ("OPENAI_BASE_URL", body.get("base_url")),
        ("DYN_VALIDATION_MODEL", models[0] if models else None),
        ("DYN_VALIDATION_GATE", body.get("gate")),
    ):
        if value:
            env[key] = str(value)
    return env


def make_server(
    port: int, checks: dict[str, list[str]], timeout: float
) -> ThreadingHTTPServer:
    """Serve ``checks`` on loopback only: the runner runs commands."""

    class Handler(BaseHTTPRequestHandler):
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
            try:
                result = subprocess.run(
                    argv,
                    env=check_env(body),
                    stdin=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                return self.reply(500, f"no verdict within {timeout:g}s")
            except OSError as error:
                return self.reply(500, f"could not run {argv[0]!r}: {error}")
            if result.returncode == 0:
                return self.reply(200, "pass")
            tail = result.stderr.decode(errors="replace")[-1000:].strip()
            self.reply(500, f"exit status {result.returncode}: {tail}")

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
    args = parser.parse_args()
    checks = {}
    for spec in args.check:
        name, _, command = spec.partition("=")
        if not name or not command.strip():
            parser.error(f"--check must be NAME=COMMAND, got {spec!r}")
        checks[name] = shlex.split(command)
    make_server(args.port, checks, args.timeout).serve_forever()


if __name__ == "__main__":
    main()
