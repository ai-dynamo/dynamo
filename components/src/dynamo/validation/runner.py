# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check runner for ``url`` validation gates, run as a sidecar with its own image.

Each ``POST /validate/<name>`` runs the command configured for ``<name>`` with
the gate's environment from the request body (``OPENAI_BASE_URL``,
``DYN_VALIDATION_MODEL``, ``DYN_VALIDATION_GATE``). It answers 200 if the
command exits 0, and otherwise 500 with the end of its stderr. It keeps no
state: the worker owns every verdict and asks again after a restart.

    python3 -m dynamo.validation.runner --port 9100 \\
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


def make_server(port: int, checks: dict[str, list[str]]) -> ThreadingHTTPServer:
    """Serve ``checks`` on loopback only: the runner runs commands."""

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            name = self.path.removeprefix("/validate/")
            argv = checks.get(name)
            if argv is None:
                return self.reply(404, f"no check named {name!r}")
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                body = {}
            env = {key: str(body[key]) for key in GATE_ENV_KEYS if key in body}
            result = subprocess.run(
                argv,
                env={**os.environ, **env},
                stdin=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            if result.returncode == 0:
                return self.reply(200, "pass")
            tail = result.stderr.decode(errors="replace")[-2048:].strip()
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
        "--check", action="append", required=True, help="NAME=COMMAND; repeatable"
    )
    args = parser.parse_args()
    checks = {}
    for spec in args.check:
        name, _, command = spec.partition("=")
        if not name or not command.strip():
            parser.error(f"--check must be NAME=COMMAND, got {spec!r}")
        checks[name] = shlex.split(command)
    make_server(args.port, checks).serve_forever()


if __name__ == "__main__":
    main()
