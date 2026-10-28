# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A stand-in frontend: enough OpenAI surface to exercise the harness.

Run as a subprocess by the end-to-end test. It is not a mock inside the test
process — the point is that the provider really starts a process, really speaks
HTTP to it, really restarts it, and really collects its log.

    python stub_frontend.py --port 8123 --model Qwen/Qwen3-0.6B [--fail-after N]
"""

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--model", default="stub/model")
parser.add_argument("--max-model-len", type=int, default=2048)
# Lets a test drive a role into failure without killing it, which is a different
# fault from a crash and is worth being able to express.
parser.add_argument("--fail-after", type=int, default=0)
args = parser.parse_args()

STATE = {"served": 0}


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, payload):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/v1/models":
            self._send(200, {"data": [{"id": args.model, "object": "model"}]})
        elif self.path == "/health":
            self._send(200, {"status": "ok", "served": STATE["served"]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length) or b"{}")
        STATE["served"] += 1
        if args.fail_after and STATE["served"] > args.fail_after:
            print(f"ERROR: refusing request {STATE['served']}", flush=True)
            self._send(503, {"error": "overloaded"})
            return
        self._send(
            200,
            {
                "id": f"cmpl-{STATE['served']}",
                "model": args.model,
                "max_model_len": args.max_model_len,
                "choices": [{"text": f"echo: {body.get('prompt', '')}"}],
            },
        )

    def log_message(self, fmt, *a):
        # One line per request, on stdout, so the collected log is assertable.
        print("request " + (fmt % a), flush=True)


print(
    f"serving {args.model} on {args.port} max_model_len={args.max_model_len}",
    flush=True,
)
server = HTTPServer(("127.0.0.1", args.port), Handler)
try:
    server.serve_forever()
except KeyboardInterrupt:
    sys.exit(0)
