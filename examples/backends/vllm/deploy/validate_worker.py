# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readiness sidecar for a vLLM worker.

Every 5 seconds, ask the worker (port 8790) to generate one token. Serve HTTP
on port 9100: 200 if the last check passed and the worker has not restarted
since, 503 otherwise. The kubelet's readinessProbe polls it.
"""

import json
import logging
from pathlib import Path
from threading import Thread
from time import sleep
from urllib.request import ProxyHandler, Request, build_opener
from wsgiref.simple_server import make_server

HTTP = build_opener(ProxyHandler({}))


def _request(path, timeout, data=None):
    body = json.dumps(data).encode() if data is not None else None
    request = Request(
        f"http://127.0.0.1:8790/v1/{path}", body, {"Content-Type": "application/json"}
    )
    with HTTP.open(request, timeout=timeout) as reply:
        return json.load(reply)


class Validator:
    def __init__(self, startup_id_file):
        self.startup_id_file = Path(startup_id_file)
        self.validated_startup_id = None

    def read_startup_id(self):
        # Re-read every time so it follows the worker's atomic rename.
        return self.startup_id_file.read_text(encoding="ascii").strip()

    def check(self):
        """Generate one token. Any error, timeout or malformed reply fails the check."""
        startup_id = self.read_startup_id()
        model = _request("models", 0.5)["data"][0]["id"]
        completion = _request(
            "completions", 30, {"model": model, "prompt": "Hello", "max_tokens": 1}
        )
        # Ignore the text: one token can decode to "" or whitespace on a healthy
        # worker. "stop" (EOS) and "length" both mean generation finished.
        finish_reason = completion["choices"][0].get("finish_reason")
        if finish_reason not in ("stop", "length"):
            raise ValueError(f"generation did not finish: {completion}")
        if self.read_startup_id() != startup_id:
            raise ValueError("worker restarted during check")
        return startup_id

    def run(self):
        while True:
            try:
                startup_id = self.check()
            except Exception as exc:
                startup_id = None
                logging.warning(
                    "Worker failed inference check; /ready returns 503: %s", exc
                )
            else:
                # Log only the first pass per worker launch, not every 5 seconds.
                if startup_id != self.validated_startup_id:
                    logging.info(
                        "Worker passed inference check (startup ID %s); /ready returns 200",
                        startup_id,
                    )
            self.validated_startup_id = startup_id
            sleep(5)

    def ready_probe(self, environ, respond):
        """Readiness probe: 200 if the current worker passed its last check."""
        try:
            ready = self.validated_startup_id == self.read_startup_id()
        except OSError:
            ready = False
        respond("200 OK" if ready else "503 Service Unavailable", [])
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    validator = Validator("/worker-validation/startup-id")
    with make_server("0.0.0.0", 9100, validator.ready_probe) as server:
        Thread(target=validator.run, daemon=True).start()
        server.serve_forever()
