# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python owns DeepInfer; a Rust extension owns OpenEngine gRPC."""

import argparse
import json
import logging
import signal
import threading
from pathlib import Path

import _deepinfer_openengine
from engine import RuntimeHandle

logger = logging.getLogger(__name__)


def serve(args):
    args.model_dir = args.model_dir.resolve(strict=True)
    args.engine_options = (
        json.loads(args.engine_args.read_text()) if args.engine_args else {}
    )
    stopped = threading.Event()
    previous = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, lambda *_: stopped.set())
    runtime = None
    server = None
    try:
        runtime = RuntimeHandle(args)
        server = _deepinfer_openengine.start_server(
            runtime,
            str(args.model_dir),
            args.listen,
            args.context_length,
            args.max_requests,
        )
        logger.info(
            "Rust OpenEngine server ready on %s; Python host PID is the engine PID",
            args.listen,
        )
        while not stopped.wait(0.1):
            if runtime.error or not server.is_alive():
                raise RuntimeError(runtime.error or "Rust OpenEngine server stopped")
    finally:
        try:
            if server is not None:
                server.shutdown()
        finally:
            if runtime is not None:
                runtime.close()
            for signum, handler in previous.items():
                signal.signal(signum, handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen", default="127.0.0.1:30001")
    parser.add_argument("--engine", choices=("mock", "deepinfer"), default="mock")
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--max-requests", type=int, default=32)
    parser.add_argument("--engine-args", type=Path)
    parser.add_argument("--tick-ms", type=int, default=2)
    args = parser.parse_args()
    if args.context_length < 2 or args.max_requests < 1 or args.tick_ms < 0:
        parser.error(
            "context-length >= 2, max-requests >= 1 and tick-ms >= 0 are required"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    serve(args)
