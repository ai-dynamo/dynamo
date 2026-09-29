# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only, single-use trigger and publication gate for resident GMS V1 ranks.

Each generation uses a fresh shared directory. Readiness requires all server
sockets and transfer lanes to be initialized without allocating or reading any
weight payload. POST /load begins transfers; only exact allocation verification
allows the restored engine to pass /gms/all-ready.
"""

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from resolve_plan import resolve


def emit(event, **fields):
    print(json.dumps({"event": event, "epoch": time.time(), **fields}), flush=True)


def atomic_write(path, text, exclusive=False):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(text)
        if exclusive:
            os.link(temporary, path)
        else:
            os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class LoadConflict(ValueError):
    pass


class Coordinator:
    def __init__(self, plan, generation, workers, chunk_mib, root, app):
        self.plan = plan
        self.capture_id = plan["capture_id"]
        self.generation = generation
        self.workers = workers
        self.chunk_mib = chunk_mib
        self.root = Path(root)
        self.app = Path(app)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = "waiting"
        self.error = None
        self.ranks = {rank["rank"]: rank for rank in plan["ranks"]}
        if set(self.ranks) != set(range(8)):
            raise ValueError("resident experiment requires exactly ranks 0 through 7")
        self.root.mkdir(parents=True, exist_ok=True)
        stale = [
            name
            for name in [
                "load-trigger.json",
                "all-ready",
                "coordinator-error.json",
                *(f"rank-{rank}.json" for rank in self.ranks),
                *(f"published-{rank}" for rank in self.ranks),
            ]
            if (self.root / name).exists()
        ]
        if stale:
            raise ValueError(f"resident generation directory has stale state: {stale}")
        atomic_write(self.root / "restore-plan.json", json.dumps(plan))

    def _publications(self):
        return sum((self.root / f"published-{rank}").exists() for rank in self.ranks)

    def _online(self):
        records = []
        for rank, expected in self.ranks.items():
            path = self.root / f"online-{rank}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            required = {
                "rank": rank,
                "capture_id": self.capture_id,
                "generation": self.generation,
                "uuid": expected["destination_uuid"],
                "workers": self.workers,
                "chunk_mib": self.chunk_mib,
                "loader_lanes_ready": self.workers,
                "pinned_bytes": self.workers * 2 * self.chunk_mib * 1024**2,
                "payload_bytes_read": 0,
                "weight_allocations": 0,
            }
            for key, value in required.items():
                if record.get(key) != value:
                    raise ValueError(f"rank {rank} online field {key} mismatch")
            if sorted(record.get("server_domains_responsive", [])) != [
                "kv_cache",
                "weights",
            ]:
                raise ValueError(f"rank {rank} server domains are not responsive")
            online = record.get("online_epoch", 0)
            daemon = record.get("daemon_started_epoch", 0)
            if not (
                math.isfinite(online) and math.isfinite(daemon) and 0 < daemon <= online
            ):
                raise ValueError(f"rank {rank} has invalid initialization timestamps")
            records.append(record)
        return records

    def _ready(self):
        publications = self._publications()
        records = self._online() if self.state == "waiting" else []
        stale = any(
            (self.root / name).exists()
            for name in [
                "load-trigger.json",
                "all-ready",
                *(f"rank-{rank}.json" for rank in self.ranks),
            ]
        )
        return {
            "capture_id": self.capture_id,
            "generation": self.generation,
            "ready": self.state == "waiting"
            and len(records) == 8
            and publications == 0
            and not stale,
            "publications": publications,
            "state": self.state,
            "online": records,
            "error": self.error,
        }

    def ready(self):
        with self.lock:
            try:
                return self._ready()
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._fail(error)
                return self._ready()

    def health(self):
        with self.lock:
            return {
                "healthy": self.state != "error",
                "state": self.state,
                "error": self.error,
            }

    def load(self, request):
        if not isinstance(request, dict) or set(request) != {
            "capture_id",
            "generation",
        }:
            raise ValueError("load request must contain only capture_id and generation")
        with self.lock:
            if request != {
                "capture_id": self.capture_id,
                "generation": self.generation,
            }:
                raise LoadConflict("load request capture or generation mismatch")
            if not self._ready()["ready"]:
                raise LoadConflict("generation is not pristine and ready for a load")
            trigger = {**request, "trigger_written_epoch": time.time()}
            try:
                atomic_write(
                    self.root / "load-trigger.json", json.dumps(trigger), exclusive=True
                )
            except FileExistsError as error:
                raise LoadConflict("load trigger already exists") from error
            self.state = "loading"
            emit("load_trigger_written", **trigger)
            return trigger

    def _fail(self, error):
        self.error = str(error)
        self.state = "error"
        record = {
            "error": self.error,
            "capture_id": self.capture_id,
            "generation": self.generation,
            "epoch": time.time(),
        }
        atomic_write(self.root / "coordinator-error.json", json.dumps(record))
        emit("coordinator_error", **record)

    def watch_publication(self):
        while not self.stop.wait(0.005):
            with self.lock:
                if self.state != "loading":
                    continue
            try:
                if self._publications() != 8:
                    continue
                for rank, expected in self.ranks.items():
                    record = json.loads((self.root / f"rank-{rank}.json").read_text())
                    if any(
                        record.get(key) != value
                        for key, value in {
                            "rank": rank,
                            "capture_id": self.capture_id,
                            "generation": self.generation,
                            "uuid": expected["destination_uuid"],
                            "workers": self.workers,
                            "chunk_mib": self.chunk_mib,
                        }.items()
                    ):
                        raise ValueError(f"rank {rank} published identity mismatch")
                    if (
                        self.root / f"published-{rank}"
                    ).read_text().strip() != expected["destination_uuid"]:
                        raise ValueError(f"rank {rank} published marker UUID mismatch")
                emit("publication_verification_started")
                verified = subprocess.run(
                    [
                        sys.executable,
                        str(self.app / "verify_publication.py"),
                        str(self.root / "restore-plan.json"),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                (self.root / "publication-verification.txt").write_text(verified.stdout)
                published = time.time()
                atomic_write(self.root / "all-ready", str(published), exclusive=True)
                with self.lock:
                    self.state = "published"
                emit(
                    "all_ranks_verified",
                    published_epoch=published,
                    capture_id=self.capture_id,
                    generation=self.generation,
                )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                subprocess.SubprocessError,
            ) as error:
                if isinstance(error, subprocess.CalledProcessError):
                    error = RuntimeError(f"publication verifier failed: {error.stderr}")
                with self.lock:
                    self._fail(error)


def make_server(coordinator, address):
    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, payload):
            encoded = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def do_GET(self):
            if self.path == "/ready":
                status = coordinator.ready()
                self.respond(200 if status["ready"] else 503, status)
            elif self.path == "/healthz":
                status = coordinator.health()
                self.respond(200 if status["healthy"] else 503, status)
            else:
                self.respond(404, {"error": "unknown endpoint"})

        def do_POST(self):
            if self.path != "/load":
                self.respond(404, {"error": "unknown endpoint"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 4096:
                    raise ValueError("invalid load body length")
                payload = json.loads(self.rfile.read(length))
                result = coordinator.load(payload)
            except LoadConflict as error:
                self.respond(409, {"error": str(error)})
            except (ValueError, TypeError) as error:
                self.respond(400, {"error": str(error)})
            except (OSError, RuntimeError) as error:
                with coordinator.lock:
                    coordinator._fail(error)
                self.respond(500, {"error": str(error)})
            else:
                self.respond(200, result)

        def log_message(self, message, *args):
            emit("http", message=message % args)

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-id", required=True)
    parser.add_argument("--generation", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=128)
    parser.add_argument("--root", type=Path, default=Path("/gms"))
    parser.add_argument("--app", type=Path, default=Path("/snapshot-app"))
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    if args.workers < 1 or args.chunk_mib < 1:
        parser.error("workers and chunk-mib must be positive")
    plan = resolve(
        *(
            json.loads((args.app / name).read_text())
            for name in ["capture.json", "claim.json", "slices.json"]
        )
    )
    if plan["capture_id"] != args.capture_id:
        raise ValueError("coordinator capture ID differs from the resolved artifact")
    if plan["cuda_device_map"] != os.environ.get("SNAPSHOT_CUDA_DEVICE_MAP"):
        raise ValueError("coordinator CUDA map differs from the resolved rank mapping")
    coordinator = Coordinator(
        plan, args.generation, args.workers, args.chunk_mib, args.root, args.app
    )
    server = make_server(coordinator, ("0.0.0.0", args.port))
    watcher = threading.Thread(target=coordinator.watch_publication, daemon=True)
    watcher.start()

    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    emit(
        "coordinator_listening",
        port=args.port,
        capture_id=args.capture_id,
        generation=args.generation,
    )
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        coordinator.stop.set()
        server.server_close()
        watcher.join(timeout=65)


if __name__ == "__main__":
    main()
