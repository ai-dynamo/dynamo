# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generic GMS service coordinator; select snapshot metadata after DGD creation.

Startup knows only the resident allocation and service incarnation. One POST
binds this qualification coordinator to a DGD and discovers its PVC manifest.
No payload is read here. Reset/reuse of the coordinator is intentionally unsupported.
"""

import argparse
import array
import hashlib
import json
import math
import os
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pagebroker_pb2 as pb
from google.protobuf.message import DecodeError
from resident_coordinator import LoadConflict, atomic_write, emit, make_server
from resolve_plan import gpu_uuid, resolve


def rpc(path, request, timeout=240):
    """Public PageBroker framing, with no ancillary descriptors permitted."""
    deadline = time.monotonic() + timeout

    def receive(stream, size):
        result = bytearray()
        while len(result) < size:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("PageBroker RPC deadline exceeded")
            stream.settimeout(remaining)
            data, controls, flags, _ = stream.recvmsg(
                size - len(result), socket.CMSG_SPACE(16 * 4)
            )
            for level, kind, raw in controls:
                if level == socket.SOL_SOCKET and kind == socket.SCM_RIGHTS:
                    fds = array.array("i")
                    fds.frombytes(raw[: len(raw) - len(raw) % fds.itemsize])
                    for fd in fds:
                        os.close(fd)
            if controls or flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
                raise ValueError("PageBroker public reply carried ancillary data")
            if not data:
                raise ValueError("PageBroker reply truncated")
            result.extend(data)
        return bytes(result)

    data = request.SerializeToString()
    if len(data) > 65536:
        raise ValueError("PageBroker request exceeds frame limit")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as stream:
        stream.settimeout(timeout)
        stream.connect(str(path))
        stream.sendall(struct.pack("!I", len(data)) + data)
        size = struct.unpack("!I", receive(stream, 4))[0]
        if size > 65536:
            raise ValueError("PageBroker reply exceeds frame limit")
        try:
            response = pb.Response.FromString(receive(stream, size))
        except DecodeError as error:
            raise ValueError("PageBroker reply is not valid protobuf") from error
    if (response.request_id, response.transaction_id) != (
        request.request_id,
        request.transaction_id,
    ):
        raise ValueError("PageBroker reply identity mismatch")
    return response


def allocation_destinations(claim, slices):
    """Join resident tp-N allocation names to UUIDs, without a capture document."""
    devices = {}
    for item in slices["items"]:
        spec = item["spec"]
        if spec["driver"] != "gpu.nvidia.com":
            continue
        for device in spec.get("devices", []):
            key = (spec["driver"], spec["pool"]["name"], device["name"])
            value = gpu_uuid(device["attributes"]["uuid"]["string"])
            if key in devices and devices[key] != value:
                raise ValueError("conflicting resident ResourceSlice UUID")
            devices[key] = value
    requests = {}
    for entry in claim["status"]["allocation"]["devices"]["results"]:
        request = entry["request"]
        if request in requests:
            raise ValueError("resident request must select exactly one GPU")
        requests[request] = devices[(entry["driver"], entry["pool"], entry["device"])]
    if set(requests) != {f"tp-{rank}" for rank in range(8)}:
        raise ValueError("resident allocation requires exactly tp-0 through tp-7")
    if len(set(requests.values())) != 8:
        raise ValueError("resident destination UUIDs must be unique")
    return {rank: requests[f"tp-{rank}"] for rank in range(8)}


def canonical_uuid(value, name):
    if not isinstance(value, str) or str(uuid.UUID(value)) != value:
        raise ValueError(f"{name} must be a canonical UUID")
    return value


class PageBrokerCoordinator:
    def __init__(
        self,
        claim,
        slices,
        service_generation,
        root,
        app,
        broker_socket,
        *,
        artifact_root=Path("/checkpoints/artifacts"),
        broker_pvc_root=Path("/checkpoints/gms-pvc"),
    ):
        self.claim = json.loads(json.dumps(claim))
        self.slices = json.loads(json.dumps(slices))
        self.destinations = allocation_destinations(self.claim, self.slices)
        self.service_generation = canonical_uuid(
            service_generation, "service generation"
        )
        self.root = Path(root)
        self.app = Path(app)
        self.broker_socket = broker_socket
        # Do not stat, resolve, or open this PVC path during service initialization.
        self.artifact_root = Path(artifact_root)
        self.broker_pvc_root = Path(broker_pvc_root)
        if any(
            not path.is_absolute() or ".." in path.parts
            for path in (self.artifact_root, self.broker_pvc_root)
        ):
            raise ValueError("artifact and broker PVC roots must be absolute")
        self.pvc_root = self.artifact_root.parent
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = "waiting"
        self.error = None
        self.plan = None
        self.capture_id = None
        self.generation = None
        self.dgd_uid = None
        self.snapshot_id = None
        self.ranks = {}
        self.bound_servers = {}
        self.root.mkdir(parents=True, exist_ok=True)
        self._marker_names = [
            "load-trigger.json",
            "load-request.json",
            "restore-plan.json",
            "all-ready",
            "coordinator-error.json",
            *(f"rank-{rank}.json" for rank in self.destinations),
            *(f"published-{rank}" for rank in self.destinations),
        ]
        if any((self.root / name).exists() for name in self._marker_names):
            raise ValueError(
                "qualification coordinator requires fresh load state; reset is unsupported"
            )

    def _publications(self):
        return sum(
            (self.root / f"published-{rank}").exists() for rank in self.destinations
        )

    def _online(self):
        records = []
        for rank, destination_uuid in self.destinations.items():
            path = self.root / f"online-{rank}.json"
            if not path.exists():
                continue
            record = json.loads(path.read_text())
            for key, value in {
                "rank": rank,
                "service_generation": self.service_generation,
                "uuid": destination_uuid,
                "mode": "pagebroker",
                "payload_bytes_read": 0,
                "weight_allocations": 0,
                "cuda_context_current": False,
                "cuda_primary_context_active": False,
                "socket_device": rank,
                "socket_dir": "/gms",
            }.items():
                if record.get(key) != value:
                    raise ValueError(f"rank {rank} online field {key} mismatch")
            if (
                not isinstance(record.get("weights_server_nonce"), str)
                or not record["weights_server_nonce"]
            ):
                raise ValueError("missing GMS server incarnation")
            if sorted(record.get("server_domains_responsive", [])) != [
                "kv_cache",
                "weights",
            ]:
                raise ValueError("GMS domains not responsive")
            daemon, online = record["daemon_started_epoch"], record["online_epoch"]
            if not (
                math.isfinite(daemon) and math.isfinite(online) and 0 < daemon <= online
            ):
                raise ValueError("invalid GMS startup timestamps")
            records.append(record)
        return records

    def _ready(self):
        records = self._online() if self.state == "waiting" else []
        ready = (
            self.state == "waiting"
            and len(records) == 8
            and not any((self.root / name).exists() for name in self._marker_names)
        )
        if ready:
            request = pb.Request(
                request_id=str(uuid.uuid4()), transaction_id="gms-ready"
            )
            response = rpc(self.broker_socket, request, timeout=5)
            if (
                response.WhichOneof("result") != "failure"
                or response.failure.code != pb.Failure.INVALID_REQUEST
            ):
                raise ValueError(
                    "PageBroker readiness probe returned unexpected response"
                )
        return {
            "service_generation": self.service_generation,
            "capture_id": self.capture_id,
            "generation": self.generation,
            "ready": ready,
            "publications": self._publications(),
            "state": self.state,
            "online": records,
            "error": self.error,
            "transfer_owner": "resident_pagebroker_gpu_engine",
            "transfer_slots_per_gpu": 32,
            "chunk_mib": 128,
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

    def _fail(self, error):
        self.error = str(error)
        self.state = "error"
        record = {
            "error": self.error,
            "service_generation": self.service_generation,
            "dgd_uid": self.dgd_uid,
            "snapshot_id": self.snapshot_id,
            "capture_id": self.capture_id,
            "generation": self.generation,
            "epoch": time.time(),
        }
        atomic_write(self.root / "coordinator-error.json", json.dumps(record))
        emit("coordinator_error", **record)

    def load(self, request):
        required = {"dgd_uid", "snapshot_id", "generation", "capture_manifest_path"}
        if not isinstance(request, dict) or set(request) != required:
            raise ValueError(
                "load requires only dgd_uid, snapshot_id, generation, capture_manifest_path"
            )
        dgd_uid = canonical_uuid(request["dgd_uid"], "DGD UID")
        snapshot_id = canonical_uuid(request["snapshot_id"], "snapshot ID")
        generation = canonical_uuid(request["generation"], "load generation")
        if generation != dgd_uid:
            raise ValueError("qualification load generation must equal its DGD UID")
        expected_path = self.artifact_root / snapshot_id / "gms/capture.json"
        if (
            not isinstance(request["capture_manifest_path"], str)
            or Path(request["capture_manifest_path"]) != expected_path
        ):
            raise ValueError(
                "capture manifest must be the selected Snapshot artifact's gms/capture.json"
            )
        with self.lock:
            ready = self._ready()
            if not ready["ready"]:
                raise LoadConflict(
                    "qualification coordinator is not pristine and ready; reset is unsupported"
                )
            self.dgd_uid, self.snapshot_id, self.generation = (
                dgd_uid,
                snapshot_id,
                generation,
            )
            self.bound_servers = {record["rank"]: record for record in ready["online"]}
            self.state = "discovering"
            request_epoch = time.time()
            try:
                atomic_write(
                    self.root / "load-request.json",
                    json.dumps({**request, "load_request_epoch": request_epoch}),
                    exclusive=True,
                )
                discovery_start = time.time()
                emit(
                    "manifest_discovery_start",
                    dgd_uid=dgd_uid,
                    snapshot_id=snapshot_id,
                    generation=generation,
                    capture_manifest_path=str(expected_path),
                    manifest_discovery_start_epoch=discovery_start,
                )
                if expected_path.resolve(strict=True) != expected_path:
                    raise ValueError("capture manifest path must not traverse symlinks")
                with expected_path.open("rb") as manifest_file:
                    data = manifest_file.read((1 << 20) + 1)
                if len(data) > 1 << 20:
                    raise ValueError("capture manifest exceeds metadata size limit")
                capture = json.loads(data)
                if not isinstance(capture, dict):
                    raise TypeError("capture manifest must be an object")
                if capture.get("snapshot_content_uid") != snapshot_id:
                    raise ValueError("capture manifest Snapshot content UID mismatch")
                if (
                    not isinstance(capture.get("snapshot_name"), str)
                    or not capture["snapshot_name"]
                    or capture["snapshot_name"] != capture.get("capture_id")
                ):
                    raise ValueError(
                        "capture manifest snapshot name/capture identity mismatch"
                    )
                if capture.get("layout") != {"tp": 8, "pp": 1, "dp": 1}:
                    raise ValueError("PageBroker qualification requires a TP8 capture")
                if not isinstance(capture.get("ranks"), list) or not all(
                    isinstance(rank, dict) for rank in capture["ranks"]
                ):
                    raise ValueError("capture ranks must be a list of records")
                for rank in capture["ranks"]:
                    artifact = Path(rank["artifact"])
                    if (
                        not artifact.is_absolute()
                        or ".." in artifact.parts
                        or not artifact.is_relative_to(self.pvc_root)
                    ):
                        raise ValueError("rank artifact must remain on the mounted PVC")
                    if (
                        rank["socket_device"] != rank["rank"]
                        or rank["socket_dir"] != "/gms"
                    ):
                        raise ValueError("unsupported captured GMS socket layout")
                # This opens only per-rank manifest metadata and checks exact IDs,
                # sizes and hashes. Payload shards are opened by native PB later.
                plan = resolve(capture, self.claim, self.slices)
                ranks = {rank["rank"]: rank for rank in plan["ranks"]}
                if {
                    rank: record["destination_uuid"] for rank, record in ranks.items()
                } != self.destinations:
                    raise ValueError(
                        "selected capture differs from resident DRA allocation"
                    )
                self.plan, self.ranks, self.capture_id = plan, ranks, plan["capture_id"]
                discovery_complete = time.time()
                selection = {
                    **request,
                    "service_generation": self.service_generation,
                    "capture_id": self.capture_id,
                    "capture_manifest_sha256": hashlib.sha256(data).hexdigest(),
                    "load_request_epoch": request_epoch,
                    "manifest_discovery_start_epoch": discovery_start,
                    "manifest_discovery_complete_epoch": discovery_complete,
                }
                atomic_write(
                    self.root / "restore-plan.json", json.dumps(plan), exclusive=True
                )
                emit("manifest_discovery_complete", **selection)
                trigger = {**selection, "trigger_written_epoch": time.time()}
                atomic_write(
                    self.root / "load-trigger.json", json.dumps(trigger), exclusive=True
                )
                self.state = "loading"
                emit("load_trigger_written", **trigger)
                return trigger
            except (OSError, ValueError, KeyError, TypeError) as error:
                self._fail(error)
                raise

    def _load_rank(self, rank, trigger):
        expected = self.ranks[rank]
        online = self.bound_servers[rank]
        operation = pb.LoadGmsWeightsRequest(
            capture_id=self.capture_id,
            generation=self.generation,
            rank=rank,
            destination_uuid=expected["destination_uuid"],
            expected_server_nonce=online["weights_server_nonce"],
            socket_path=f"/gms/{self.service_generation}/gms_{rank}_weights.sock",
            artifact_directory=str(
                self.broker_pvc_root
                / Path(expected["artifact"]).relative_to(self.pvc_root)
            ),
            manifest_sha256=expected["manifest_sha256"],
            allocations=[
                pb.GmsAllocation(**allocation) for allocation in expected["allocations"]
            ],
            timeout_seconds=180,
        )
        request = pb.Request(
            request_id=str(uuid.uuid4()),
            transaction_id=f"gms-{self.generation}-{rank}",
            load_gms_weights=operation,
        )
        started = time.time()
        emit(
            "pagebroker_request_start",
            rank=rank,
            generation=self.generation,
            dgd_uid=self.dgd_uid,
        )
        response = rpc(self.broker_socket, request)
        if response.WhichOneof("result") != "gms_weights_loaded":
            raise RuntimeError(f"rank {rank} PageBroker failed: {response}")
        report = json.loads(response.gms_weights_loaded.report)
        for key, value in {
            "capture_id": self.capture_id,
            "generation": self.generation,
            "rank": rank,
            "destination_uuid": expected["destination_uuid"],
            "server_nonce": online["weights_server_nonce"],
            "manifest_sha256": expected["manifest_sha256"],
            "allocations": len(expected["allocations"]),
            "bytes": sum(a["aligned_size"] for a in expected["allocations"]),
        }.items():
            if report.get(key) != value:
                raise ValueError(f"PageBroker report field {key} mismatch")
        published = time.time()
        if not started <= report["committed_epoch_ns"] / 1e9 <= published:
            raise ValueError("PageBroker commit timestamp is outside its request")
        record = {
            "rank": rank,
            "uuid": expected["destination_uuid"],
            "dgd_uid": self.dgd_uid,
            "snapshot_id": self.snapshot_id,
            "service_generation": self.service_generation,
            "capture_id": self.capture_id,
            "generation": self.generation,
            "workers": 32,
            "chunk_mib": 128,
            "mode": "pagebroker",
            "daemon_started_epoch": online["daemon_started_epoch"],
            "online_epoch": online["online_epoch"],
            "trigger_written_epoch": trigger["trigger_written_epoch"],
            "started_epoch": started,
            "published_epoch": published,
            "elapsed_s": published - started,
            "pagebroker_report": report,
        }
        atomic_write(
            self.root / f"rank-{rank}.json", json.dumps(record), exclusive=True
        )
        atomic_write(
            self.root / f"published-{rank}",
            expected["destination_uuid"],
            exclusive=True,
        )
        emit("pagebroker_published", **record)

    def watch_publication(self):
        while not self.stop.wait(0.002):
            with self.lock:
                if self.state != "loading":
                    continue
                self.state = "transferring"
            try:
                trigger = json.loads((self.root / "load-trigger.json").read_text())
                with ThreadPoolExecutor(max_workers=8) as pool:
                    futures = [
                        pool.submit(self._load_rank, rank, trigger)
                        for rank in self.ranks
                    ]
                    for future in futures:
                        future.result()
                online = {record["rank"]: record for record in self._online()}
                if set(online) != set(self.bound_servers) or any(
                    online[rank]["weights_server_nonce"]
                    != record["weights_server_nonce"]
                    for rank, record in self.bound_servers.items()
                ):
                    raise ValueError("GMS server incarnation changed during load")
                emit("publication_verification_started", dgd_uid=self.dgd_uid)
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
                    dgd_uid=self.dgd_uid,
                    snapshot_id=self.snapshot_id,
                    capture_id=self.capture_id,
                    generation=self.generation,
                )
            except (
                OSError,
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                subprocess.SubprocessError,
            ) as error:
                if isinstance(error, subprocess.CalledProcessError):
                    error = RuntimeError(f"publication verifier failed: {error.stderr}")
                with self.lock:
                    self._fail(error)
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-generation", required=True)
    parser.add_argument("--root", type=Path, default=Path("/gms"))
    parser.add_argument("--app", type=Path, default=Path("/snapshot-app"))
    parser.add_argument(
        "--broker-socket", default="/pagebroker/control/pagebroker.sock"
    )
    args = parser.parse_args()
    coordinator = PageBrokerCoordinator(
        json.loads((args.app / "claim.json").read_text()),
        json.loads((args.app / "slices.json").read_text()),
        args.service_generation,
        args.root,
        args.app,
        args.broker_socket,
    )
    server = make_server(coordinator, ("0.0.0.0", 18081))
    watcher = threading.Thread(target=coordinator.watch_publication, daemon=True)
    watcher.start()

    def shutdown(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    emit("pagebroker_coordinator_listening", service_generation=args.service_generation)
    try:
        server.serve_forever(poll_interval=0.1)
    finally:
        coordinator.stop.set()
        server.server_close()
        watcher.join(timeout=245)


if __name__ == "__main__":
    main()
