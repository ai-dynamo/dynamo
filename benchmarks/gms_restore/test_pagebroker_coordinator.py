# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU protocol and publication checks for the native PageBroker coordinator."""

import array
import hashlib
import json
import os
import socket
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import pagebroker_coordinator as coordinator_module
import pagebroker_pb2 as pb
import pytest
from pagebroker_coordinator import PageBrokerCoordinator, rpc
from resident_coordinator import LoadConflict


def receive_request(stream):
    def exact(size):
        data = bytearray()
        while len(data) < size:
            chunk = stream.recv(size - len(data))
            if not chunk:
                raise EOFError("test peer received a truncated request")
            data.extend(chunk)
        return bytes(data)

    size = struct.unpack("!I", exact(4))[0]
    return pb.Request.FromString(exact(size))


def framed(response):
    payload = response if isinstance(response, bytes) else response.SerializeToString()
    return struct.pack("!I", len(payload)) + payload


def response_for(request):
    return pb.Response(
        request_id=request.request_id, transaction_id=request.transaction_id
    )


def ping_reply(request):
    response = response_for(request)
    response.failure.code = pb.Failure.INVALID_REQUEST
    return response


@pytest.fixture
def unix_peer(tmp_path):
    @contextmanager
    def serve(handler):
        path = tmp_path / "native.sock"
        errors = []
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(str(path))
            listener.listen(1)
            listener.settimeout(2)

            def run():
                try:
                    stream, _ = listener.accept()
                    with stream:
                        stream.settimeout(2)
                        handler(stream, receive_request(stream))
                except Exception as error:  # noqa: BLE001 -- Re-raise on pytest's thread.
                    errors.append(error)

            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            try:
                yield path
            finally:
                thread.join(timeout=3)
                assert not thread.is_alive(), "test native peer failed to finish"
                path.unlink(missing_ok=True)
                if errors:
                    raise errors[0]

    return serve


def test_rpc_accepts_fragmented_protobuf_header_and_payload(unix_peer):
    request = pb.Request(request_id="request-one", transaction_id="transaction-one")
    observed = []

    def peer(stream, actual):
        observed.append(actual)
        for byte in framed(ping_reply(actual)):
            stream.sendall(bytes([byte]))
            time.sleep(0.001)

    with unix_peer(peer) as path:
        response = rpc(path, request, timeout=2)
    assert observed == [request]
    assert response.failure.code == pb.Failure.INVALID_REQUEST


@pytest.mark.parametrize("descriptor_count", [1, 32])
def test_rpc_rejects_and_closes_received_or_truncated_descriptors(
    unix_peer, descriptor_count
):
    before = len(list(Path("/proc/self/fd").iterdir()))
    read_fd, write_fd = os.pipe()
    try:

        def peer(stream, request):
            message = framed(ping_reply(request))
            # Deliver SCM_RIGHTS with a payload byte, not the frame header.
            stream.sendall(message[:4])
            descriptors = array.array("i", [read_fd] * descriptor_count)
            stream.sendmsg(
                [message[4:]], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, descriptors)]
            )

        with (
            unix_peer(peer) as path,
            pytest.raises(ValueError, match="ancillary"),
        ):
            rpc(path, pb.Request(request_id="r", transaction_id="t"))
    finally:
        os.close(read_fd)
        os.close(write_fd)
    assert len(list(Path("/proc/self/fd").iterdir())) == before


@pytest.mark.parametrize("field", ["request_id", "transaction_id"])
def test_rpc_rejects_reply_from_a_different_request(unix_peer, field):
    def peer(stream, request):
        response = ping_reply(request)
        setattr(response, field, "another-request")
        stream.sendall(framed(response))

    with unix_peer(peer) as path, pytest.raises(ValueError, match="identity mismatch"):
        rpc(path, pb.Request(request_id="r", transaction_id="t"))


@pytest.mark.parametrize("mode", ["protobuf", "truncated", "oversized"])
def test_rpc_normalizes_bad_frames_to_readiness_errors(unix_peer, mode):
    def peer(stream, _request):
        payload = {
            "protobuf": framed(b"\xff"),
            "truncated": struct.pack("!I", 10) + b"short",
            "oversized": struct.pack("!I", 65537),
        }[mode]
        stream.sendall(payload)

    with unix_peer(peer) as path, pytest.raises(ValueError):
        rpc(path, pb.Request(request_id="r", transaction_id="t"))


def test_rpc_has_one_deadline_across_slow_fragment_receives(unix_peer):
    def peer(stream, _request):
        try:
            stream.sendall(struct.pack("!I", 100))
            for _ in range(8):
                stream.sendall(b"x")
                time.sleep(0.025)
        except (BrokenPipeError, ConnectionResetError):
            pass

    with unix_peer(peer) as path:
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            rpc(path, pb.Request(request_id="r", transaction_id="t"), timeout=0.06)
        elapsed = time.monotonic() - started
    assert elapsed < 0.18  # A timeout reset per byte would reach the peer's EOF.


def test_rpc_rejects_large_request_before_connecting(tmp_path):
    with pytest.raises(ValueError, match="request exceeds"):
        rpc(tmp_path / "does-not-exist", pb.Request(request_id="x" * 65537))


@pytest.fixture
def coordinator(tmp_path):
    service_generation = "10000000-0000-0000-0000-000000000001"
    dgd_uid = "20000000-0000-0000-0000-000000000002"
    snapshot_id = "30000000-0000-0000-0000-000000000003"
    pvc = tmp_path / "pvc"
    ranks = []
    devices = []
    allocated = []
    for rank in [4, 1, 7, 2, 0, 6, 3, 5]:
        artifact = pvc / f"capture-a/model/device-{rank}"
        artifact.mkdir(parents=True)
        allocations = [
            {
                "allocation_id": f"capture-rank-{rank}-allocation-{index}",
                "aligned_size": 2 * 1024**2,
                "shard": "shards/shard_0000.bin",
                "offset": index * 2 * 1024**2,
            }
            for index in range(2)
        ]
        manifest = json.dumps({"version": 1, "allocations": allocations}).encode()
        (artifact / "manifest.json").write_bytes(manifest)
        ranks.append(
            {
                "rank": rank,
                "source_uuid": f"GPU-00000000-0000-0000-0000-{rank + 100:012x}",
                "captured_ordinal": rank,
                "socket_device": rank,
                "socket_dir": "/gms",
                "artifact": str(artifact),
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
                "allocations": allocations,
            }
        )
        devices.append(
            {
                "name": f"device-{rank}",
                "attributes": {
                    "uuid": {"string": f"GPU-00000000-0000-0000-0000-{rank + 10:012x}"}
                },
            }
        )
        allocated.append(
            {
                "request": f"tp-{rank}",
                "driver": "gpu.nvidia.com",
                "pool": "node-a",
                "device": f"device-{rank}",
            }
        )
    claim = {
        "metadata": {"uid": "claim-a"},
        "status": {"allocation": {"devices": {"results": allocated}}},
    }
    slices = {
        "items": [
            {
                "spec": {
                    "driver": "gpu.nvidia.com",
                    "pool": {"name": "node-a"},
                    "devices": devices,
                }
            }
        ]
    }
    capture = {
        "capture_id": "capture-a",
        "snapshot_name": "capture-a",
        "snapshot_content_uid": snapshot_id,
        "layout": {"tp": 8, "pp": 1, "dp": 1},
        "ranks": ranks,
    }
    artifact_root = pvc / "artifacts"
    capture_path = artifact_root / snapshot_id / "gms/capture.json"
    capture_path.parent.mkdir(parents=True)
    capture_path.write_text(json.dumps(capture))
    instance = PageBrokerCoordinator(
        claim,
        slices,
        service_generation,
        tmp_path / "gms",
        tmp_path,
        tmp_path / "native.sock",
        artifact_root=artifact_root,
    )
    instance.test_request = {
        "dgd_uid": dgd_uid,
        "snapshot_id": snapshot_id,
        "generation": dgd_uid,
        "capture_manifest_path": str(capture_path),
    }
    write_online(instance)
    return instance


def write_online(instance):
    for rank, destination_uuid in instance.destinations.items():
        online = {
            "rank": rank,
            "service_generation": instance.service_generation,
            "uuid": destination_uuid,
            "socket_device": rank,
            "socket_dir": "/gms",
            "mode": "pagebroker",
            "payload_bytes_read": 0,
            "weight_allocations": 0,
            "cuda_context_current": False,
            "cuda_primary_context_active": False,
            "weights_server_nonce": f"current-server-rank-{rank}",
            "server_domains_responsive": ["weights", "kv_cache"],
            "daemon_started_epoch": 1,
            "online_epoch": 2,
        }
        (instance.root / f"online-{rank}.json").write_text(json.dumps(online))


def native_reply(_path, request, timeout=240):
    del timeout
    if request.WhichOneof("command") is None:
        return ping_reply(request)
    operation = request.load_gms_weights
    response = response_for(request)
    response.gms_weights_loaded.report = json.dumps(
        {
            "capture_id": operation.capture_id,
            "generation": operation.generation,
            "rank": operation.rank,
            "destination_uuid": operation.destination_uuid,
            "server_nonce": operation.expected_server_nonce,
            "manifest_sha256": operation.manifest_sha256,
            "allocations": len(operation.allocations),
            "bytes": sum(a.aligned_size for a in operation.allocations),
            "committed_epoch_ns": time.time_ns(),
        }
    )
    return response


def trigger_payload(coordinator):
    return dict(coordinator.test_request)


def test_native_request_preserves_rank_uuid_nonce_and_capture_extents(coordinator):
    rank = 7
    with patch.object(coordinator_module, "rpc", side_effect=native_reply) as native:
        trigger = coordinator.load(trigger_payload(coordinator))
        coordinator._load_rank(rank, trigger)
    path, request = native.call_args.args
    expected = coordinator.ranks[rank]
    operation = request.load_gms_weights
    assert path == coordinator.broker_socket
    assert request.transaction_id == f"gms-{coordinator.generation}-{rank}"
    assert operation.capture_id == coordinator.capture_id
    assert operation.generation == coordinator.generation
    assert operation.rank == rank
    assert operation.destination_uuid == expected["destination_uuid"]
    assert operation.expected_server_nonce == "current-server-rank-7"
    assert (
        operation.socket_path
        == f"/gms/{coordinator.service_generation}/gms_7_weights.sock"
    )
    assert (
        operation.artifact_directory == "/checkpoints/gms-pvc/capture-a/model/device-7"
    )
    assert operation.manifest_sha256 == expected["manifest_sha256"]
    assert operation.timeout_seconds == 180
    assert list(operation.allocations) == [
        pb.GmsAllocation(**record) for record in expected["allocations"]
    ]
    record = json.loads((coordinator.root / "rank-7.json").read_text())
    assert record["uuid"] == expected["destination_uuid"]
    assert (
        record["pagebroker_report"]["server_nonce"] == operation.expected_server_nonce
    )
    assert record["trigger_written_epoch"] <= record["started_epoch"]
    assert (
        record["started_epoch"]
        <= record["pagebroker_report"]["committed_epoch_ns"] / 1e9
        <= record["published_epoch"]
    )
    assert (coordinator.root / "published-7").read_text() == expected[
        "destination_uuid"
    ]
    assert not (coordinator.root / "all-ready").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("server_nonce", "another-incarnation"),
        ("manifest_sha256", "unrelated-artifact"),
        ("destination_uuid", "wrong-gpu"),
        ("rank", 5),
        ("bytes", 1),
        ("committed_epoch_ns", 1),
        ("committed_epoch_ns", 2**63 - 1),
    ],
)
def test_bad_native_report_never_publishes_rank(coordinator, field, value):
    def invalid(*args, **kwargs):
        response = native_reply(*args, **kwargs)
        report = json.loads(response.gms_weights_loaded.report)
        report[field] = value
        response.gms_weights_loaded.report = json.dumps(report)
        return response

    with patch.object(coordinator_module, "rpc", side_effect=native_reply):
        coordinator.load(trigger_payload(coordinator))
    with (
        patch.object(coordinator_module, "rpc", side_effect=invalid),
        pytest.raises(ValueError),
    ):
        coordinator._load_rank(3, {"trigger_written_epoch": time.time()})
    for name in ["rank-3.json", "published-3", "all-ready"]:
        assert not (coordinator.root / name).exists()


def test_pagebroker_trigger_is_generation_bound_and_single_use(coordinator):
    payload = trigger_payload(coordinator)
    with patch.object(coordinator_module, "rpc", side_effect=native_reply):
        assert coordinator.ready()["ready"]
        with pytest.raises(ValueError):
            coordinator.load(
                {**payload, "generation": "40000000-0000-0000-0000-000000000004"}
            )
        assert not (coordinator.root / "load-trigger.json").exists()

        def load():
            try:
                return coordinator.load(payload)
            except LoadConflict:
                return None

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(lambda _: load(), range(2)))
        (accepted,) = [result for result in results if result is not None]
        assert (
            json.loads((coordinator.root / "load-trigger.json").read_text()) == accepted
        )
        assert not coordinator.ready()["ready"]
        assert coordinator.health()["healthy"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("payload_bytes_read", 4096),
        ("weight_allocations", 1),
        ("cuda_context_current", True),
        ("cuda_primary_context_active", True),
        ("weights_server_nonce", ""),
    ],
)
def test_invalid_server_readiness_fails_before_trigger(coordinator, field, value):
    path = coordinator.root / "online-3.json"
    online = json.loads(path.read_text())
    online[field] = value
    path.write_text(json.dumps(online))
    with patch.object(coordinator_module, "rpc") as native:
        status = coordinator.ready()
    assert status["state"] == "error" and not status["ready"]
    assert not native.called
    assert (coordinator.root / "coordinator-error.json").exists()
    assert not (coordinator.root / "load-trigger.json").exists()


def test_truncated_broker_readiness_response_latches_error(coordinator, unix_peer):
    def peer(stream, _request):
        stream.sendall(struct.pack("!I", 20) + b"short")

    with unix_peer(peer) as path:
        coordinator.broker_socket = path
        status = coordinator.ready()
    assert not status["ready"] and status["state"] == "error"
    assert "truncated" in status["error"]
    assert not coordinator.health()["healthy"]


@pytest.mark.parametrize("failure", [None, "native", "verifier"])
def test_global_gate_requires_all_native_successes_and_inventory_verifier(
    coordinator, failure
):
    native_ranks = []

    def native(path, request, timeout=240):
        response = native_reply(path, request, timeout)
        if request.WhichOneof("command") == "load_gms_weights":
            native_ranks.append(request.load_gms_weights.rank)
            if failure == "native" and request.load_gms_weights.rank == 4:
                response.ClearField("gms_weights_loaded")
                response.failure.code = pb.Failure.INVALID_REQUEST
        return response

    def verify(*_args, **_kwargs):
        assert sorted(native_ranks) == list(range(8))
        assert all(
            (coordinator.root / f"published-{rank}").exists() for rank in range(8)
        )
        if failure == "verifier":
            raise subprocess.CalledProcessError(
                1, ["verifier"], stderr="wrong capture allocation inventory"
            )
        return subprocess.CompletedProcess(["verifier"], 0, stdout='{"published":true}')

    with (
        patch.object(coordinator_module, "rpc", side_effect=native),
        patch.object(
            coordinator_module.subprocess, "run", side_effect=verify
        ) as verifier,
    ):
        coordinator.load(trigger_payload(coordinator))
        # One native batch, one inventory attempt, then stop the parent watcher.
        with patch.object(coordinator.stop, "wait", side_effect=[False]):
            coordinator.watch_publication()
    assert sorted(native_ranks) == list(range(8))
    assert verifier.call_count == (0 if failure == "native" else 1)
    if failure:
        assert coordinator.state == "error"
        assert (coordinator.root / "coordinator-error.json").exists()
        assert not (coordinator.root / "all-ready").exists()
    else:
        assert coordinator.state == "published"
        assert float((coordinator.root / "all-ready").read_text()) > 0
        assert verifier.call_args.args[0][-1] == str(
            coordinator.root / "restore-plan.json"
        )
        assert verifier.call_args.kwargs["check"] is True


def test_generic_startup_and_ready_never_access_snapshot_metadata(
    coordinator, tmp_path, monkeypatch
):
    original_open = Path.open

    def forbid_pvc_open(path, *args, **kwargs):
        if path.is_relative_to(coordinator.pvc_root):
            raise AssertionError("PVC metadata opened before DGD-bound load")
        return original_open(path, *args, **kwargs)

    with (
        patch.object(Path, "open", forbid_pvc_open),
        patch.object(coordinator_module, "resolve") as resolve,
        patch.object(coordinator_module, "rpc", side_effect=native_reply),
    ):
        generic = PageBrokerCoordinator(
            coordinator.claim,
            coordinator.slices,
            coordinator.service_generation,
            tmp_path / "other-service",
            coordinator.app,
            coordinator.broker_socket,
            artifact_root=coordinator.artifact_root,
        )
        write_online(generic)
        ready = generic.ready()
    assert ready["ready"]
    assert ready["capture_id"] is None and ready["generation"] is None
    assert generic.plan is None and generic.ranks == {}
    assert not resolve.called
    assert not (generic.root / "restore-plan.json").exists()


def test_manifest_can_arrive_after_ready_and_discovery_is_inside_load(
    coordinator, monkeypatch
):
    path = Path(coordinator.test_request["capture_manifest_path"])
    document = path.read_bytes()
    path.unlink()
    with patch.object(coordinator_module, "rpc", side_effect=native_reply):
        assert coordinator.ready()["ready"]
    assert not path.exists()
    path.write_bytes(document)  # The selected DGD's metadata arrives afterward.
    reads = []
    original_open = Path.open

    def observe_pvc_open(selected, *args, **kwargs):
        if selected.is_relative_to(coordinator.pvc_root):
            reads.append((selected, time.time()))
        return original_open(selected, *args, **kwargs)

    monkeypatch.setattr(Path, "open", observe_pvc_open)
    with (
        patch.object(coordinator_module, "rpc", side_effect=native_reply) as native,
        patch.object(coordinator_module, "emit") as emit,
    ):
        trigger = coordinator.load(trigger_payload(coordinator))
    assert native.call_count == 1  # Readiness ping; no weight-load RPC yet.
    assert native.call_args.args[1].WhichOneof("command") is None
    assert len(reads) == 9  # Selected capture metadata and eight rank manifests.
    assert reads[0][0] == path
    assert (
        trigger["load_request_epoch"]
        <= trigger["manifest_discovery_start_epoch"]
        <= min(epoch for _, epoch in reads)
        <= max(epoch for _, epoch in reads)
        <= trigger["manifest_discovery_complete_epoch"]
        <= trigger["trigger_written_epoch"]
    )
    assert trigger["dgd_uid"] == trigger["generation"]
    assert trigger["snapshot_id"] == coordinator.test_request["snapshot_id"]
    assert trigger["service_generation"] == coordinator.service_generation
    assert trigger["capture_manifest_sha256"] == hashlib.sha256(document).hexdigest()
    assert [call.args[0] for call in emit.call_args_list] == [
        "manifest_discovery_start",
        "manifest_discovery_complete",
        "load_trigger_written",
    ]
    # Metadata discovery does not require payload shards to exist locally.
    assert not list(coordinator.pvc_root.rglob("*.bin"))
    assert not (coordinator.root / "all-ready").exists()


@pytest.mark.parametrize("kind", ["outside", "wrong-content-directory", "symlink"])
def test_selected_manifest_path_cannot_escape_snapshot_artifact(
    coordinator, tmp_path, kind
):
    request = trigger_payload(coordinator)
    expected = Path(request["capture_manifest_path"])
    if kind == "outside":
        request["capture_manifest_path"] = str(tmp_path / "unrelated.json")
    elif kind == "wrong-content-directory":
        request["capture_manifest_path"] = str(
            coordinator.artifact_root
            / "40000000-0000-0000-0000-000000000004/gms/capture.json"
        )
    else:
        unrelated = tmp_path / "unrelated.json"
        unrelated.write_bytes(expected.read_bytes())
        expected.unlink()
        expected.symlink_to(unrelated)
    with (
        patch.object(coordinator_module, "rpc", side_effect=native_reply),
        pytest.raises(ValueError),
    ):
        coordinator.load(request)
    assert not (coordinator.root / "restore-plan.json").exists()
    assert not (coordinator.root / "load-trigger.json").exists()
    assert not (coordinator.root / "all-ready").exists()


@pytest.mark.parametrize(
    "kind", ["content-uid", "manifest-hash", "allocation-id", "socket", "json-shape"]
)
def test_selected_metadata_must_match_snapshot_and_exact_capture(coordinator, kind):
    path = Path(coordinator.test_request["capture_manifest_path"])
    document = json.loads(path.read_text())
    if kind == "content-uid":
        document["snapshot_content_uid"] = "40000000-0000-0000-0000-000000000004"
    elif kind == "manifest-hash":
        document["ranks"][0]["manifest_sha256"] = "0" * 64
    elif kind == "allocation-id":
        document["ranks"][0]["allocations"][0]["allocation_id"] = "unrelated-capture"
    elif kind == "socket":
        document["ranks"][0]["socket_device"] = 99
    else:
        document = []
    path.write_text(json.dumps(document))
    with (
        patch.object(coordinator_module, "rpc", side_effect=native_reply) as native,
        pytest.raises((TypeError, ValueError)),
    ):
        coordinator.load(trigger_payload(coordinator))
    assert native.call_count == 1
    assert native.call_args.args[1].WhichOneof("command") is None
    assert coordinator.state == "error"
    assert (coordinator.root / "coordinator-error.json").exists()
    assert not (coordinator.root / "load-trigger.json").exists()
    assert not (coordinator.root / "all-ready").exists()


def test_server_nonce_is_bound_at_load_and_change_blocks_global_gate(coordinator):
    with patch.object(coordinator_module, "rpc", side_effect=native_reply) as native:
        coordinator.load(trigger_payload(coordinator))
        online_path = coordinator.root / "online-3.json"
        online = json.loads(online_path.read_text())
        online["weights_server_nonce"] = "replacement-server"
        online_path.write_text(json.dumps(online))
        with (
            patch.object(coordinator.stop, "wait", return_value=False),
            patch.object(coordinator_module.subprocess, "run") as verifier,
        ):
            coordinator.watch_publication()
    request = next(
        call.args[1].load_gms_weights
        for call in native.call_args_list
        if call.args[1].WhichOneof("command") == "load_gms_weights"
        and call.args[1].load_gms_weights.rank == 3
    )
    assert request.expected_server_nonce == "current-server-rank-3"
    assert coordinator.state == "error"
    assert "incarnation changed" in coordinator.error
    assert not verifier.called
    assert not (coordinator.root / "all-ready").exists()
