# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import subprocess
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import pytest
from resident_coordinator import Coordinator, make_server


@pytest.fixture
def coordinator(tmp_path):
    plan = {
        "capture_id": "capture-a",
        "ranks": [
            {"rank": rank, "destination_uuid": f"gpu-{rank}"} for rank in range(8)
        ],
    }
    return Coordinator(plan, "generation-a", 16, 128, tmp_path / "gms", tmp_path)


def write_online(coordinator):
    for rank in range(8):
        record = {
            "rank": rank,
            "capture_id": coordinator.capture_id,
            "generation": coordinator.generation,
            "uuid": f"gpu-{rank}",
            "workers": 16,
            "chunk_mib": 128,
            "loader_lanes_ready": 16,
            "pinned_bytes": 16 * 2 * 128 * 1024**2,
            "payload_bytes_read": 0,
            "weight_allocations": 0,
            "server_domains_responsive": ["weights", "kv_cache"],
            "daemon_started_epoch": 1,
            "online_epoch": 2,
        }
        (coordinator.root / f"online-{rank}.json").write_text(json.dumps(record))


@pytest.fixture
def http_server(coordinator):
    server = make_server(coordinator, ("127.0.0.1", 0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(url, payload=None):
    body = None if payload is None else json.dumps(payload).encode()
    try:
        response = urllib.request.urlopen(
            urllib.request.Request(url, data=body), timeout=5
        )
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, json.load(response)


def test_http_requires_pristine_generation_and_accepts_only_one_trigger(
    coordinator, http_server
):
    payload = {
        "capture_id": coordinator.capture_id,
        "generation": coordinator.generation,
    }
    assert request(http_server + "/ready")[0] == 503
    assert request(http_server + "/load", payload)[0] == 409
    write_online(coordinator)
    status, ready = request(http_server + "/ready")
    assert status == 200 and ready["ready"] and ready["publications"] == 0
    assert request(http_server + "/load", {**payload, "generation": "stale"})[0] == 409
    assert not (coordinator.root / "load-trigger.json").exists()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(request, http_server + "/load", payload) for _ in range(2)
        ]
        outcomes = [future.result() for future in futures]
    assert sorted(status for status, _ in outcomes) == [200, 409]
    accepted = next(value for status, value in outcomes if status == 200)
    assert json.loads((coordinator.root / "load-trigger.json").read_text()) == accepted
    assert accepted["generation"] == coordinator.generation
    assert accepted["trigger_written_epoch"] > 0
    assert request(http_server + "/ready")[0] == 503
    assert request(http_server + "/healthz")[0] == 200


@pytest.mark.parametrize(
    "field,value",
    [("uuid", "wrong-gpu"), ("payload_bytes_read", 4096), ("weight_allocations", 1)],
)
def test_online_identity_and_no_preload_contract_fail_closed(coordinator, field, value):
    write_online(coordinator)
    path = coordinator.root / "online-3.json"
    record = json.loads(path.read_text())
    path.write_text(json.dumps({**record, field: value}))
    ready = coordinator.ready()
    assert not ready["ready"] and ready["state"] == "error"
    assert field in ready["error"]
    assert (coordinator.root / "coordinator-error.json").exists()
    assert not (coordinator.root / "load-trigger.json").exists()


@pytest.mark.parametrize("verifier_fails", [False, True])
def test_publication_requires_exact_inventory_verifier(coordinator, verifier_fails):
    write_online(coordinator)
    coordinator.load(
        {"capture_id": coordinator.capture_id, "generation": coordinator.generation}
    )
    for rank in range(8):
        record = {
            "rank": rank,
            "capture_id": coordinator.capture_id,
            "generation": coordinator.generation,
            "uuid": f"gpu-{rank}",
            "workers": 16,
            "chunk_mib": 128,
        }
        (coordinator.root / f"rank-{rank}.json").write_text(json.dumps(record))
        (coordinator.root / f"published-{rank}").write_text(f"gpu-{rank}")
    failure = subprocess.CalledProcessError(
        1, ["verifier"], stderr="wrong capture allocations"
    )
    with (
        patch.object(coordinator.stop, "wait", side_effect=[False, True]),
        patch(
            "resident_coordinator.subprocess.run",
            side_effect=failure if verifier_fails else None,
            return_value=subprocess.CompletedProcess(
                ["verifier"], 0, stdout='{"published":true}'
            ),
        ) as verifier,
    ):
        coordinator.watch_publication()
    assert verifier.call_count == 1
    assert verifier.call_args.args[0][-1] == str(coordinator.root / "restore-plan.json")
    assert verifier.call_args.kwargs["check"] is True
    assert (coordinator.root / "all-ready").exists() is not verifier_fails
    assert coordinator.state == ("error" if verifier_fails else "published")
    if verifier_fails:
        assert "wrong capture allocations" in coordinator.error


def test_stale_trigger_is_not_reused_or_overwritten(coordinator):
    path = coordinator.root / "load-trigger.json"
    path.write_text('{"generation":"previous"}')
    with pytest.raises(ValueError, match="stale state"):
        Coordinator(
            coordinator.plan, "replacement", 16, 128, coordinator.root, coordinator.app
        )
    assert json.loads(path.read_text()) == {"generation": "previous"}
