# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HTTP triggering and pod wiring for a prestarted, one-shot GMS DaemonSet."""

import copy
import json
import time
import urllib.request


def coordinator_request(url, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def trigger_load(url, capture_id, generation):
    """The node-side write time excludes local HTTP transport overhead."""
    requested = time.time()
    response = coordinator_request(
        url, "/load", {"capture_id": capture_id, "generation": generation}
    )
    returned = time.time()
    return {
        "request_epoch": requested,
        "trigger_written_epoch": float(response["trigger_written_epoch"]),
        "return_epoch": returned,
        "response": response,
    }


def engine_only_pod(pod, host_path):
    """Keep captured socket names while moving server ownership out of the DGD."""
    result = copy.deepcopy(pod)
    result["spec"]["containers"] = [
        c for c in result["spec"]["containers"] if c["name"] == "main"
    ]
    assert len(result["spec"]["containers"]) == 1
    referenced = {
        mount["name"]
        for container in (
            result["spec"]["containers"] + result["spec"].get("initContainers", [])
        )
        for mount in container.get("volumeMounts", [])
    }
    result["spec"]["volumes"] = [
        v for v in result["spec"]["volumes"] if v["name"] in referenced
    ]
    volume = next(v for v in result["spec"]["volumes"] if v["name"] == "gms")
    volume.clear()
    volume.update(name="gms", hostPath={"path": host_path, "type": "Directory"})
    return result


def validate_resident_pod(
    pod, claim_name, host_path, node_name, cpu_request, cpu_limit, numa
):
    """Require the same held claim and a distinct one-GPU request per rank."""
    assert any(
        owner["kind"] == "DaemonSet" and owner.get("controller")
        for owner in pod["metadata"].get("ownerReferences", [])
    ), "resident GMS Pod must be owned by a DaemonSet"
    assert pod["spec"].get("nodeName") == node_name, "resident GMS is on another node"
    assert any(
        condition["type"] == "Ready" and condition["status"] == "True"
        for condition in pod.get("status", {}).get("conditions", [])
    ), "resident GMS Pod is not Ready before the trial"
    validate_resident_restarts(pod)
    claims = {claim["name"]: claim for claim in pod["spec"].get("resourceClaims", [])}
    containers = {c["name"]: c for c in pod["spec"]["containers"]}
    volumes = {v["name"]: v for v in pod["spec"]["volumes"]}
    for rank in range(8):
        container = containers[f"gms-{rank}"]
        claim_refs = container["resources"]["claims"]
        assert len(claim_refs) == 1
        claim = claim_refs[0]
        assert claim["request"] == f"tp-{rank}", "resident rank request mismatch"
        assert claims[claim["name"]]["resourceClaimName"] == claim_name
        mount = next(m for m in container["volumeMounts"] if m["mountPath"] == "/gms")
        assert volumes[mount["name"]]["hostPath"]["path"] == host_path
        assert container["resources"]["requests"]["cpu"] == str(cpu_request)
        assert container["resources"]["limits"]["cpu"] == str(cpu_limit)
        command = container.get("command", []) + container.get("args", [])
        assert ("--numa" in command) == numa, "resident rank NUMA setting mismatch"


def validate_resident_restarts(pod):
    """A replacement resident process invalidates one-shot readiness evidence."""
    statuses = {
        status["name"]: status
        for status in pod.get("status", {}).get("containerStatuses", [])
    }
    for container in pod["spec"]["containers"]:
        name = container["name"]
        assert name in statuses, f"resident container {name} status is missing"
        assert statuses[name]["restartCount"] == 0, (
            f"resident container {name} restarted"
        )
        assert "running" in statuses[name]["state"], (
            f"resident container {name} is not running"
        )


def validate_resident_online(ready, workers, chunk_mib):
    """Check the actual warmed configuration before either timed request starts."""
    online = ready["online"]
    assert len(online) == 8 and {rank["rank"] for rank in online} == set(range(8))
    for rank in online:
        assert rank["workers"] == workers, "resident loader worker count mismatch"
        assert rank["chunk_mib"] == chunk_mib, "resident loader chunk size mismatch"
