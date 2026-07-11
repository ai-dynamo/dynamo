# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HA topology evidence and destructive failover proof for the sweep."""

from __future__ import annotations

import json
import secrets
import subprocess
import time
from typing import Any, Callable

from .cluster import (
    Cluster,
    GROUP_IDENTITIES,
    pods,
    registered_mocker_stats,
    replication_snapshot,
    selected_valkey_info,
    sentinel_master,
    wait_for_ha,
    wait_for_no_pods,
    wait_for_valkey_value,
)
from .model import MODEL


ApplicationProbe = Callable[[], dict[str, Any]]


def elected_identity(
    cluster: Cluster, master_name: str
) -> tuple[str, str, dict[str, dict[str, str]], tuple[str, int]]:
    return replication_snapshot(cluster, master_name)


def ha_snapshot(cluster: Cluster) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    for master_name in GROUP_IDENTITIES:
        elected, replica, infos, address = elected_identity(cluster, master_name)
        connected = int(infos[elected].get("connected_slaves", "0"))
        link = infos[replica].get("master_link_status")
        if connected < 1 or link != "up":
            raise RuntimeError(
                f"unhealthy replication for {master_name}: "
                f"connected_slaves={connected}, replica_link={link}, infos={infos}"
            )
        topology_infos = {
            identity: {
                key: value
                for key, value in info.items()
                if key in {"role", "connected_slaves", "master_link_status"}
            }
            for identity, info in infos.items()
        }
        snapshot[master_name] = {
            "sentinel_master": list(address),
            "elected_identity": elected,
            "replica_identity": replica,
            "identities": topology_infos,
        }
    return snapshot


def valkey_telemetry(
    cluster: Cluster, topology: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    return {
        group: {
            "elected_master": evidence["elected_identity"],
            "replica": evidence["replica_identity"],
            "identities": {
                identity: selected_valkey_info(cluster, identity)
                for identity in GROUP_IDENTITIES[group]
            },
        }
        for group, evidence in topology.items()
    }


def valkey_counter_deltas(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> dict[str, dict[str, dict[str, int]]]:
    counters = (
        "total_commands_processed",
        "rejected_connections",
        "total_error_replies",
    )
    return {
        group: {
            identity: {
                counter: int(after[group]["identities"][identity].get(counter, "0"))
                - int(before[group]["identities"][identity].get(counter, "0"))
                for counter in counters
            }
            for identity in GROUP_IDENTITIES[group]
        }
        for group in GROUP_IDENTITIES
    }


def wait_for_master_change(
    cluster: Cluster, master_name: str, previous: tuple[str, int]
) -> tuple[str, int]:
    deadline = time.monotonic() + 180
    last_error = "not queried"
    while time.monotonic() < deadline:
        try:
            current = sentinel_master(cluster, master_name)
            if current != previous:
                return current
            last_error = f"master remains {current}"
        except (RuntimeError, subprocess.SubprocessError, OSError) as error:
            last_error = str(error)
        time.sleep(1)
    raise TimeoutError(f"Sentinel did not fail over {master_name}: {last_error}")


def wait_for_roles(
    cluster: Cluster, expected_replica: str, expected_master: str
) -> dict[str, str]:
    deadline = time.monotonic() + 180
    last_roles: dict[str, str] = {}
    while time.monotonic() < deadline:
        try:
            replica_role = selected_valkey_info(cluster, expected_replica).get("role")
            master_role = selected_valkey_info(cluster, expected_master).get("role")
            last_roles = {
                expected_replica: replica_role or "unknown",
                expected_master: master_role or "unknown",
            }
            if replica_role == "slave" and master_role == "master":
                return last_roles
        except (RuntimeError, subprocess.SubprocessError, OSError):
            pass
        time.sleep(1)
    raise TimeoutError(
        f"restarted Valkey identities have incorrect roles: {last_roles}"
    )


def _write_canary(
    cluster: Cluster,
    master_name: str,
    address: tuple[str, int],
    replica: str,
) -> tuple[str, str, list[str]]:
    key = f"dynamo:ha-proof:{master_name}:{secrets.token_hex(8)}"
    value = secrets.token_hex(16)
    write = cluster.valkey(address[0], "SET", key, value)
    if write != ["OK"]:
        raise RuntimeError(
            f"failed to write {master_name} failover canary: SET={write}"
        )
    replica_value = wait_for_valkey_value(cluster, replica, ("GET", key), [value])
    return key, value, replica_value


def _tokenizer_dbsize(cluster: Cluster) -> int:
    host, _ = sentinel_master(cluster, "dynamo-tokenizer")
    response = cluster.valkey(host, "DBSIZE")
    if len(response) != 1:
        raise RuntimeError(f"invalid tokenizer DBSIZE response: {response}")
    return int(response[0])


def _wait_for_tokenizer_growth(cluster: Cluster, before: int) -> int:
    deadline = time.monotonic() + 30
    observed = before
    while time.monotonic() < deadline:
        observed = _tokenizer_dbsize(cluster)
        if observed > before:
            return observed
        time.sleep(0.25)
    raise TimeoutError(
        f"tokenizer L2 did not record application canary: before={before}, "
        f"observed={observed}"
    )


def dynamo_application_canary(
    cluster: Cluster,
    frontend_url: str,
    runtime_namespace: str,
    expected_mockers: int,
) -> dict[str, Any]:
    before = _tokenizer_dbsize(cluster)
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": f"HA reconnect canary {secrets.token_hex(16)}",
            }
        ],
        "max_tokens": 8,
        "min_tokens": 8,
        "ignore_eos": True,
        "temperature": 0.0,
    }
    request_script = """\
import json
import sys
import urllib.request

request = urllib.request.Request(
    sys.argv[1] + "/v1/chat/completions",
    data=sys.argv[2].encode(),
    headers={"content-type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=120) as response:
    body = json.load(response)
if not body.get("choices"):
    raise RuntimeError(f"application canary returned no choices: {body}")
print(json.dumps({"choices": len(body["choices"]), "model": body.get("model")}))
"""
    response = cluster.client_exec(
        (
            "/app/.venv/bin/python",
            "-c",
            request_script,
            frontend_url,
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
        ),
        timeout=180,
    )
    response_evidence = json.loads(response.stdout)
    after = _wait_for_tokenizer_growth(cluster, before)
    stats = registered_mocker_stats(cluster, runtime_namespace)
    if len(stats) != 3 or stats[1] != expected_mockers:
        raise RuntimeError(
            f"application canary lost router registrations: expected={expected_mockers}, "
            f"observed={stats}"
        )
    return {
        "frontend_url": frontend_url,
        "response": response_evidence,
        "registered_index_stats": stats,
        "tokenizer_dbsize_before": before,
        "tokenizer_dbsize_after": after,
    }


def prove_failover_and_restart(
    cluster: Cluster, application_probe: ApplicationProbe | None = None
) -> dict[str, Any]:
    proof: dict[str, Any] = {
        "status": "running",
        "scope": "storage-and-dynamo" if application_probe else "storage-only",
        "groups": {},
        "application_canaries": [],
    }
    if application_probe:
        proof["application_canaries"].append(
            {"phase": "before-failover", "evidence": application_probe()}
        )
    for master_name in GROUP_IDENTITIES:
        stopped, promotion_candidate, initial_infos, initial = elected_identity(
            cluster, master_name
        )
        key, value, initial_replica_value = _write_canary(
            cluster, master_name, initial, promotion_candidate
        )
        started = time.monotonic()
        cluster.kubectl(("scale", f"statefulset/{stopped}", "--replicas=0"))
        wait_for_no_pods(cluster, stopped)
        outage_application_evidence = None
        try:
            promoted = wait_for_master_change(cluster, master_name, initial)
            promoted_value = cluster.valkey(promoted[0], "GET", key)
            if promoted_value != [value]:
                raise RuntimeError(
                    f"{master_name} lost replicated canary during promotion: "
                    f"expected={value}, observed={promoted_value}"
                )
            if application_probe:
                outage_application_evidence = application_probe()
                proof["application_canaries"].append(
                    {
                        "phase": f"during-{master_name}-outage",
                        "evidence": outage_application_evidence,
                    }
                )
        finally:
            cluster.kubectl(("scale", f"statefulset/{stopped}", "--replicas=1"))
        cluster.kubectl(
            ("rollout", "status", f"statefulset/{stopped}", "--timeout=10m"),
            timeout=660,
        )
        roles = wait_for_roles(cluster, stopped, promotion_candidate)
        restarted_value = wait_for_valkey_value(cluster, stopped, ("GET", key), [value])
        delete = cluster.valkey(promoted[0], "DEL", key)
        if delete != ["1"]:
            raise RuntimeError(f"failed to delete {master_name} canary: DEL={delete}")
        replica_delete = wait_for_valkey_value(cluster, stopped, ("GET", key), [])
        proof["groups"][master_name] = {
            "initial_master": list(initial),
            "initial_roles": initial_infos,
            "stopped_identity": stopped,
            "promotion_candidate": promotion_candidate,
            "promoted_master": list(promoted),
            "initial_replica_canary": initial_replica_value,
            "promoted_canary": promoted_value,
            "outage_application_canary": outage_application_evidence,
            "restarted_canary": restarted_value,
            "roles_after_restart": roles,
            "replica_delete": replica_delete,
            "elapsed_seconds": time.monotonic() - started,
        }
        if application_probe:
            proof["application_canaries"].append(
                {
                    "phase": f"after-{master_name}-failover",
                    "evidence": application_probe(),
                }
            )

    masters_before = {name: sentinel_master(cluster, name) for name in GROUP_IDENTITIES}
    cluster.kubectl(("scale", "statefulset/valkey-sweep-sentinel", "--replicas=0"))
    wait_for_no_pods(cluster, "valkey-sweep-sentinel")
    try:
        cluster.kubectl(("scale", "statefulset/valkey-sweep-sentinel", "--replicas=3"))
    finally:
        if not pods(cluster, "app.kubernetes.io/name=valkey-sweep-sentinel"):
            cluster.kubectl(
                ("scale", "statefulset/valkey-sweep-sentinel", "--replicas=3")
            )
    cluster.kubectl(
        (
            "rollout",
            "status",
            "statefulset/valkey-sweep-sentinel",
            "--timeout=10m",
        ),
        timeout=660,
    )
    wait_for_ha(cluster)
    masters_after = {name: sentinel_master(cluster, name) for name in GROUP_IDENTITIES}
    if masters_after != masters_before:
        raise RuntimeError(
            f"Sentinel restart lost elected masters: before={masters_before}, "
            f"after={masters_after}"
        )
    proof["sentinel_restart"] = {
        "status": "passed",
        "masters": {name: list(master) for name, master in masters_after.items()},
    }
    if application_probe:
        proof["application_canaries"].append(
            {"phase": "after-sentinel-restart", "evidence": application_probe()}
        )
    proof["final_ha_snapshot"] = ha_snapshot(cluster)
    proof["status"] = "passed"
    return proof
