# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render one experiment-only DaemonSet with eight named DRA rank containers."""

import copy
import json
from pathlib import Path

from resolve_plan import resolve_destinations


def render(root, source, capture, claim, slices, generation, host_path):
    name = "gms-v1-resident-0929"
    cm_name = name + "-" + generation[:8]
    source_pods = [item for item in source["items"] if item["kind"] == "Pod"]
    if len(source_pods) != 1:
        raise ValueError("resident prototype requires exactly one source Pod")
    original = source_pods[0]["spec"]
    if capture["layout"].get("tp") != 8 or any(
        capture["layout"].get(dimension, 1) != 1 for dimension in ("pp", "dp")
    ):
        raise ValueError("resident prototype supports only TP8, PP1, DP1")
    ranks = {rank["rank"]: rank for rank in capture["ranks"]}
    if len(capture["ranks"]) != 8 or set(ranks) != set(range(8)):
        raise ValueError(
            "resident prototype requires each rank 0 through 7 exactly once"
        )
    artifact_roots = set()
    for rank, record in ranks.items():
        if record["socket_device"] != rank or record["socket_dir"] != "/gms":
            raise ValueError(
                "resident prototype requires socket_device=rank under /gms"
            )
        artifact = Path(record["artifact"])
        if (
            not artifact.is_absolute()
            or ".." in artifact.parts
            or artifact.name != f"device-{rank}"
        ):
            raise ValueError(
                "resident prototype requires absolute device-N artifact directories"
            )
        artifact_roots.add(str(artifact.parent))
    if len(artifact_roots) != 1:
        raise ValueError("resident prototype requires one common rank artifact root")
    artifact_root = next(iter(artifact_roots))
    by_name = {container["name"]: container for container in original["containers"]}
    if len(by_name) != len(original["containers"]) or not all(
        f"gms-{rank}" in by_name for rank in ranks
    ):
        raise ValueError("source Pod must have unique gms-N containers for each rank")
    destinations = resolve_destinations(capture, claim, slices)
    mapping = ",".join(
        ranks[rank]["source_uuid"] + "=" + destinations[f"tp-{rank}"]
        for rank in sorted(ranks)
    )
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": cm_name},
        "immutable": True,
        "data": {
            f: (Path(root) / f).read_text()
            for f in [
                "resident_server.py",
                "resident_coordinator.py",
                "posix_direct.py",
                "resolve_plan.py",
                "verify_publication.py",
            ]
        },
    }
    cm["data"].update(
        {
            "capture.json": json.dumps(capture),
            "claim.json": json.dumps(claim),
            "slices.json": json.dumps(slices),
        }
    )
    containers = []
    for rank in range(8):
        container = copy.deepcopy(by_name[f"gms-{rank}"])
        claims = container["resources"].get("claims", [])
        if len(claims) != 1 or claims[0].get("request") != f"tp-{rank}":
            raise ValueError(
                f"source gms-{rank} must consume only its tp-{rank} request"
            )
        container["command"] = [
            "python3",
            "-u",
            "/snapshot-app/resident_server.py",
            "--rank",
            str(rank),
            "--workers",
            "16",
            "--chunk-mib",
            "128",
            "--numa",
            "--artifact-root",
            artifact_root,
            "--capture-id",
            capture["capture_id"],
            "--generation",
            generation,
            "--expected-uuid",
            destinations[f"tp-{rank}"],
        ]
        container["resources"]["limits"] = {"cpu": "8", "memory": "16Gi"}
        container["readinessProbe"] = {
            "exec": {"command": ["test", "-f", f"/gms/online-{rank}.json"]},
            "periodSeconds": 1,
        }
        for mount in container["volumeMounts"]:
            if mount["mountPath"] == "/checkpoints":
                mount["readOnly"] = True
        containers.append(container)
    coordinator = copy.deepcopy(containers[0])
    coordinator.update(
        name="coordinator",
        command=[
            "python3",
            "-u",
            "/snapshot-app/resident_coordinator.py",
            "--capture-id",
            capture["capture_id"],
            "--generation",
            generation,
            "--workers",
            "16",
            "--chunk-mib",
            "128",
        ],
        resources={
            "requests": {"cpu": "100m", "memory": "128Mi"},
            "limits": {"cpu": "1", "memory": "512Mi"},
        },
        readinessProbe={
            "httpGet": {"path": "/healthz", "port": 18081},
            "periodSeconds": 1,
        },
    )
    coordinator["env"].append({"name": "SNAPSHOT_CUDA_DEVICE_MAP", "value": mapping})
    containers.append(coordinator)
    spec = {
        k: copy.deepcopy(original[k])
        for k in [
            "nodeSelector",
            "securityContext",
            "imagePullSecrets",
            "tolerations",
            "runtimeClassName",
            "resourceClaims",
        ]
    }
    spec.update(
        restartPolicy="Always",
        terminationGracePeriodSeconds=5,
        containers=containers,
        volumes=[
            {"name": "app", "configMap": {"name": cm_name}},
            {
                "name": "gms",
                "hostPath": {"path": host_path, "type": "DirectoryOrCreate"},
            },
            {
                "name": "artifacts",
                "hostPath": {
                    "path": "/var/lib/schwinns-gms-0928/gms-pvc-nfs",
                    "type": "Directory",
                },
            },
        ],
    )
    labels = {"app": name}
    ds = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": name},
        "spec": {
            "selector": {"matchLabels": labels},
            "template": {"metadata": {"labels": labels}, "spec": spec},
        },
    }
    return cm, ds
