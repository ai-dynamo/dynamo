# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Generic independent GMS services; no selected snapshot at service startup."""

import copy
import json
from pathlib import Path

from resolve_plan import resolve_destinations


def render(root, source, _capture, claim, slices, generation, host_path):
    # This fixture supplies image/resources and one named DRA request per rank.
    # Artifact selection belongs to POST /load after a DGD exists.
    original = next(item for item in source["items"] if item["kind"] == "Pod")["spec"]
    destinations = resolve_destinations(
        {
            "capture_id": "unbound-service",
            "layout": {"tp": 8},
            "ranks": [{"rank": r} for r in range(8)],
        },
        claim,
        slices,
    )
    name = "gms-v1-pagebroker-0929"
    cm_name = name + "-" + generation[:8]
    cm = {
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": cm_name},
        "immutable": True,
        "data": {
            script: (Path(root) / script).read_text()
            for script in [
                "pagebroker_server.py",
                "pagebroker_coordinator.py",
                "pagebroker_pb2.py",
                "resident_coordinator.py",
                "resolve_plan.py",
                "verify_publication.py",
            ]
        },
    }
    cm["data"].update(
        {"claim.json": json.dumps(claim), "slices.json": json.dumps(slices)}
    )
    by_name = {container["name"]: container for container in original["containers"]}
    containers = []
    for rank in range(8):
        container = copy.deepcopy(by_name[f"gms-{rank}"])
        claims = container["resources"]["claims"]
        if len(claims) != 1 or claims[0]["request"] != f"tp-{rank}":
            raise ValueError("GMS service must consume its named one-GPU DRA request")
        container["command"] = [
            "python3",
            "-u",
            "/snapshot-app/pagebroker_server.py",
            "--rank",
            str(rank),
            "--numa",
            "--service-generation",
            generation,
            "--expected-uuid",
            destinations[f"tp-{rank}"],
        ]
        container["resources"]["limits"] = {"cpu": "8", "memory": "16Gi"}
        container["volumeMounts"] = [
            m for m in container["volumeMounts"] if m["name"] != "artifacts"
        ]
        container["readinessProbe"] = {
            "exec": {"command": ["test", "-f", f"/gms/online-{rank}.json"]},
            "periodSeconds": 1,
        }
        containers.append(container)
    coordinator = copy.deepcopy(containers[0])
    coordinator.update(
        name="coordinator",
        command=[
            "python3",
            "-u",
            "/snapshot-app/pagebroker_coordinator.py",
            "--service-generation",
            generation,
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
    coordinator["volumeMounts"].extend(
        [
            {"name": "pagebroker", "mountPath": "/pagebroker"},
            {"name": "artifacts", "mountPath": "/checkpoints", "readOnly": True},
        ]
    )
    containers.append(coordinator)
    spec = {
        key: copy.deepcopy(original[key])
        for key in [
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
            {
                "name": "pagebroker",
                "hostPath": {
                    "path": "/var/lib/schwinns-pb-gms-0929/control",
                    "type": "Directory",
                },
            },
        ],
    )
    ds = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": name},
        "spec": {
            "selector": {"matchLabels": {"app": name}},
            "template": {"metadata": {"labels": {"app": name}}, "spec": spec},
        },
    }
    return cm, ds
