# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a single claim shared by one all-GPU and eight isolated containers."""

import argparse
import json

p = argparse.ArgumentParser()
p.add_argument("--node", required=True)
p.add_argument("--image", required=True)
p.add_argument("--namespace", required=True)
p.add_argument("--name", default="gms-restore-probe-0928")
a = p.parse_args()
meta = {"name": a.name, "namespace": a.namespace, "labels": {"experiment": a.name}}
claim = {
    "apiVersion": "resource.k8s.io/v1",
    "kind": "ResourceClaim",
    "metadata": meta,
    "spec": {
        "devices": {
            "requests": [
                {
                    "name": f"tp-{r}",
                    "exactly": {"deviceClassName": "gpu.nvidia.com", "count": 1},
                }
                for r in range(8)
            ]
        }
    },
}
containers = []
for name, request in [("all", None), *[(f"rank-{r}", f"tp-{r}") for r in range(8)]]:
    ref = {"name": "gpus"}
    if request:
        ref["request"] = request
    containers.append(
        {
            "name": name,
            "image": a.image,
            "imagePullPolicy": "IfNotPresent",
            "command": ["sleep", "infinity"],
            "resources": {"requests": {"cpu": "1", "memory": "2Gi"}, "claims": [ref]},
            "env": [
                {"name": "DYN_GMS_USE_V1", "value": "true"},
                {"name": "GMS_SOCKET_DIR", "value": "/sockets/" + name},
            ],
            "volumeMounts": [
                {"name": "work", "mountPath": "/work"},
                {"name": "sockets", "mountPath": "/sockets"},
                {"name": "models", "mountPath": "/model-cache"},
                {"name": "artifacts", "mountPath": "/checkpoints"},
            ],
        }
    )
pod = {
    "apiVersion": "v1",
    "kind": "Pod",
    "metadata": meta,
    "spec": {
        "nodeSelector": {"kubernetes.io/hostname": a.node},
        "tolerations": [{"operator": "Exists", "effect": "NoSchedule"}],
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": "nvcr-secret"}, {"name": "acr-token-secret"}],
        "resourceClaims": [{"name": "gpus", "resourceClaimName": a.name}],
        "containers": containers,
        "volumes": [
            {"name": "work", "emptyDir": {}},
            {"name": "sockets", "emptyDir": {}},
            {
                "name": "models",
                "persistentVolumeClaim": {"claimName": "shared-model-cache"},
            },
            {
                "name": "artifacts",
                "persistentVolumeClaim": {
                    "claimName": "snapshot-pvc-x-schwinns-vcluster-x-schwinns-vcluster"
                },
            },
        ],
    },
}
print(json.dumps({"apiVersion": "v1", "kind": "List", "items": [claim, pod]}, indent=2))
