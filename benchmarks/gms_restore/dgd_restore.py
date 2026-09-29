# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the captured Engine API prototype through the Dynamo operator.

The existing capture predates Dynamo's native checkpoint compatibility contract.
This deliberately uses the experimental Snapshot annotation on an operator-owned
workload; it does not claim native checkpointRef or Dynamo runtime registration.
"""

import copy
from datetime import datetime

DGD_NAME = "gms-v1-glm-dgd-0929"
TRIAL_LABEL = "nvidia.com/gms-restore-trial"


def build_dgd(pod, name=DGD_NAME):
    """Keep rank wiring intact while supplying the custom app's health contract."""
    template = copy.deepcopy(pod)
    template["metadata"] = {
        key: template["metadata"][key]
        for key in ("annotations", "labels")
        if key in template["metadata"]
    }
    template["metadata"].setdefault("labels", {})[TRIAL_LABEL] = name
    template["spec"]["restartPolicy"] = "Always"
    main = next(c for c in template["spec"]["containers"] if c["name"] == "main")
    # Worker defaults probe Dynamo's system API, absent from this retained capture.
    main["startupProbe"] = copy.deepcopy(main["readinessProbe"])
    main["startupProbe"].update(timeoutSeconds=1, failureThreshold=300)
    main["livenessProbe"] = copy.deepcopy(main["readinessProbe"])
    main["livenessProbe"].update(periodSeconds=5, timeoutSeconds=1, failureThreshold=3)
    return {
        "apiVersion": "nvidia.com/v1beta1",
        "kind": "DynamoGraphDeployment",
        "metadata": {"name": name, "namespace": pod["metadata"]["namespace"]},
        "spec": {
            "backendFramework": "sglang",
            "components": [
                {
                    "name": "Engine",
                    "type": "worker",
                    "replicas": 1,
                    "sharedMemorySize": "32Gi",
                    "runtimeVersionOverride": "1.5.0",
                    "podTemplate": template,
                }
            ],
        },
    }


def timestamp_epoch(value):
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return value.timestamp()


def object_identity(obj):
    """Record only ownership/timing metadata, never service-account annotations."""
    metadata = obj["metadata"]
    return {
        "apiVersion": obj.get("apiVersion"),
        "kind": obj.get("kind"),
        "name": metadata["name"],
        "namespace": metadata.get("namespace"),
        "uid": metadata["uid"],
        "creationTimestamp": metadata["creationTimestamp"],
        "creation_epoch": timestamp_epoch(metadata["creationTimestamp"]),
        "ownerReferences": metadata.get("ownerReferences", []),
    }


def ownership_chain(pod, api, apps, custom):
    """Validate every UID in Pod → ReplicaSet → Deployment → DCD → DGD."""
    current = api.sanitize_for_serialization(pod)
    current.update(apiVersion="v1", kind="Pod")
    chain = [object_identity(current)]
    for kind in (
        "ReplicaSet",
        "Deployment",
        "DynamoComponentDeployment",
        "DynamoGraphDeployment",
    ):
        owner = next(
            ref
            for ref in current["metadata"].get("ownerReferences", [])
            if ref.get("controller") and ref["kind"] == kind
        )
        namespace = current["metadata"]["namespace"]
        if kind == "ReplicaSet":
            current = api.sanitize_for_serialization(
                apps.read_namespaced_replica_set(owner["name"], namespace)
            )
        elif kind == "Deployment":
            current = api.sanitize_for_serialization(
                apps.read_namespaced_deployment(owner["name"], namespace)
            )
        else:
            group, version = owner["apiVersion"].split("/", 1)
            current = custom.get_namespaced_custom_object(
                group, version, namespace, kind.lower() + "s", owner["name"]
            )
        assert current["metadata"]["uid"] == owner["uid"], "owner incarnation changed"
        current.setdefault("kind", kind)
        current.setdefault("apiVersion", owner["apiVersion"])
        chain.append(object_identity(current))
    assert chain[-1]["name"] == DGD_NAME
    return chain
