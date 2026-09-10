# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure contracts for observing fixed-capacity Kubernetes rollouts."""

from scripts.compatibility.runner import check


def pod_record(pod):
    metadata, spec, status = pod["metadata"], pod["spec"], pod.get("status", {})
    main = next(c for c in spec["containers"] if c["name"] == "main")
    containers = status.get("containerStatuses", [])
    main_status = next((c for c in containers if c["name"] == "main"), {})
    # Include terminating Pods until Kubernetes reports a terminal phase.
    active = status.get("phase") not in ("Succeeded", "Failed")
    return {
        "uid": metadata["uid"],
        "name": metadata["name"],
        "component": metadata["labels"]["nvidia.com/dynamo-component"],
        "image": main["image"],
        "image_id": main_status.get("imageID", ""),
        "ip": status.get("podIP"),
        "active": active,
        "terminating": bool(metadata.get("deletionTimestamp")),
        "ready": active
        and not metadata.get("deletionTimestamp")
        and any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in status.get("conditions", [])
        ),
        "gpus": sum(
            int(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
            for c in spec["containers"]
        )
        if active
        else 0,
        "restarts": sum(c.get("restartCount", 0) for c in containers),
    }


def check_budget(pods):
    check(
        sum(p["gpus"] for p in pods) <= 2,
        "More than two GPU allocations, including terminating Pods",
    )
    check(all(p["restarts"] == 0 for p in pods), "Container restarted during rollout")


def converged(pods, component, image, old_uids):
    active = [p for p in pods if p["active"] and p["component"] == component]
    return len(active) == 2 and all(
        p["ready"]
        and p["image"] == image
        and p["image_id"]
        and p["uid"] not in old_uids
        for p in active
    )


def check_requests(summary, phases):
    check(summary["failures"] == 0, f"HTTP contract failures: {summary['failures']}")
    for phase in phases:
        counts = summary["phases"].get(phase, {})
        check(
            len(counts) == 3 and min(counts.values()) >= 2,
            f"Insufficient request coverage in {phase}: {counts}",
        )


def request_count(metrics):
    """Dynamo ingress counter, excluding management endpoints."""
    import re

    values = []
    for line in metrics.splitlines():
        if line.startswith("dynamo_component_requests_total{") and re.search(
            r'dynamo_endpoint="generate"', line
        ):
            values.append(float(line.rsplit(" ", 1)[1]))
    check(bool(values), "Missing generate request counter")
    return sum(values)
