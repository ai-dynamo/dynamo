# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure contracts for observing fixed-capacity Kubernetes rollouts."""

import logging
import subprocess
import xml.etree.ElementTree as ET

from tests.deploy.n2.rolling_http import check


def command(*args, timeout=600):
    try:
        result = subprocess.run(
            args, text=True, capture_output=True, timeout=timeout, check=True
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        logging.getLogger(__name__).error(
            "Command %r failed: %s; stdout=%r; stderr=%r",
            args,
            error,
            error.stdout,
            error.stderr,
        )
        raise
    if result.stderr:
        logging.getLogger(__name__).warning(
            "Command %r stderr: %s", args, result.stderr
        )
    return result.stdout


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
        "gpus": (
            sum(
                int(c.get("resources", {}).get("limits", {}).get("nvidia.com/gpu", 0))
                for c in spec["containers"]
            )
            if active
            else 0
        ),
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


class PodTracker:
    def __init__(self, pods, old_uids, component, image):
        self.pods = {pod["uid"]: pod for pod in pods}
        self.old_uids = old_uids
        self.component = component
        self.image = image
        self.ready_overlap_seen = False

    def apply(self, event_type, pod):
        if event_type == "DELETED":
            self.pods.pop(pod["uid"], None)
        else:
            self.pods[pod["uid"]] = pod
        current = list(self.pods.values())
        check_budget(current)
        ready = [p for p in current if p["component"] == self.component and p["ready"]]
        self.ready_overlap_seen |= any(
            p["uid"] in self.old_uids for p in ready
        ) and any(
            p["uid"] not in self.old_uids and p["image"] == self.image for p in ready
        )


class Cleanup:
    def __init__(self):
        self.errors = []

    async def attempt(self, label, operation):
        try:
            return await operation()
        except Exception as error:
            # Cleanup must continue after arbitrary API/log parsing failures.
            # BaseException is deliberately not caught so cancellation propagates.
            diagnostic = RuntimeError(f"{label}: {error}")
            diagnostic.__cause__ = error
            self.errors.append(diagnostic)

    def finish(self, primary_failure):
        if self.errors and not primary_failure:
            raise ExceptionGroup("Rolling-upgrade cleanup failed", self.errors)


def require_executed_cases(paths):
    cases = [
        case
        for path in paths
        for case in ET.parse(path).iter("testcase")
        if case.get("name", "").startswith("test_n2_rolling_upgrade[")
    ]
    check(len(cases) == 8, f"Expected 8 rolling-upgrade cases, found {len(cases)}")
    check(
        len({case.get("name") for case in cases}) == 8,
        "Duplicate rolling-upgrade cases",
    )
    check(
        all(
            not any(
                case.find(tag) is not None for tag in ("skipped", "failure", "error")
            )
            for case in cases
        ),
        "All 8 rolling-upgrade cases must execute and pass",
    )
