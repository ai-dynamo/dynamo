# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture the TP2 smoke engine only after sleep and both exact artifacts exist."""

import json
import os
import subprocess
import time
from pathlib import Path

KUBECONFIG = os.environ["GMS_VCLUSTER_KUBECONFIG"]
OUT = Path(os.environ.get("GMS_RESULTS", "results"))
OUT.mkdir(parents=True, exist_ok=True)
NS = "schwinns-vcluster"
POD = os.environ.get("GMS_SOURCE_POD", "gms-v1-qwen-source-0928")
SNAPSHOT = os.environ.get("GMS_SNAPSHOT", "gms-v1-qwen-0928")
RANKS = int(os.environ.get("GMS_RANKS", "2"))


def vk(*args):
    return subprocess.check_output(
        ["kubectl", "--kubeconfig=" + KUBECONFIG, "-n", NS, *args], text=True
    )


def hostpod():
    data = json.loads(
        subprocess.check_output(
            ["kubectl", "-n", NS, "get", "pods", "-o", "json"], text=True
        )
    )
    return next(
        p["metadata"]["name"]
        for p in data["items"]
        if p["metadata"].get("annotations", {}).get("vcluster.loft.sh/object-name")
        == POD
    )


host = hostpod()
end = time.monotonic() + 1800
while time.monotonic() < end:
    pod = json.loads(vk("get", "pod", POD, "-o", "json"))
    states = pod.get("status", {}).get("containerStatuses", [])
    if any(s.get("state", {}).get("terminated") for s in states):
        raise RuntimeError("source container exited")
    ready = any(
        c["type"] == "Ready" and c["status"] == "True"
        for c in pod.get("status", {}).get("conditions", [])
    )
    if ready:
        break
    time.sleep(5)
else:
    raise TimeoutError("source ready")
(OUT / "source-pod.json").write_text(json.dumps(pod, indent=2))
for container in ["main"] + [f"gms-{r}" for r in range(RANKS)]:
    (OUT / f"source-{container}.txt").write_text(
        subprocess.check_output(
            ["kubectl", "-n", NS, "logs", host, "-c", container], text=True
        )
    )
snapshot = {
    "apiVersion": "nvidia.com/v1alpha1",
    "kind": "PodSnapshot",
    "metadata": {"name": SNAPSHOT, "namespace": NS},
    "spec": {
        "source": {
            "podRef": {
                "name": POD,
                "uid": pod["metadata"]["uid"],
                "containers": ["main"],
            }
        }
    },
}
subprocess.run(
    ["kubectl", "--kubeconfig=" + KUBECONFIG, "apply", "-f", "-"],
    input=json.dumps(snapshot),
    text=True,
    check=True,
)
started = time.monotonic()
while time.monotonic() - started < 600:
    snap = json.loads(vk("get", "podsnapshot", SNAPSHOT, "-o", "json"))
    (OUT / "snapshot.json").write_text(json.dumps(snap, indent=2))
    conditions = snap.get("status", {}).get("conditions", [])
    if snap.get("status", {}).get("phase") == "Ready" or any(
        c["type"] == "Ready" and c["status"] == "True" for c in conditions
    ):
        print(
            json.dumps(
                {"capture_wall_s": time.monotonic() - started, "status": snap["status"]}
            ),
            flush=True,
        )
        break
    if snap.get("status", {}).get("phase") == "Failed":
        raise RuntimeError(snap["status"])
    time.sleep(2)
else:
    raise TimeoutError("checkpoint completion")
