# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify rank publications, trigger existing staged pod, collect restore outcome."""

import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).parent
out = root / "results/qwen"
kc = [
    "kubectl",
    "--kubeconfig=" + os.environ["GMS_VCLUSTER_KUBECONFIG"],
    "-n",
    "schwinns-vcluster",
]
name = "gms-v1-qwen-restore-0928"
plan = json.loads((out / "restore-plan.json").read_text())
pods = json.loads(
    subprocess.check_output(
        ["kubectl", "-n", "schwinns-vcluster", "get", "pods", "-o", "json"]
    )
)["items"]
host = next(
    p["metadata"]["name"]
    for p in pods
    if p["metadata"].get("annotations", {}).get("vcluster.loft.sh/object-name") == name
)
base = ["kubectl", "-n", "schwinns-vcluster"]
for src, dst in [
    (root / "verify_publication.py", "/gms/verify_publication.py"),
    (out / "restore-plan.json", "/gms/restore-plan.json"),
]:
    subprocess.run(base + ["cp", str(src), host + ":" + dst, "-c", "gms-0"], check=True)
# Plan must still describe this claim and actual placeholder environment.
pod = json.loads(subprocess.check_output(kc + ["get", "pod", name, "-o", "json"]))
claim = json.loads(
    subprocess.check_output(
        kc + ["get", "resourceclaim", "gms-v1-target-0928", "-o", "json"]
    )
)
assert claim["metadata"]["uid"] == plan["claim_uid"]
assert (
    next(
        e["value"]
        for e in pod["spec"]["containers"][0]["env"]
        if e["name"] == "SNAPSHOT_CUDA_DEVICE_MAP"
    )
    == plan["cuda_device_map"]
)
subprocess.run(
    base
    + [
        "exec",
        host,
        "-c",
        "gms-0",
        "--",
        "python3",
        "/gms/verify_publication.py",
        "/gms/restore-plan.json",
    ],
    check=True,
)
started = time.time()
subprocess.run(
    kc + ["annotate", "pod", name, "nvidia.com/restore-from=gms-v1-qwen-0928"],
    check=True,
)
while time.time() - started < 180:
    pod = json.loads(subprocess.check_output(kc + ["get", "pod", name, "-o", "json"]))
    conditions = {c["type"]: c for c in pod.get("status", {}).get("conditions", [])}
    if conditions.get("nvidia.com/Restored", {}).get("reason") == "RestoreFailed":
        (out / "restore-failed-pod.json").write_text(json.dumps(pod, indent=2))
        raise RuntimeError(conditions["nvidia.com/Restored"])
    if conditions.get("Ready", {}).get("status") == "True":
        result = {
            "trigger_to_ready_s": time.time() - started,
            "pod": name,
            "conditions": conditions,
        }
        (out / "restore-success.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result), flush=True)
        print(
            subprocess.check_output(
                base
                + [
                    "exec",
                    host,
                    "-c",
                    "main",
                    "--",
                    "cat",
                    "/snapshot-control/sglang-restore-ready",
                ],
                text=True,
            ),
            flush=True,
        )
        break
    time.sleep(1)
else:
    raise TimeoutError("restored engine readiness")
