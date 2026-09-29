# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster trial: publish all exact rank artifacts, then trigger engine restore.

Requires a captured source, staged prototype agent, and GMS_VCLUSTER_KUBECONFIG.
The host kubectl context must reach translated vcluster pods. Never changes agents.
"""

import argparse
import copy
import json
import os
import subprocess
import time
from pathlib import Path

from kubernetes import client, config

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--backend", choices=["fused", "nixl"], default="fused")
a = p.parse_args()
root = Path(__file__).parent
out = root / "results/glm" / a.case
out.mkdir(parents=True, exist_ok=True)
config.load_kube_config(os.environ["GMS_VCLUSTER_KUBECONFIG"])
api = client.ApiClient()
core = client.CoreV1Api()
custom = client.CustomObjectsApi()
ns = "schwinns-vcluster"
name = "gms-v1-glm-restore-0928"
claimname = "gms-v1-glm-target-0928"
base = ["kubectl", "-n", ns]


def host_name():
    pods = json.loads(subprocess.check_output(base + ["get", "pods", "-o", "json"]))[
        "items"
    ]
    return next(
        p["metadata"]["name"]
        for p in pods
        if p["metadata"].get("annotations", {}).get("vcluster.loft.sh/object-name")
        == name
    )


def remote(cmd, container="gms-0"):
    return subprocess.check_output(
        base + ["exec", host, "-c", container, "--", *cmd], text=True
    )


def copy_file(path, target, container="gms-0"):
    subprocess.run(
        base + ["cp", str(path), host + ":" + target, "-c", container], check=True
    )


def save(path, data):
    path.write_text(json.dumps(api.sanitize_for_serialization(data), indent=2))


source = json.loads((root / "results/glm/source-manifest-nccl.json").read_text())
capture = json.loads((root / "results/glm/capture.json").read_text())
targets = list(
    reversed(
        json.loads((root / "results/cuinit-all.jsonl").read_text().splitlines()[0])[
            "uuids"
        ]
    )
)
claim = copy.deepcopy(source["items"][0])
claim["metadata"]["name"] = claimname
for r, request in enumerate(claim["spec"]["devices"]["requests"]):
    request["exactly"]["selectors"] = [
        {
            "cel": {
                "expression": f'device.attributes["gpu.nvidia.com"].uuid == "{targets[r]}"'
            }
        }
    ]
try:
    custom.create_namespaced_custom_object(
        "resource.k8s.io", "v1", ns, "resourceclaims", claim
    )
except client.ApiException as e:
    if e.status != 409:
        raise
cm = copy.deepcopy(source["items"][1])
cm["metadata"]["name"] = "gms-v1-glm-restore-app-0928"
cm["data"].update(
    {
        f: (root / f).read_text()
        for f in [
            "fast_server.py",
            "posix_direct.py",
            "verify_publication.py",
            "resolve_plan.py",
        ]
    }
)
try:
    core.create_namespaced_config_map(ns, cm)
except client.ApiException as e:
    if e.status != 409:
        raise
    core.patch_namespaced_config_map(cm["metadata"]["name"], ns, {"data": cm["data"]})
pod = copy.deepcopy(source["items"][2])
pod["metadata"]["name"] = name
pod["spec"]["nodeSelector"] = {
    "kubernetes.io/hostname": "cluster-0967a26d-pool-14bee067-prctr-s2877"
}
pod["spec"]["resourceClaims"][0]["resourceClaimName"] = claimname
for v in pod["spec"]["volumes"]:
    if v["name"] == "app":
        v["configMap"]["name"] = cm["metadata"]["name"]
main = pod["spec"]["containers"][0]
main["command"] = ["sleep", "infinity"]
main.pop("args", None)
main["readinessProbe"] = {
    "exec": {"command": ["test", "-f", "/snapshot-control/sglang-restore-ready"]},
    "periodSeconds": 1,
}
map_value = ",".join(
    r["source_uuid"] + "=" + targets[r["rank"]] for r in capture["ranks"]
)
main["env"].append({"name": "SNAPSHOT_CUDA_DEVICE_MAP", "value": map_value})
for rank, c in enumerate(pod["spec"]["containers"][1:]):
    c["command"] = [
        "python3",
        "-u",
        "/snapshot-app/"
        + ("fast_server.py" if a.backend == "fused" else "rank_server.py"),
        "--rank",
        str(rank),
        "--artifact-root",
        "/checkpoints/gms-restore-0928/glm-capture-1",
    ]
    if a.backend == "nixl":
        c["command"] += ["--mode", "load"]
    c["readinessProbe"] = {
        "exec": {"command": ["test", "-f", f"/gms/published-{rank}"]},
        "periodSeconds": 1,
    }
    c["resources"]["limits"] = {"cpu": "8", "memory": "16Gi"}
save(out / "manifest.json", pod)
started = time.time()
core.create_namespaced_pod(ns, pod)
end = time.monotonic() + 300
while time.monotonic() < end:
    current = core.read_namespaced_pod(name, ns)
    states = current.status.container_statuses or []
    if any(s.state.terminated for s in states):
        raise RuntimeError("container exited before publication")
    if sum(s.ready for s in states if s.name.startswith("gms-")) == 8:
        break
    time.sleep(0.5)
else:
    raise TimeoutError("GMS publication")
publication_observed = time.time()
host = host_name()
allocated = custom.get_namespaced_custom_object(
    "resource.k8s.io", "v1", ns, "resourceclaims", claimname
)
slices = json.loads(
    subprocess.check_output(["kubectl", "get", "resourceslices", "-o", "json"])
)
save(out / "claim.json", allocated)
save(out / "slices.json", slices)
for path, dst in [
    (out / "claim.json", "/gms/claim.json"),
    (out / "slices.json", "/gms/slices.json"),
    (root / "results/glm/capture.json", "/gms/capture.json"),
]:
    copy_file(path, dst)
plan = json.loads(
    remote(
        [
            "python3",
            "/snapshot-app/resolve_plan.py",
            "--capture",
            "/gms/capture.json",
            "--claim",
            "/gms/claim.json",
            "--slices",
            "/gms/slices.json",
        ]
    )
)
assert plan["cuda_device_map"] == map_value
save(out / "plan.json", plan)
copy_file(out / "plan.json", "/gms/plan.json")
verified = remote(["python3", "/snapshot-app/verify_publication.py", "/gms/plan.json"])
(out / "publication.txt").write_text(verified)
records = json.loads(
    remote(
        [
            "python3",
            "-c",
            'import json;from pathlib import Path;print(json.dumps([json.loads(Path(f"/gms/rank-{r}.json").read_text()) for r in range(8)]))',
        ]
    )
)
save(out / "publications.json", records)
triggered = time.time()
core.patch_namespaced_pod(
    name,
    ns,
    {
        "metadata": {
            "annotations": {"nvidia.com/gms-prototype-restore-from": "gms-v1-glm-0928"}
        }
    },
)
end = time.monotonic() + 240
while time.monotonic() < end:
    current = core.read_namespaced_pod(name, ns)
    conditions = {x.type: x for x in current.status.conditions or []}
    if (
        conditions.get("nvidia.com/Restored")
        and conditions["nvidia.com/Restored"].reason == "RestoreFailed"
    ):
        save(out / "failed-pod.json", current)
        raise RuntimeError(conditions["nvidia.com/Restored"].message)
    if conditions.get("Ready") and conditions["Ready"].status == "True":
        break
    time.sleep(0.5)
else:
    save(out / "timeout-pod.json", current)
    raise TimeoutError("engine readiness")
ready = time.time()
text = remote(["cat", "/snapshot-control/sglang-restore-ready"], "main")
assert "berlin" in text.lower(), text
result = {
    "backend": a.backend,
    "create_epoch": started,
    "publication_observed_epoch": publication_observed,
    "trigger_epoch": triggered,
    "ready_epoch": ready,
    "pod_create_to_ready_s": ready - started,
    "pod_create_to_publication_observed_s": publication_observed - started,
    "verification_orchestration_gap_s": triggered - publication_observed,
    "trigger_to_ready_s": ready - triggered,
    "restored_text": text,
}
save(out / "timing.json", result)
print(json.dumps(result), flush=True)
for c in ["gms-" + str(r) for r in range(8)]:
    (out / (c + ".txt")).write_text(
        subprocess.check_output(base + ["logs", host, "-c", c], text=True)
    )
# A second response exercises the restored engine beyond the readiness prompt.
response = remote(
    [
        "python3",
        "-c",
        'import urllib.request;print(urllib.request.urlopen(urllib.request.Request("http://localhost:8000/generate",data=b\'{"prompt":"Explain in two sentences why the sky is blue."}\',headers={"Content-Type":"application/json"}),timeout=90).read().decode())',
    ],
    "main",
)
(out / "inference.json").write_text(response)
