# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cluster trial: restore exact rank artifacts and engine, optionally overlapping.

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
from resolve_plan import resolve_destinations

p = argparse.ArgumentParser()
p.add_argument("--case", required=True)
p.add_argument("--storage", choices=["pvc", "tmpfs"], default="pvc")
p.add_argument("--qualified-pvc-mount", action="store_true")
p.add_argument("--isolated-pvc-transport", action="store_true")
p.add_argument("--workers", type=int, default=4)
p.add_argument("--chunk-mib", type=int, default=16)
p.add_argument("--gms-cpu-request", type=int, default=1)
p.add_argument("--gms-cpu-limit", type=int, default=8)
p.add_argument("--numa", action="store_true")
p.add_argument("--capture-dir", type=Path)
p.add_argument("--same-claim", action="store_true")
p.add_argument("--fast-gate", action="store_true")
p.add_argument("--overlap", action="store_true")
p.add_argument("--early-trigger", action="store_true")
p.add_argument("--backend", choices=["fused", "nixl"], default="fused")
a = p.parse_args()
if not 0 < a.gms_cpu_request <= a.gms_cpu_limit:
    p.error("GMS CPU request must be positive and no greater than its limit")
if a.chunk_mib < 1:
    p.error("--chunk-mib must be positive")
if a.backend != "fused" and a.chunk_mib != 16:
    p.error("--chunk-mib applies to the fused backend")
if a.overlap and not a.fast_gate:
    p.error("--overlap requires --fast-gate")
if a.qualified_pvc_mount and a.storage != "pvc":
    p.error("--qualified-pvc-mount requires PVC storage")
if a.isolated_pvc_transport and not a.qualified_pvc_mount:
    p.error("--isolated-pvc-transport requires --qualified-pvc-mount")
if a.early_trigger and not (a.overlap and a.fast_gate):
    p.error("--early-trigger requires --overlap --fast-gate")
root = Path(__file__).parent
capture_dir = a.capture_dir or root / "results/glm"
out = capture_dir / a.case
out.mkdir(parents=True, exist_ok=True)
config.load_kube_config(os.environ["GMS_VCLUSTER_KUBECONFIG"])
api = client.ApiClient()
core = client.CoreV1Api()
custom = client.CustomObjectsApi()
ns = "schwinns-vcluster"
name = "gms-v1-glm-restore-0928"
claimname = "gms-v1-glm-source-0928" if a.same_claim else "gms-v1-glm-target-0928"
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


source = json.loads((capture_dir / "source-manifest-nccl.json").read_text())
capture = json.loads((capture_dir / "capture.json").read_text())
if a.storage == "pvc":
    assert all(r["artifact"].startswith("/checkpoints/") for r in capture["ranks"]), (
        "PVC trial requires PVC artifact paths"
    )
targets = (
    [r["source_uuid"] for r in capture["ranks"]]
    if a.same_claim
    else list(
        reversed(
            json.loads((root / "results/cuinit-all.jsonl").read_text().splitlines()[0])[
                "uuids"
            ]
        )
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
preallocated = custom.get_namespaced_custom_object(
    "resource.k8s.io", "v1", ns, "resourceclaims", claimname
)
pre_slices = json.loads(
    subprocess.check_output(["kubectl", "get", "resourceslices", "-o", "json"])
)
if a.fast_gate:
    assert preallocated.get("status", {}).get("allocation"), (
        "fast gate needs an allocated claim"
    )
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
            "publication_gate.py",
            "rank_server.py",
        ]
    }
)
if a.fast_gate:
    cm["data"].update(
        {
            "capture.json": json.dumps(capture),
            "claim.json": json.dumps(preallocated),
            "slices.json": json.dumps(pre_slices),
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
pod["spec"]["volumes"] = [
    v for v in pod["spec"]["volumes"] if v["name"] != "ram-weights"
]
if a.storage == "tmpfs":
    pod["spec"]["volumes"].append(
        {
            "name": "ram-weights",
            "hostPath": {
                "path": "/var/lib/schwinns-gms-0928/ram-weights",
                "type": "Directory",
            },
        }
    )
if a.qualified_pvc_mount:
    pod["spec"]["volumes"].append(
        {
            "name": "qualified-pvc",
            "hostPath": {
                "path": (
                    "/var/lib/schwinns-gms-0928/gms-pvc-nfs"
                    if a.isolated_pvc_transport
                    else "/var/lib/snapshot-restore-perf-nfs"
                ),
                "type": "Directory",
            },
        }
    )
main = pod["spec"]["containers"][0]
main["imagePullPolicy"] = "IfNotPresent"
main["command"] = (
    ["python3", "-u", "/snapshot-app/publication_gate.py"]
    if a.fast_gate and not a.overlap
    else ["sleep", "infinity"]
)
if a.fast_gate and a.storage == "tmpfs":
    main["volumeMounts"].append(
        {"name": "ram-weights", "mountPath": "/gms-artifacts", "readOnly": True}
    )
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
    c["imagePullPolicy"] = "IfNotPresent"
    c["env"] = [e for e in c["env"] if e["name"] != "GMS_PROTOTYPE_BUFFERED_READS"]
    if a.storage == "tmpfs":
        c["env"].append({"name": "GMS_PROTOTYPE_BUFFERED_READS", "value": "1"})
    c["volumeMounts"] = [v for v in c["volumeMounts"] if v["name"] != "ram-weights"]
    if a.qualified_pvc_mount:
        for mount in c["volumeMounts"]:
            if mount["mountPath"] == "/checkpoints":
                mount["name"] = "qualified-pvc"
                mount["readOnly"] = True
    if a.storage == "tmpfs":
        c["volumeMounts"].append(
            {"name": "ram-weights", "mountPath": "/gms-artifacts", "readOnly": True}
        )
    c["command"] = [
        "python3",
        "-u",
        "/snapshot-app/"
        + ("fast_server.py" if a.backend == "fused" else "rank_server.py"),
        "--rank",
        str(rank),
        "--artifact-root",
        str(Path(capture["ranks"][0]["artifact"]).parent),
    ]
    if a.backend == "fused":
        c["command"] += ["--workers", str(a.workers), "--chunk-mib", str(a.chunk_mib)]
        if a.numa:
            c["command"].append("--numa")
    if a.backend == "nixl":
        c["command"] += ["--mode", "load"]
    c["readinessProbe"] = {
        "exec": {"command": ["test", "-f", f"/gms/published-{rank}"]},
        "periodSeconds": 1,
    }
    c["resources"]["requests"]["cpu"] = str(a.gms_cpu_request)
    c["resources"]["limits"] = {"cpu": str(a.gms_cpu_limit), "memory": "16Gi"}
if a.fast_gate:
    pod["spec"]["containers"][1]["readinessProbe"]["exec"]["command"] = [
        "test",
        "-f",
        "/gms/all-ready",
    ]
save(out / "manifest.json", pod)
if a.overlap:
    assert "GMS_WAKE_GATE" in cm["data"]["app.py"], (
        "capture must include the wake-time publication gate"
    )
    gate = copy.deepcopy(main)
    gate["name"] = "publication-gate"
    gate["command"] = ["python3", "-u", "/snapshot-app/publication_gate.py"]
    gate.pop("ports", None)
    gate["resources"] = {
        "requests": {"cpu": "100m", "memory": "128Mi"},
        "limits": {"cpu": "1", "memory": "512Mi"},
    }
    gate["env"] = [
        e
        for e in gate["env"]
        if e["name"] in {"PYTHONPATH", "SNAPSHOT_CUDA_DEVICE_MAP", "GMS_SOCKET_DIR"}
    ]
    gate["volumeMounts"] = [
        v
        for v in gate["volumeMounts"]
        if v["name"] in {"app", "gms", "artifacts", "ram-weights"}
    ]
    gate["readinessProbe"] = {
        "exec": {"command": ["test", "-f", "/gms/all-ready"]},
        "periodSeconds": 1,
    }
    pod["spec"]["containers"].append(gate)
    save(out / "manifest.json", pod)
if a.early_trigger:
    # The held claim's allocation is immutable while reserved by the holder.
    # Revalidate before creating the pod; no GPU enumeration or remote exec is needed.
    allocated = custom.get_namespaced_custom_object(
        "resource.k8s.io", "v1", ns, "resourceclaims", claimname
    )
    assert allocated["metadata"]["uid"] == preallocated["metadata"]["uid"]
    assert allocated["status"]["allocation"] == preallocated["status"]["allocation"]
    destinations = resolve_destinations(capture, allocated, pre_slices)
    assert all(destinations[f"tp-{rank}"] == uuid for rank, uuid in enumerate(targets))
    save(out / "claim.json", allocated)
    save(out / "slices.json", pre_slices)
    pod["metadata"].setdefault("annotations", {})[
        "nvidia.com/gms-prototype-restore-from"
    ] = capture["capture_id"]
    save(out / "manifest.json", pod)
started = time.time()
core.create_namespaced_pod(ns, pod)
pod_create_returned = time.time()
if a.early_trigger:
    triggered = started
    publication_observed = None
else:
    end = time.monotonic() + 300
    while time.monotonic() < end:
        current = core.read_namespaced_pod(name, ns)
        states = current.status.container_statuses or []
        if any(s.state.terminated for s in states):
            raise RuntimeError("container exited before publication")
        if a.overlap:
            if len(states) == len(pod["spec"]["containers"]) and all(
                s.state.running for s in states
            ):
                break
        elif sum(s.ready for s in states if s.name.startswith("gms-")) == 8:
            break
        time.sleep(0.5)
    else:
        raise TimeoutError("GMS publication")
    publication_observed = None if a.overlap else time.time()
    host = host_name()
    if a.fast_gate:
        allocated = custom.get_namespaced_custom_object(
            "resource.k8s.io", "v1", ns, "resourceclaims", claimname
        )
        assert allocated["metadata"]["uid"] == preallocated["metadata"]["uid"]
        assert allocated["status"]["allocation"] == preallocated["status"]["allocation"]
        save(out / "claim.json", allocated)
        save(out / "slices.json", pre_slices)
    else:
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
            (capture_dir / "capture.json", "/gms/capture.json"),
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
        verified = remote(
            ["python3", "/snapshot-app/verify_publication.py", "/gms/plan.json"]
        )
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
                "annotations": {
                    "nvidia.com/gms-prototype-restore-from": capture["capture_id"]
                }
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
save(out / "ready-pod.json", current)
if a.early_trigger:
    host = host_name()
if a.fast_gate:
    plan = json.loads(remote(["cat", "/gms/restore-plan.json"]))
    save(out / "plan.json", plan)
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
    (out / "gate.txt").write_text(remote(["cat", "/gms/all-ready"]))
text = remote(["cat", "/snapshot-control/sglang-restore-ready"], "main")
assert "berlin" in text.lower(), text
result = {
    "backend": a.backend,
    "chunk_mib": a.chunk_mib if a.backend == "fused" else None,
    "gms_cpu_request": a.gms_cpu_request,
    "gms_cpu_limit": a.gms_cpu_limit,
    "early_trigger": a.early_trigger,
    "pod_create_return_epoch": pod_create_returned,
    "workers": a.workers if a.backend == "fused" else 16,
    "numa": a.numa,
    "qualified_pvc_mount": a.qualified_pvc_mount,
    "isolated_pvc_transport": a.isolated_pvc_transport,
    "fast_gate": a.fast_gate,
    "overlap": a.overlap,
    "capture_id": capture["capture_id"],
    "storage": "PVC O_DIRECT weights; PVC engine checkpoint"
    if a.storage == "pvc"
    else "tmpfs weights; PVC engine checkpoint",
    "create_epoch": started,
    "publication_observed_epoch": publication_observed,
    "trigger_epoch": triggered,
    "ready_epoch": ready,
    "pod_create_to_ready_s": ready - started,
    "pod_create_to_publication_observed_s": None
    if publication_observed is None
    else publication_observed - started,
    "verification_orchestration_gap_s": None
    if publication_observed is None
    else triggered - publication_observed,
    "trigger_to_ready_s": ready - triggered,
    "restored_text": text,
}
save(out / "timing.json", result)
print(json.dumps(result), flush=True)
mount = json.loads(
    remote(["findmnt", "-J", "-T", "/checkpoints", "-o", "SOURCE,FSTYPE,OPTIONS"])
)
save(out / "pvc-mount.json", mount)
if a.qualified_pvc_mount:
    observed = mount["filesystems"][0]
    assert observed["fstype"].startswith("nfs")
    assert "pvc-1df82048-23b9-47dc-91cd-2b2b7dff7e21" in observed["source"]
    assert "nconnect=32" in observed["options"]
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
