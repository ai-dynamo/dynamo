# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render a TP2 restore with deliberately reversed destination physical order."""

import copy
import json
from pathlib import Path

root = Path(__file__).parent
source = json.loads((root / "results/source-manifest.json").read_text())
capture = json.loads((root / "results/qwen/capture.json").read_text())
# Explicitly select known compatible B200s; resolve_plan verifies actual DRA results.
probe = json.loads((root / "results/cuinit-all.jsonl").read_text().splitlines()[0])
targets = list(reversed(probe["uuids"][:2]))
claim = copy.deepcopy(source["items"][0])
claim["metadata"]["name"] = "gms-v1-target-0928"
for r, request in enumerate(claim["spec"]["devices"]["requests"]):
    request["exactly"]["selectors"] = [
        {
            "cel": {
                "expression": f'device.attributes["gpu.nvidia.com"].uuid == "{targets[r]}"'
            }
        }
    ]
pod = copy.deepcopy(source["items"][2])
pod["metadata"]["name"] = "gms-v1-qwen-restore-0928"
pod["metadata"]["labels"] = {"experiment": "gms-v1-0928"}
pod["spec"]["nodeSelector"] = {
    "kubernetes.io/hostname": "cluster-0967a26d-pool-14bee067-prctr-s2877"
}
pod["spec"]["resourceClaims"][0]["resourceClaimName"] = claim["metadata"]["name"]
main = pod["spec"]["containers"][0]
main["command"] = ["/bin/sh", "-c", "exec sleep infinity"]
main.pop("args", None)
main["readinessProbe"] = {
    "exec": {"command": ["test", "-f", "/snapshot-control/sglang-restore-ready"]},
    "periodSeconds": 1,
}
value = ",".join(r["source_uuid"] + "=" + targets[r["rank"]] for r in capture["ranks"])
main["env"].append({"name": "SNAPSHOT_CUDA_DEVICE_MAP", "value": value})
for c in pod["spec"]["containers"][1:]:
    c["command"] += ["--mode", "load"]
(root / "results/qwen/restore-manifest.json").write_text(
    json.dumps({"apiVersion": "v1", "kind": "List", "items": [claim, pod]}, indent=2)
)
