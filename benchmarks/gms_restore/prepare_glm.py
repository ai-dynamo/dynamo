# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render the GLM-5.2 TP8 evidence workload with one GMS V1 owner per GPU."""

import copy
import json
from pathlib import Path

root = Path(__file__).parent
small = json.loads((root / "results/source-manifest.json").read_text())
pod = copy.deepcopy(small["items"][2])
pod["metadata"]["name"] = "gms-v1-glm-source-0928"
pod["spec"]["resourceClaims"][0]["resourceClaimName"] = "gms-v1-glm-source-0928"
for v in pod["spec"]["volumes"]:
    if v["name"] == "app":
        v["configMap"]["name"] = "gms-v1-glm-app-0928"
main = pod["spec"]["containers"][0]
main["resources"]["limits"] = {"cpu": "96", "memory": "1536Gi"}
main["resources"]["requests"] = {"cpu": "32", "memory": "128Gi"}
for env in main["env"]:
    if env["name"] == "SNAPSHOT_MODEL":
        env["value"] = "nvidia/GLM-5.2-NVFP4"
main["env"].append({"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"})
server = pod["spec"]["containers"][1]
pod["spec"]["containers"] = [main]
for rank in range(8):
    c = copy.deepcopy(server)
    c["name"] = f"gms-{rank}"
    c["command"] = [
        "python3",
        "-u",
        "/snapshot-app/rank_server.py",
        "--rank",
        str(rank),
        "--artifact-root",
        "/checkpoints/gms-restore-0928/glm-capture-1",
    ]
    c["resources"]["claims"][0]["request"] = f"tp-{rank}"
    pod["spec"]["containers"].append(c)
app = small["items"][1]["data"]["app.py"]
app = (
    app.replace(
        "tp_size=2, mem_fraction_static=0.15,",
        "tp_size=8, ep_size=8, mem_fraction_static=0.70,",
    )
    .replace("disable_cuda_graph=True, disable_custom_all_reduce=True,", "")
    .replace("range(2)", "range(8)")
)
# SGLang's custom allreduce parses integer CVD; NVML discovery avoids cuInit.
app = app.replace(
    "os.environ['CUDA_VISIBLE_DEVICES']=','.join(rank_ids)",
    """import subprocess
    rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,pci.bus_id','--format=csv,noheader'],text=True)
    ordered=[u.strip() for u,b in sorted((line.split(',') for line in rows.splitlines()),key=lambda x:x[1].strip())]
    os.environ['CUDA_VISIBLE_DEVICES']=','.join(str(ordered.index(u)) for u in rank_ids)""",
)
cm = copy.deepcopy(small["items"][1])
cm["metadata"]["name"] = "gms-v1-glm-app-0928"
cm["data"]["app.py"] = app
claim = copy.deepcopy(small["items"][0])
claim["metadata"]["name"] = "gms-v1-glm-source-0928"
claim["spec"]["devices"]["requests"] = [
    {"name": f"tp-{r}", "exactly": {"deviceClassName": "gpu.nvidia.com", "count": 1}}
    for r in range(8)
]
(root / "results/glm").mkdir(exist_ok=True)
(root / "results/glm/source-manifest.json").write_text(
    json.dumps(
        {"apiVersion": "v1", "kind": "List", "items": [claim, cm, pod]}, indent=2
    )
)
