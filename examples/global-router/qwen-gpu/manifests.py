# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Render real GPU workers using the existing isolated Dynamo pool support."""

import argparse
import importlib.util
import json
from pathlib import Path

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
RUNTIME = (
    "nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:"
    "99b42c3586442132f9873e356d1f411fc6045e60cd7d280671b88de28a819805"
)


def render(namespace, site_id, frontend_image, relay_image, runtime_image=RUNTIME):
    source = Path(__file__).resolve().parents[1] / "rr-backend" / "manifests.py"
    spec = importlib.util.spec_from_file_location("isolated_pool", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = module.render(
        namespace, "qwen-gpu", site_id, frontend_image, relay_image, MODEL, {}
    )
    for item in manifest["items"]:
        if item["kind"] != "Deployment":
            continue
        item["spec"]["template"]["spec"]["nodeSelector"] = {
            "dynamo.nvidia.com/purpose": "gpu"
        }
        if item["metadata"]["name"] != "qwen-gpu-worker":
            continue
        item["spec"]["strategy"] = {
            "type": "RollingUpdate",
            "rollingUpdate": {"maxSurge": 0, "maxUnavailable": 1},
        }
        pod = item["spec"]["template"]["spec"]
        pod["nodeSelector"] = {"dynamo.nvidia.com/purpose": "gpu"}
        pod["securityContext"] = {
            "runAsUser": 1000,
            "runAsGroup": 1000,
            "fsGroup": 1000,
        }
        pod["volumes"] = [
            {
                "name": "model-cache",
                "hostPath": {
                    "path": "/var/lib/dynamo/model-cache/qwen3-4b-instruct-2507",
                    "type": "DirectoryOrCreate",
                },
            },
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "2Gi"}},
        ]
        pod["initContainers"] = [
            {
                "name": "cache-owner",
                "image": runtime_image,
                "command": [
                    "python3",
                    "-c",
                    "import os; os.chown('/model-cache',1000,1000)",
                ],
                "securityContext": {
                    "runAsUser": 0,
                    "runAsGroup": 0,
                    "allowPrivilegeEscalation": False,
                },
                "volumeMounts": [{"name": "model-cache", "mountPath": "/model-cache"}],
                "resources": {
                    "requests": {"cpu": "50m", "memory": "64Mi"},
                    "limits": {"memory": "256Mi"},
                },
            }
        ]
        worker = pod["containers"][0]
        worker["name"] = "vllm"
        worker["image"] = runtime_image
        worker["command"] = ["python3", "-m", "dynamo.vllm"]
        worker["args"] = [
            "--model",
            MODEL,
            "--revision",
            REVISION,
            "--served-model-name",
            MODEL,
            "--enforce-eager",
            "--max-model-len",
            "4096",
            "--max-num-seqs",
            "8",
            "--gpu-memory-utilization",
            "0.75",
            "--dtype",
            "half",
            "--enable-prefix-caching",
        ]
        worker["env"] += [
            {"name": "HF_HOME", "value": "/model-cache"},
            {"name": "HF_HUB_DISABLE_TELEMETRY", "value": "1"},
        ]
        worker["resources"] = {
            "requests": {
                "cpu": "2",
                "memory": "12Gi",
                "ephemeral-storage": "2Gi",
                "nvidia.com/gpu": "1",
            },
            "limits": {"memory": "24Gi", "nvidia.com/gpu": "1"},
        }
        worker["startupProbe"]["failureThreshold"] = 180
        worker["startupProbe"]["periodSeconds"] = 10
        worker["volumeMounts"] = [
            {"name": "model-cache", "mountPath": "/model-cache"},
            {"name": "shm", "mountPath": "/dev/shm"},
        ]
        worker["securityContext"] = {
            "allowPrivilegeEscalation": False,
            "capabilities": {"drop": ["ALL"]},
        }

    def labels(value):
        if isinstance(value, dict):
            for key, entry in value.items():
                if key == "app.kubernetes.io/part-of":
                    value[key] = "global-router-qwen"
                else:
                    labels(entry)
        elif isinstance(value, list):
            for entry in value:
                labels(entry)

    labels(manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="global-router-qwen")
    parser.add_argument("--site-id", required=True)
    parser.add_argument("--frontend-image", required=True)
    parser.add_argument("--relay-image", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            render(args.namespace, args.site_id, args.frontend_image, args.relay_image),
            indent=2,
        )
    )
