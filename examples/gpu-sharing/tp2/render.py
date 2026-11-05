#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Render Kubernetes JSON for the bounded two-GPU example; never apply it."""

import argparse
import copy
import json
import os
import re

REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"
IMAGE = "nvcr.io/nvidia/ai-dynamo/vllm-runtime@sha256:5913d0f4002c9f90f61ea9872b37719828851511af5a3fabb0ce56d0d0805f72"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=["dgds", "policy"])
    parser.add_argument("--workers", type=int, choices=[1, 2], default=1)
    parser.add_argument("--memory-mib", type=int, default=19456)
    parser.add_argument("--utilization", type=float, default=0.8)
    parser.add_argument(
        "--idle", action="store_true", help="Hold workers before CUDA initialization"
    )
    args = parser.parse_args()
    if not 0 < args.utilization < 1 or not 0 < args.memory_mib <= 20480:
        parser.error(
            "Require 0 < utilization < 1 and 0 < memory-mib <= 20480 on this A100 example"
        )
    if args.workers == 2 and args.memory_mib > 19456:
        parser.error(
            "Two-worker examples require <= 19456 MiB each to retain memory headroom"
        )

    namespace = os.environ.get("NAMESPACE", "tp2-gpu-fractions-example")
    node = os.environ["NODE_NAME"]
    cache = os.environ.get("HF_CACHE_DIR", "/opt/hf-cache")
    platform = os.environ.get("PLATFORM_NAMESPACE", "dynamo-system")
    image = os.environ.get("RUNTIME_IMAGE", IMAGE)
    model = f"{cache}/hub/models--Qwen--Qwen3-4B/snapshots/{REVISION}"
    if not cache.startswith("/") or not node:
        parser.error("Require an absolute HF_CACHE_DIR and a nonempty NODE_NAME")

    if args.kind == "policy":
        devices = [os.environ["GPU0_UUID"], os.environ["GPU1_UUID"]]
        pattern = r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}"
        if len(set(devices)) != 2 or not all(re.fullmatch(pattern, d) for d in devices):
            parser.error(
                "GPU0_UUID and GPU1_UUID must be distinct canonical physical GPU UUIDs"
            )
        rules = []
        for worker, chunks in [("a", [18, 36]), ("b", [36, 18])]:
            rules.append(
                {
                    "namespace": namespace,
                    "container": "main",
                    "matchLabels": {
                        "experiment.nvidia.com/dgd": f"tp2-{worker}",
                        "experiment.nvidia.com/role": "worker",
                    },
                    "partitions": [
                        {
                            "device": device,
                            "name": f"{namespace.replace('-', '_')}_{worker}_gpu{i}",
                            "chunks": chunks[i],
                        }
                        for i, device in enumerate(devices)
                    ],
                }
            )
        result = {
            "apiVersion": "v1",
            "kind": "ConfigMap",
            "metadata": {
                "name": f"{namespace}-static-v1",
                "namespace": "gpu-fractioning",
            },
            "immutable": True,
            "data": {"policy.json": json.dumps({"version": 1, "rules": rules})},
        }
    else:
        items = []
        for worker in ["a", "b"][: args.workers]:
            name = f"tp2-{worker}"
            env = [
                {"name": k, "value": v}
                for k, v in {
                    "DYN_NAMESPACE": f"{namespace}-{name}",
                    "DYN_DISCOVERY_BACKEND": "etcd",
                    "HF_HOME": cache,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "ETCD_ENDPOINTS": f"http://dynamo-platform-etcd.{platform}.svc.cluster.local:2379",
                    "NATS_SERVER": f"nats://dynamo-platform-nats.{platform}.svc.cluster.local:4222",
                }.items()
            ]
            components = []
            for role in ["frontend", "worker"]:
                container = {
                    "name": "main",
                    "image": image,
                    "imagePullPolicy": "IfNotPresent",
                    "command": [
                        "python3",
                        "-m",
                        f"dynamo.{role if role == 'frontend' else 'vllm'}",
                    ],
                    "args": ["--http-port", "8000"]
                    if role == "frontend"
                    else [
                        "--model",
                        model,
                        "--served-model-name",
                        "Qwen/Qwen3-4B",
                        "--tensor-parallel-size",
                        "2",
                        "--gpu-memory-utilization",
                        str(args.utilization),
                        "--max-model-len",
                        "4096",
                        "--max-num-seqs",
                        "32",
                    ],
                    "env": copy.deepcopy(env),
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                    },
                    "resources": {
                        "requests": {"cpu": "2", "memory": "12Gi"},
                        "limits": {"cpu": "8", "memory": "32Gi"},
                    }
                    if role == "worker"
                    else {
                        "requests": {"cpu": "100m", "memory": "1Gi"},
                        "limits": {"cpu": "2", "memory": "4Gi"},
                    },
                    "volumeMounts": [
                        {"name": "model-cache", "mountPath": cache, "readOnly": True}
                    ],
                }
                metadata = {
                    "labels": {
                        "experiment.nvidia.com/dgd": name,
                        "experiment.nvidia.com/role": role,
                    }
                }
                spec = {
                    "nodeSelector": {"kubernetes.io/hostname": node},
                    "runtimeClassName": "nvidia",
                    "securityContext": {
                        "runAsNonRoot": True,
                        "runAsUser": 1000,
                        "runAsGroup": 1000,
                        "seccompProfile": {"type": "RuntimeDefault"},
                    },
                    "containers": [container],
                    "volumes": [
                        {
                            "name": "model-cache",
                            "hostPath": {"path": cache, "type": "Directory"},
                        }
                    ],
                }
                if role == "worker":
                    metadata["labels"]["kai.scheduler/queue"] = "default-queue"
                    metadata["annotations"] = {
                        "gpu-fraction-num-devices": "2",
                        "nvidia.com/container.main.gpu-compute.mode": "sm-sharing",
                        "nvidia.com/container.main.gpu-compute.request": "0.5",
                        "nvidia.com/container.main.gpu-memory.request": f"{args.memory_mib}Mi",
                        "nvidia.com/container.main.gpu-memory.limit": f"{args.memory_mib}Mi",
                    }
                    spec.update(
                        {
                            "schedulerName": "kai-scheduler",
                            "terminationGracePeriodSeconds": 180,
                        }
                    )
                    spec["volumes"].append(
                        {
                            "name": "shm",
                            "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"},
                        }
                    )
                    container["volumeMounts"].append(
                        {"name": "shm", "mountPath": "/dev/shm"}
                    )
                    container["env"].extend(
                        [
                            {"name": "DYN_NAMESPACE_WORKER_SUFFIX", "value": ""},
                            {"name": "NCCL_DEBUG", "value": "INFO"},
                        ]
                    )
                    if args.idle:
                        container["command"] = ["sleep", "infinity"]
                        container["args"] = []
                components.append(
                    {
                        "name": "Frontend" if role == "frontend" else "worker",
                        "type": role,
                        "replicas": 1,
                        "runtimeVersionOverride": "1.3.0",
                        "podTemplate": {"metadata": metadata, "spec": spec},
                    }
                )
            items.append(
                {
                    "apiVersion": "nvidia.com/v1beta1",
                    "kind": "DynamoGraphDeployment",
                    "metadata": {
                        "name": name,
                        "namespace": namespace,
                        "annotations": {
                            "experiment.nvidia.com/model-revision": REVISION
                        },
                    },
                    "spec": {"backendFramework": "vllm", "components": components},
                }
            )
        result = {"apiVersion": "v1", "kind": "List", "items": items}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
