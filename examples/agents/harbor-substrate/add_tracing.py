# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add native request and TLS tool-event capture to a recipe's frontend."""

import argparse
from pathlib import Path

import yaml


def add_tracing(deployment: dict) -> dict:
    if deployment.get("kind") != "DynamoGraphDeployment":
        raise ValueError("Expected one DynamoGraphDeployment")
    components = deployment["spec"]["components"]
    frontend = next(c for c in components if c["name"] == "Frontend")
    pod = frontend["podTemplate"]["spec"]
    main = next(c for c in pod["containers"] if c["name"] == "main")
    values = {
        "DYN_REQUEST_TRACE": "1",
        "DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT": "tcp://127.0.0.1:20390",
        "DYN_REQUEST_TRACE_OUTPUT_PATH": "/traces/request-trace",
        "DYN_REQUEST_TRACE_FILE_FORMAT": "jsonl",
    }
    env = main.setdefault("env", [])
    for name, value in values.items():
        if any(e["name"] == name for e in env):
            raise ValueError(f"Existing trace setting: {name}; merge it explicitly")
        env.append({"name": name, "value": value})
    volumes = pod.setdefault("volumes", [])
    if any(
        v["name"] in {"harbor-traces", "tool-trace-tls", "tool-trace-config"}
        for v in volumes
    ):
        raise ValueError("Trace volume names already exist")
    if any(c["name"] == "tool-events-tls" for c in pod["containers"]):
        raise ValueError("Trace sidecar already exists")
    main.setdefault("volumeMounts", []).append(
        {"name": "harbor-traces", "mountPath": "/traces"}
    )
    volumes.extend(
        [
            {"name": "harbor-traces", "emptyDir": {"sizeLimit": "2Gi"}},
            {
                "name": "tool-trace-tls",
                "secret": {"secretName": "harbor-tool-trace-tls"},
            },
            {
                "name": "tool-trace-config",
                "configMap": {"name": "harbor-tool-trace-tls"},
            },
        ]
    )
    pod["containers"].append(
        {
            "name": "tool-events-tls",
            "image": "alpine:3.22",
            "command": [
                "sh",
                "-ec",
                "apk add --no-cache stunnel; exec stunnel /config/stunnel.conf",
            ],
            "ports": [{"name": "tls-zmq", "containerPort": 20391}],
            "volumeMounts": [
                {"name": "tool-trace-tls", "mountPath": "/tls", "readOnly": True},
                {"name": "tool-trace-config", "mountPath": "/config", "readOnly": True},
            ],
            "resources": {
                "requests": {"cpu": "50m", "memory": "32Mi"},
                "limits": {"cpu": "1", "memory": "128Mi"},
            },
        }
    )
    return deployment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    documents = list(yaml.safe_load_all(args.input.read_text()))
    deployments = [
        item
        for item in documents
        if isinstance(item, dict) and item.get("kind") == "DynamoGraphDeployment"
    ]
    if len(deployments) != 1:
        raise ValueError("Expected exactly one DynamoGraphDeployment in the recipe")
    add_tracing(deployments[0])
    args.output.write_text(yaml.safe_dump_all(documents, sort_keys=False))


if __name__ == "__main__":
    main()
