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

"""Render one dynamo-batch Helm values file for a DGD; never deploy resources."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml


def required_text(values: dict, key: str) -> str:
    value = values[key]
    if not isinstance(value, str) or not value.strip() or value == "REPLACE_ME":
        raise ValueError(f"{key} must be a non-empty configured string")
    return value


def load_values(name: str) -> dict:
    with (Path(__file__).parent / name).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def render_values(profile: dict) -> dict[str, dict]:
    for values, allowed in (
        (profile, {"targetDGD", "model", "asyncConcurrency", "storage"}),
        (profile["targetDGD"], {"name", "namespace"}),
        (
            profile["storage"],
            {"mode", "secretName", "s3"},
        ),
        (
            profile["storage"].get("s3", {}),
            {"region", "bucket", "accessKeyId", "endpoint", "usePathStyle", "prefix"},
        ),
    ):
        unknown = values.keys() - allowed
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
    target = profile["targetDGD"]
    name = required_text(target, "name")
    namespace = required_text(target, "namespace")
    for field, value in (("name", name), ("namespace", namespace)):
        if len(value) > 63 or not re.fullmatch(
            r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", value
        ):
            raise ValueError(
                f"targetDGD.{field} must be a DNS label of at most 63 characters"
            )
    model = required_text(profile, "model")
    if len(f"{name}-frontend") > 63:
        raise ValueError(
            "targetDGD.name makes the frontend Service name exceed 63 characters"
        )
    if len(f"{name}-batch-async") > 63:
        raise ValueError(
            "targetDGD.name makes integration resource names exceed 63 characters"
        )
    storage = profile["storage"]
    if storage.get("mode", "external") != "external":
        raise ValueError("storage.mode must be external; provision storage separately")
    secret = required_text(storage, "secretName")
    if len(secret) > 253 or not re.fullmatch(
        r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?(?:\.[a-z0-9](?:[-a-z0-9]*[a-z0-9])?)*",
        secret,
    ):
        raise ValueError("secretName must be a Kubernetes Secret name")
    s3 = storage["s3"]
    for field in ("region", "bucket", "accessKeyId"):
        required_text(s3, field)
    if "prefix" in s3:
        required_text(s3, "prefix")
    if not isinstance(s3.get("endpoint", ""), str):
        raise ValueError("endpoint must be a string")
    if type(s3.get("usePathStyle", False)) is not bool:
        raise ValueError("usePathStyle must be a boolean")
    workers = profile.get("asyncConcurrency", 64)
    if type(workers) is not int or workers < 1:
        raise ValueError("asyncConcurrency must be a positive integer")

    frontend = f"http://{name}-frontend.{namespace}.svc:8000"
    prefix = f"dynamo-batch:{namespace}:{name}"
    request_queue = f"{prefix}:requests"
    result_queue = f"{prefix}:results"

    parent = load_values("chart/values.yaml")
    batch = parent["batch-gateway"]
    batch["global"]["secretName"] = secret
    file_config = batch["global"]["fileClient"]["s3"]
    for key in ("region", "bucket", "accessKeyId"):
        file_config[key] = s3[key]
    file_config["endpoint"] = s3.get("endpoint", "")
    file_config["usePathStyle"] = s3.get("usePathStyle", False)
    file_config["prefix"] = s3.get("prefix", f"{namespace}/{name}")
    processor = batch["processor"]["config"]
    concurrency = processor["concurrency"]
    concurrency["global"] = max(concurrency["global"], workers)
    concurrency["perEndpoint"] = max(concurrency["perEndpoint"], workers)
    processor["asyncDispatch"]["models"] = {
        model: {
            "inferencePoolName": f"{namespace}-{name}",
            "requestQueueName": request_queue,
            "resultQueueName": result_queue,
        }
    }

    async_values = parent["async"]
    ap = async_values["ap"]
    ap["concurrency"] = workers
    ap["transportConfig"]["urlSecret"]["name"] = secret
    ap["transportConfig"]["batch_size"] = workers
    ap["transportConfig"]["result_queue_name"] = result_queue
    queue = ap["transportConfig"]["queues"][0]
    queue["queue_name"] = request_queue
    queue["igw_base_url"] = frontend
    queue["gate_params"]["url"] = f"{frontend}/metrics"
    queue["gate_params"]["labels"]["model"] = model

    # Helm does not transform parent values into independent dependency values.
    # Derive the shared contract here, while reusing the published workload charts.
    batch["apiserver"]["fullnameOverride"] = f"{name}-batch-api"
    async_values["fullnameOverride"] = f"{name}-batch-async"
    parent["targetDGD"] = {"name": name, "namespace": namespace}
    parent["model"] = model
    return {"dynamo-batch-values.yaml": parent}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    with args.profile.open(encoding="utf-8") as source:
        values = render_values(yaml.safe_load(source))
    # Require a new directory so a rerender cannot overwrite a reviewed profile.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for filename, config in values.items():
        with (args.output_dir / filename).open("x", encoding="utf-8") as destination:
            yaml.safe_dump(config, destination, sort_keys=False)
    print(f"Rendered dynamo-batch-values.yaml in {args.output_dir}.")


if __name__ == "__main__":
    main()
