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

"""Apply namespaced manifests through an existing private CDK kubectl provider.

Fleet input is operator-local JSON with a ``regions`` list. Each member supplies
region, clusterName, kubectlProviderServiceToken, and kubectlProviderHandlerArn.
No IAM roles, VPCs, EC2 instances, operators, or CRDs are created by this helper.
"""

import argparse
import json
import subprocess
from pathlib import Path


def aws(region, *args):
    result = subprocess.run(
        ["aws", "--region", region, *args, "--output", "json", "--no-cli-pager"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    if result.returncode:
        raise RuntimeError(result.stderr)
    return json.loads(result.stdout) if result.stdout.strip() else {}


def template(member, manifest):
    items = manifest["items"]
    namespaces = [item for item in items if item["kind"] == "Namespace"]
    objects = [item for item in items if item["kind"] != "Namespace"]
    resources = {}

    def resource(values):
        return {
            "Type": "Custom::AWSCDK-EKS-KubernetesResource",
            "Properties": {
                "ServiceToken": member["kubectlProviderServiceToken"],
                "ClusterName": member["clusterName"],
                "Manifest": json.dumps(values),
                "Overwrite": True,
            },
        }

    if namespaces:
        # The namespace is shared with separately deployed ingress/discovery.
        # Removing this stack must never cascade-delete those components.
        resources["SharedNamespace"] = {
            **resource(namespaces),
            "DeletionPolicy": "Retain",
            "UpdateReplacePolicy": "Retain",
        }
    if objects:
        resources["Objects"] = resource(objects)
        if namespaces:
            resources["Objects"]["DependsOn"] = "SharedNamespace"
    return {"AWSTemplateFormatVersion": "2010-09-09", "Resources": resources}


def deploy(member, stack, manifest, evidence):
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    path = evidence / f"{member['region']}-{stack}-template.json"
    path.write_text(json.dumps(template(member, manifest), indent=2) + "\n")
    existing = aws(
        member["region"],
        "cloudformation",
        "list-stacks",
        "--query",
        "StackSummaries[?StackStatus!='DELETE_COMPLETE'].StackName",
    )
    action = "update-stack" if stack in existing else "create-stack"
    try:
        value = aws(
            member["region"],
            "cloudformation",
            action,
            "--stack-name",
            stack,
            "--template-body",
            "file://" + str(path.resolve()),
        )
    except RuntimeError as error:
        if action == "update-stack" and "No updates are to be performed" in str(error):
            return {"unchanged": True, "stack": stack}
        raise
    (evidence / f"{member['region']}-{stack}-submit.json").write_text(
        json.dumps(value, indent=2) + "\n"
    )
    return value


def read(member, kind, namespace, name, jsonpath, evidence):
    evidence = Path(evidence)
    evidence.mkdir(parents=True, exist_ok=True)
    prefix = f"{member['region']}-{namespace}-{kind.replace('.', '_')}"
    event = {
        "ResourceType": "Custom::AWSCDK-EKS-KubernetesObjectValue",
        "RequestType": "Create",
        "ResourceProperties": {
            "ClusterName": member["clusterName"],
            "ObjectType": kind,
            "ObjectName": name or "--ignore-not-found=true",
            "ObjectNamespace": namespace,
            "JsonPath": jsonpath,
            "TimeoutSeconds": 15,
        },
    }
    request, response = (
        evidence / f"{prefix}-request.json",
        evidence / f"{prefix}-response.json",
    )
    request.write_text(json.dumps(event))
    aws(
        member["region"],
        "lambda",
        "invoke",
        "--function-name",
        member["kubectlProviderHandlerArn"],
        "--cli-binary-format",
        "raw-in-base64-out",
        "--payload",
        "file://" + str(request.resolve()),
        str(response.resolve()),
    )
    value = json.loads(response.read_text())
    if "errorMessage" in value:
        raise RuntimeError(value["errorMessage"])
    text = value["Data"]["Value"]
    try:
        return json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fleet", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    commands = parser.add_subparsers(dest="command", required=True)
    apply = commands.add_parser("deploy")
    apply.add_argument("--stack", required=True)
    apply.add_argument("--manifest", type=Path, required=True)
    get = commands.add_parser("read")
    get.add_argument("--kind", required=True)
    get.add_argument("--namespace", default="global-router-v2-rr")
    get.add_argument("--name", default="")
    get.add_argument("--jsonpath", default=".items")
    args = parser.parse_args()
    member = next(
        m
        for m in json.loads(args.fleet.read_text())["regions"]
        if m["region"] == args.region
    )
    if args.command == "deploy":
        value = deploy(
            member, args.stack, json.loads(args.manifest.read_text()), args.evidence
        )
    else:
        value = read(
            member, args.kind, args.namespace, args.name, args.jsonpath, args.evidence
        )
    print(json.dumps(value, indent=2))


if __name__ == "__main__":
    main()
