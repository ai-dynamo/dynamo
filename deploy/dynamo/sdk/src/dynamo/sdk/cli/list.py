#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import json
import os
import typing as t

import bentoml
import click
from bentoml_cli.utils import BentoMLCommandGroup
from kubernetes import client, config
from rich import print as rich_print
from rich.table import Table


def build_list_command() -> click.Group:
    @click.group(name="list", cls=BentoMLCommandGroup)
    def cli():
        """List resources"""
        pass

    @cli.command(name="bentos")
    def list_bentos():
        """List all bentos in local store"""
        bentos = bentoml.list()
        table = Table(box=None, expand=True)
        table.add_column("Tag", overflow="fold")
        table.add_column("Service", overflow="fold")
        table.add_column("Created At", overflow="fold")

        for bento in bentos:
            table.add_row(
                str(bento.tag),
                bento.info.service,
                bento.info.creation_time.strftime("%Y-%m-%d %H:%M:%S"),
            )
        rich_print(table)

    @cli.command(name="deployments")
    @click.option("--namespace", type=click.STRING, help="Kubernetes namespace")
    @click.option("--cluster", type=click.STRING, help="Cluster name")
    @click.option(
        "-o",
        "--output",
        help="Display the output of this command.",
        type=click.Choice(["json", "table"]),
        default="table",
    )
    def list_deployments(
        namespace: str | None = None,
        cluster: str | None = None,
        output: t.Literal["json", "table"] = "table",
    ):
        """List deployments"""
        config.load_kube_config()
        api = client.CustomObjectsApi()

        # Define the group, version, and plural for the CRD
        group = "nvidia.com"
        version = "v1alpha1"
        plural = "dynamodeployments"

        # Get the deployments from the Kubernetes API
        deployments = api.list_namespaced_custom_object(
            group=group,
            version=version,
            namespace=namespace,
            plural=plural,
        )

        if output == "json":
            rich_print(json.dumps(deployments, indent=2))
            return

        # Create table for output
        table = Table(box=None, expand=True)
        table.add_column("Name", overflow="fold")
        table.add_column("Namespace", overflow="fold")
        table.add_column("Status", overflow="fold")
        table.add_column("Created At", overflow="fold")
        table.add_column("Replicas", overflow="fold")
        table.add_column("Resources", overflow="fold")
        table.add_column("URL", overflow="fold")

        ingress_suffix = os.getenv("DYNAMO_INGRESS_SUFFIX", "local")

        for item in deployments.get("items", []):
            metadata = item.get("metadata", {})
            spec = item.get("spec", {})
            services = spec.get("services", {}).get("main", {}).get("spec", {})
            resources = services.get("resources", {})
            ingress = services.get("ingress", {})

            # Format resources
            resources_str = (
                f"CPU: {resources.get('requests', {}).get('cpu', 'N/A')} / {resources.get('limits', {}).get('cpu', 'N/A')}\n"
                f"Memory: {resources.get('requests', {}).get('memory', 'N/A')} / {resources.get('limits', {}).get('memory', 'N/A')}\n"
                f"GPU: {resources.get('requests', {}).get('gpu', 'N/A')} / {resources.get('limits', {}).get('gpu', 'N/A')}"
            )

            # Format URL
            url = (
                f"https://{ingress.get('hostPrefix', 'N/A')}.{ingress_suffix}"
                if ingress.get("enabled", False)
                else "N/A"
            )

            table.add_row(
                metadata.get("name", "N/A"),
                metadata.get("namespace", "N/A"),
                item.get("status", {}).get("state", "Unknown"),
                metadata.get("creationTimestamp", "N/A"),
                f"{services.get('autoscaling', {}).get('minReplicas', 'N/A')} - {services.get('autoscaling', {}).get('maxReplicas', 'N/A')}",
                resources_str,
                url,
            )

        rich_print(table)

    return cli


list_command = build_list_command()
