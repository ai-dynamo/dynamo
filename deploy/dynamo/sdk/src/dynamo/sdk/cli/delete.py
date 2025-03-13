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

import logging

import bentoml
import click
from bentoml._internal.cloud.base import Spinner
from bentoml_cli.utils import BentoMLCommandGroup
from kubernetes import client, config
from rich.console import Console

logger = logging.getLogger(__name__)


def build_delete_command() -> click.Group:
    @click.group(name="delete", cls=BentoMLCommandGroup)
    def cli():
        """Delete resources"""
        pass

    @cli.command(name="bentos")
    @click.argument("bento_tag", type=click.STRING, required=False)
    @click.option(
        "--all",
        is_flag=True,
        default=False,
        help="Delete all bentos in local store",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Skip confirmation prompt",
    )
    def delete_bentos(
        bento_tag: str | None = None,
        all: bool = False,
        force: bool = False,
    ):
        """
        Delete bentos from local store

        Args:
            bento_tag: Tag of the bento to delete
            all: Delete all bentos
            force: Skip confirmation prompt
        """
        console = Console(highlight=False)

        # Validate arguments
        if not bento_tag and not all:
            raise click.ClickException(
                "Either specify a bento tag or use --all to delete all bentos"
            )

        if bento_tag and all:
            raise click.ClickException("Cannot specify both a bento tag and --all flag")

        with Spinner(console=console) as spinner:
            try:
                # Get bentos to delete
                bentos_to_delete = []

                if all:
                    # Get all bentos
                    spinner.update("Fetching all bentos")
                    bentos = bentoml.list()
                    bentos_to_delete = [str(bento.tag) for bento in bentos]

                    if not bentos_to_delete:
                        spinner.log("No bentos found in local store")
                        return
                else:
                    # Check if the specified bento exists
                    if bento_tag is not None:
                        bentos_to_delete = [bento_tag]
                    else:
                        # This should never happen due to earlier validation, but handle it anyway
                        spinner.log("[bold red]No bento tag specified[/]")
                        return

                # Confirm deletion if not forced
                if not force:
                    spinner.stop()
                    if all:
                        message = f"Are you sure you want to delete all {len(bentos_to_delete)} bentos from local store?"
                    else:
                        message = f"Are you sure you want to delete bento '{bento_tag}' from local store?"

                    if not click.confirm(message):
                        console.print("[yellow]Deletion cancelled[/]")
                        return
                    spinner.start()

                # Delete bentos
                for tag in bentos_to_delete:
                    spinner.update(f"Deleting bento '{tag}'")
                    try:
                        bentoml.delete(tag)
                        spinner.log(f"[green]Successfully deleted bento '{tag}'[/]")
                    except Exception as e:
                        spinner.log(
                            f"[bold red]Failed to delete bento '{tag}': {str(e)}[/]"
                        )
                        logger.error(f"Failed to delete bento '{tag}'", exc_info=True)

                # Final summary
                if all:
                    spinner.log(
                        f"[bold green]Deleted {len(bentos_to_delete)} bentos from local store[/]"
                    )

            except Exception as e:
                logger.error("Deletion operation failed", exc_info=True)
                spinner.log(f"[bold red]Operation failed: {str(e)}[/]")
                raise SystemExit(1)

    @cli.command(name="deployments")
    @click.argument("deployment_name", type=click.STRING, required=False)
    @click.option(
        "--namespace",
        type=click.STRING,
        default="default",
        help="Kubernetes namespace containing the deployments",
    )
    @click.option(
        "--all",
        is_flag=True,
        default=False,
        help="Delete all deployments in the namespace",
    )
    @click.option(
        "--force",
        is_flag=True,
        default=False,
        help="Skip confirmation prompt",
    )
    def delete_deployments(
        deployment_name: str | None = None,
        namespace: str = "default",
        all: bool = False,
        force: bool = False,
    ):
        """
        Delete deployments from a Kubernetes namespace

        Args:
            deployment_name: Name of the deployment to delete
            namespace: Kubernetes namespace containing the deployments
            all: Delete all deployments in the namespace
            force: Skip confirmation prompt
        """
        console = Console(highlight=False)

        # Validate arguments
        if not deployment_name and not all:
            raise click.ClickException(
                "Either specify a deployment name or use --all to delete all deployments"
            )

        if deployment_name and all:
            raise click.ClickException(
                "Cannot specify both a deployment name and --all flag"
            )

        # Load Kubernetes configuration
        try:
            config.load_kube_config()
            api = client.CustomObjectsApi()
        except Exception as e:
            logger.error("Failed to load Kubernetes configuration", exc_info=True)
            raise click.ClickException(
                f"Failed to load Kubernetes configuration: {str(e)}"
            )

        # Define the group, version, and plural for the CRD
        group = "nvidia.com"
        version = "v1alpha1"
        plural = "dynamodeployments"

        with Spinner(console=console) as spinner:
            try:
                # Get deployments to delete
                deployments_to_delete = []

                if all:
                    # Get all deployments in the namespace
                    spinner.update(
                        f"Fetching all deployments in namespace '{namespace}'"
                    )
                    deployments = api.list_namespaced_custom_object(
                        group=group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                    )
                    deployments_to_delete = [
                        item["metadata"]["name"]
                        for item in deployments.get("items", [])
                    ]

                    if not deployments_to_delete:
                        spinner.log(f"No deployments found in namespace '{namespace}'")
                        return
                else:
                    # Check if the specified deployment exists
                    try:
                        api.get_namespaced_custom_object(
                            group=group,
                            version=version,
                            namespace=namespace,
                            plural=plural,
                            name=deployment_name,
                        )
                        deployments_to_delete = [deployment_name]
                    except client.rest.ApiException as e:
                        if e.status == 404:
                            spinner.log(
                                f"[bold red]Deployment '{deployment_name}' not found in namespace '{namespace}'[/]"
                            )
                            return
                        raise

                # Confirm deletion if not forced
                if not force:
                    spinner.stop()
                    if all:
                        message = f"Are you sure you want to delete all {len(deployments_to_delete)} deployments in namespace '{namespace}'?"
                    else:
                        message = f"Are you sure you want to delete deployment '{deployment_name}' in namespace '{namespace}'?"

                    if not click.confirm(message):
                        console.print("[yellow]Deletion cancelled[/]")
                        return
                    spinner.start()

                # Delete deployments
                for name in deployments_to_delete:
                    spinner.update(
                        f"Deleting deployment '{name}' in namespace '{namespace}'"
                    )
                    try:
                        api.delete_namespaced_custom_object(
                            group=group,
                            version=version,
                            namespace=namespace,
                            plural=plural,
                            name=name,
                        )
                        spinner.log(
                            f"[green]Successfully deleted deployment '{name}'[/]"
                        )
                    except client.rest.ApiException as e:
                        if e.status == 404:
                            spinner.log(
                                f"[yellow]Deployment '{name}' not found or already deleted[/]"
                            )
                        else:
                            spinner.log(
                                f"[bold red]Failed to delete deployment '{name}': {str(e)}[/]"
                            )
                            logger.error(
                                f"Failed to delete deployment '{name}'", exc_info=True
                            )

                # Final summary
                if all:
                    spinner.log(
                        f"[bold green]Deleted {len(deployments_to_delete)} deployments from namespace '{namespace}'[/]"
                    )

            except Exception as e:
                logger.error("Deletion operation failed", exc_info=True)
                spinner.log(f"[bold red]Operation failed: {str(e)}[/]")
                raise SystemExit(1)

    return cli


delete_command = build_delete_command()
