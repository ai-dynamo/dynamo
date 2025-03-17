#  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0
#  #
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#  #
#  http://www.apache.org/licenses/LICENSE-2.0
#  #
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from __future__ import annotations

import json
import logging
import os
import sys
import time
import typing as t
from datetime import datetime

import click
from bentoml._internal.cloud.base import Spinner
from bentoml.exceptions import BentoMLException, CLIException
from rich.console import Console
from simple_di import inject

from dynamo.sdk.cli.deployment import DynamoDeployment

logger = logging.getLogger(__name__)


@click.group(name="deploy")
def deploy_command_group():
    """Deploy to a cluster"""
    pass


def convert_env_to_dict(env: tuple[str, ...] | None) -> list[dict[str, str]] | None:
    if env is None:
        return None
    collected_envs: list[dict[str, str]] = []
    if env:
        for item in env:
            if "=" in item:
                name, value = item.split("=", 1)
            else:
                name = item
                if name not in os.environ:
                    raise CLIException(f"Environment variable {name} not found")
                value = os.environ[name]
            collected_envs.append({"name": name, "value": value})
    return collected_envs


def build_deploy_command() -> click.Command:
    from bentoml._internal.utils import add_experimental_docstring

    @click.command(name="deploy")
    @click.argument("bento", type=click.STRING, default=".")
    @click.option("-n", "--name", type=click.STRING, help="Deployment name")
    @click.option(
        "--namespace",
        type=click.STRING,
        default="default",
        help="Kubernetes namespace to deploy to",
    )
    @click.option(
        "--scaling-min",
        type=click.INT,
        default=1,
        show_default=True,
        help="Minimum scaling value",
    )
    @click.option(
        "--scaling-max",
        type=click.INT,
        default=5,
        show_default=True,
        help="Maximum scaling value",
    )
    @click.option("--instance-type", type=click.STRING, help="Type of instance")
    @click.option(
        "--env",
        type=click.STRING,
        multiple=True,
        default=[],
        help="Environment variables in key=value format",
    )
    @click.option("--secret", type=click.STRING, multiple=True, help="Secret names")
    @click.option(
        "-f",
        "--config-file",
        type=click.Path(exists=True, dir_okay=False, readable=True),
        help="Configuration file path",
    )
    @click.option(
        "--wait/--no-wait", default=True, help="Wait for deployment to be ready"
    )
    @click.option(
        "--timeout",
        type=click.INT,
        default=600,
        help="Timeout for deployment readiness in seconds",
    )
    @click.option(
        "--working-dir",
        type=click.Path(),
        default=None,
        show_default=True,
        help="Directory to find the Service instance",
    )
    @click.option(
        "--access-authorization", type=click.BOOL, default=False, show_default=True
    )
    @click.option("--strategy", type=click.STRING, default="rolling-update")
    @click.option("--version", type=click.STRING, help="Version tag for the Bento")
    @add_experimental_docstring
    def deploy_command(
        bento: str | None,
        name: str | None,
        namespace: str | None = "default",
        access_authorization: bool | None = False,
        scaling_min: int | None = 1,
        scaling_max: int | None = 5,
        instance_type: str | None = None,
        strategy: str | None = "rolling-update",
        env: tuple[str, ...] | None = None,
        secret: tuple[str] | None = None,
        config_file: str | t.TextIO | None = None,
        config_dict: str | None = None,
        wait: bool = True,
        timeout: int = 600,
        working_dir: str | None = None,
        version: str | None = None,
    ):
        """
        Deploy a set of Dynamo services in a Bento to a K8s cluster

        \b
        BENTO is the serving target, it can be:
        - a tag to a Bento in local Bento store
        - a path to a built Bento
        """
        from bentoml._internal.log import configure_server_logging

        configure_server_logging()

        # Fix handling of None values
        if working_dir is None:
            if bento is not None and os.path.isdir(os.path.expanduser(bento)):
                working_dir = os.path.expanduser(bento)
            else:
                working_dir = "."

        # Make sure working_dir is in the front of sys.path for imports
        if sys.path[0] != working_dir:
            sys.path.insert(0, working_dir)

        # Load the Bento to validate
        import bentoml
        from bentoml._internal.service.loader import load

        # Check if the bento exists in the local store
        bento_exists = False
        bento_tag = None

        try:
            bentos = bentoml.list()
            bento_tags = [str(b.tag) for b in bentos]
            bento_exists = bento in bento_tags

            if bento_exists:
                bento_tag = bento
                logger.debug("Verified Bento exists: %s", bento_tag)
            else:
                # If not a tag, check if it's a path to a built Bento
                if bento is not None and os.path.isdir(bento):
                    service_name = os.path.basename(os.path.abspath(bento))
                    bento_version = (
                        version or f"v{datetime.now().strftime('%Y%m%d%H%M%S')}"
                    )
                    bento_tag = f"{service_name}:{bento_version}"
                    logger.debug(
                        "Using Bento from directory: %s with tag: %s", bento, bento_tag
                    )
                else:
                    raise click.ClickException(
                        f"Invalid Bento reference: {bento}. Ensure it's a valid Bento tag or directory."
                    )
        except Exception as exception_var:
            logger.error("Bento validation failed:", exc_info=True)
            raise click.ClickException(
                f"Failed to validate bento: {str(exception_var)}"
            )

        # Load the service to validate it
        svc = load(bento_identifier=bento_tag, working_dir=working_dir)
        print(f"Service loaded: {svc}")

        create_dynamo_deployment(
            bento=bento_tag,
            name=name,
            namespace=namespace,
            access_authorization=access_authorization,
            scaling_min=scaling_min,
            scaling_max=scaling_max,
            instance_type=instance_type,
            strategy=strategy,
            env=env,
            secret=secret,
            config_file=config_file,
            config_dict=config_dict,
            wait=wait,
            timeout=timeout,
        )

    return deploy_command


deploy_command = build_deploy_command()


@inject
def create_dynamo_deployment(
    bento: str | None = None,
    name: str | None = None,
    namespace: str | None = "default",
    access_authorization: bool | None = None,
    scaling_min: int | None = None,
    scaling_max: int | None = None,
    instance_type: str | None = None,
    strategy: str | None = None,
    env: tuple[str, ...] | None = None,
    secret: tuple[str] | None = None,
    config_file: str | t.TextIO | None = None,
    config_dict: str | None = None,
    wait: bool = True,
    timeout: int = 3600,
    dev: bool = False,
) -> DynamoDeployment:
    from bentoml._internal.cloud.deployment import DeploymentConfigParameters
    from bentoml_cli.deployment import raise_deployment_config_error
    from kubernetes import client, config

    cfg_dict = None
    if config_dict is not None and config_dict != "":
        cfg_dict = json.loads(config_dict)

    config_params = DeploymentConfigParameters(
        name=name,
        bento=bento,
        cluster=namespace,
        access_authorization=access_authorization,
        scaling_max=scaling_max,
        scaling_min=scaling_min,
        instance_type=instance_type,
        strategy=strategy,
        envs=convert_env_to_dict(tuple(env) if env else None),
        secrets=list(secret) if secret is not None else None,
        config_file=config_file,
        config_dict=cfg_dict,
        cli=True,
        dev=dev,
    )

    try:
        config_params.verify()
    except BentoMLException as exc:
        error_message = str(exc)
        raise_deployment_config_error(error_message, "create")

    # Fix the deployment name generation
    deployment_name = name
    if deployment_name is None and bento is not None:
        deployment_name = f"{bento.replace(':', '-').replace('/', '-')}-deployment"
    print(f"Deployment name: {deployment_name}")

    # Create the deployment object
    deployment = DynamoDeployment.create_deployment(
        deployment_name=deployment_name,
        namespace=namespace,
        config=config,
    )

    # Convert env tuple to k8s env format
    env_vars = []
    if env:
        for e in env:
            if "=" in e:
                k, v = e.split("=", 1)
                env_vars.append({"name": k, "value": v})

    # Get the CRD payload
    crd_payload = deployment.get_crd_payload(
        bento=bento,
        scaling_min=scaling_min or 1,
        scaling_max=scaling_max or 5,
        instance_type=instance_type,
        env_vars=env_vars,
        secret=list(secret) if secret else [],
    )

    console = Console(highlight=False)
    with Spinner(console=console) as spinner:
        try:
            spinner.update("Creating deployment via Kubernetes operator")
            config.load_kube_config()
            api = client.CustomObjectsApi()

            # Create the CRD
            group = "nvidia.com"
            version = "v1alpha1"
            plural = "dynamodeployments"

            created_crd = api.create_namespaced_custom_object(
                group=group,
                version=version,
                namespace=namespace,
                plural=plural,
                body=crd_payload,
            )

            # Add validation and logging for the created CRD
            if not created_crd:
                raise click.ClickException(
                    "Failed to create deployment: No response from Kubernetes API"
                )

            logger.debug("Created CRD: %s", json.dumps(created_crd, indent=2))
            spinner.log(
                f':white_check_mark: Created deployment "{deployment_name}" in namespace "{namespace}"'
            )

            if wait:
                spinner.update("Waiting for deployment to become ready")
                start_time = time.time()
                while time.time() - start_time < timeout:
                    # Check status using Kubernetes API
                    status_data = api.get_namespaced_custom_object(
                        group=group,
                        version=version,
                        namespace=namespace,
                        plural=plural,
                        name=deployment_name,
                    )

                    state = status_data.get("status", {}).get("state", "Pending")
                    conditions = status_data.get("status", {}).get("conditions", [])
                    events = status_data.get("status", {}).get("events", [])

                    logger.debug(f"Current deployment state: {state}")
                    logger.debug(f"Conditions: {json.dumps(conditions, indent=2)}")
                    logger.debug(f"Events: {json.dumps(events, indent=2)}")
                    logger.debug(f"Time elapsed: {int(time.time() - start_time)}s")
                    logger.debug(
                        f"Full status: {json.dumps(status_data.get('status', {}), indent=2)}"
                    )

                    # Check for successful states
                    if state.lower() in [
                        "running",
                        "ready",
                        "active",
                        "available",
                        "complete",
                    ]:
                        if deployment.ingress_url:
                            spinner.log("[bold green]Deployment ready![/]")
                            spinner.log(
                                f"[bold]Ingress URL: {deployment.ingress_url}[/]"
                            )
                        else:
                            spinner.log("[bold green]Deployment ready![/]")
                        return deployment
                    # Check for failed states
                    elif state.lower() in [
                        "failed",
                        "error",
                        "unavailable",
                        "degraded",
                    ]:
                        error_message = status_data.get("message", "Unknown error")
                        if conditions:
                            error_message = next(
                                (
                                    c.get("message", error_message)
                                    for c in conditions
                                    if c.get("type", "").lower() == "failed"
                                ),
                                error_message,
                            )

                        # Add more detailed error logging
                        logger.error(f"Deployment failed with state: {state}")
                        logger.error(f"Error message: {error_message}")
                        logger.error(f"Conditions: {json.dumps(conditions, indent=2)}")
                        logger.error(f"Events: {json.dumps(events, indent=2)}")

                        raise click.ClickException(
                            f"Deployment failed: {error_message}\n"
                            f"State: {state}\n"
                            f"Conditions: {json.dumps(conditions, indent=2)}\n"
                            f"Events: {json.dumps(events, indent=2)}"
                        )

                    time.sleep(5)

                if time.time() - start_time >= timeout:
                    # Check if deployment exists but we just timed out waiting
                    try:
                        final_status = api.get_namespaced_custom_object(
                            group=group,
                            version=version,
                            namespace=namespace,
                            plural=plural,
                            name=deployment_name,
                        )
                        if final_status.get("status", {}).get("state", "").lower() in [
                            "running",
                            "ready",
                            "active",
                            "available",
                            "complete",
                        ]:
                            if deployment.ingress_url:
                                spinner.log("[bold green]Deployment ready![/]")
                                spinner.log(
                                    f"[bold]Ingress URL: {deployment.ingress_url}[/]"
                                )
                            else:
                                spinner.log("[bold green]Deployment ready![/]")
                            return deployment
                    except Exception as e:
                        logger.error("Timeout check failed", exc_info=True)

                    # Add timeout debug information
                    pods = api.list_namespaced_pod(
                        namespace, label_selector=f"app={deployment_name}"
                    )
                    logger.error(
                        f"Timeout reached. Pod statuses: {json.dumps([p.status for p in pods.items], indent=2)}"
                    )

                    raise click.ClickException(
                        f"Deployment timeout reached\n"
                        f"Pod statuses: {json.dumps([p.status for p in pods.items], indent=2)}"
                    )

            return deployment

        except Exception as e:
            logger.error("Deployment failed", exc_info=True)
            spinner.log(f"[bold red]Deployment failed: {str(e)}[/]")
            raise SystemExit(1)


deploy_command = build_deploy_command()
