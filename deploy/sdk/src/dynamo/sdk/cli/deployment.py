#  SPDX-FileCopyrightText: Copyright (c) 2020 Atalaya Tech. Inc
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
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import typing as t

import typer
from rich.console import Console
from rich.panel import Panel

from dynamo.sdk.core.deploy.bento_cloud import BentoCloudDeploymentManager
from dynamo.sdk.core.deploy.kubernetes import KubernetesDeploymentManager
from dynamo.sdk.core.protocol.deployment import Deployment, DeploymentManager, Service
from dynamo.sdk.core.runner import TargetEnum

app = typer.Typer(
    help="Deploy Dynamo applications to Dynamo Cloud Kubernetes Platform",
    add_completion=True,
    no_args_is_help=True,
)

console = Console(highlight=False)


def get_deployment_manager(target: str, endpoint: str) -> DeploymentManager:
    """Return the appropriate DeploymentManager for the given target and endpoint."""
    if target == "kubernetes":
        return KubernetesDeploymentManager(endpoint)
    elif target == "bento_cloud":
        return BentoCloudDeploymentManager(endpoint)
    else:
        raise ValueError(f"Unknown deployment target: {target}")


def display_deployment_info(
    deployment_manager: DeploymentManager, deployment_id: str, summary: bool = False
) -> None:
    """Display deployment summary, status, and endpoint URLs using rich panels."""
    dep = deployment_manager.get_deployment(deployment_id)
    name = dep.get("name") or dep.get("uid") or dep.get("id") or deployment_id
    status = dep.get("status", "unknown")
    urls = dep.get("urls", [])
    created_at = dep.get("created_at", "")
    status_color = {
        "running": "green",
        "in_progress": "yellow",
        "pending": "yellow",
        "failed": "red",
        "terminated": "red",
    }.get(status, "white")
    summary = (
        f"[white]Name:[/] [cyan]{name}[/]\n"
        f"[white]Status:[/] [{status_color}]{status}[/]"
    )
    if created_at:
        summary += f"\n[white]Created:[/] [magenta]{created_at}[/]"
    if urls:
        summary += f"\n[white]URLs:[/] [blue]{' | '.join(urls)}[/]"
    console.print(Panel(summary, title="Deployment", style="cyan"))


def _handle_deploy_create(
    ctx: typer.Context,
    pipeline: t.Optional[str] = typer.Argument(None, help="Dynamo pipeline to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
) -> None:
    """Handle deployment creation. This is a helper function for the create and deploy commands.

    Args:
        ctx: typer context
        pipeline: pipeline to deploy
        name: name of the deployment
    """

    from dynamo.sdk.cli.utils import configure_target_environment
    from dynamo.sdk.lib.loader import find_and_load_service

    # TODO: hardcoding this is a hack to get the services for the deployment
    # we should find a better way to do this once build is finished/generic
    configure_target_environment(TargetEnum.BENTO)
    svc = find_and_load_service(pipeline)
    pipeline_services = svc.all_services()

    services_for_deployment = [
        Service(
            name=svc.name,
            namespace=svc.config["dynamo"]["namespace"],
            envs=svc.envs,
            apis=svc.get_bentoml_service().apis,
            size_bytes=getattr(svc, "size_bytes", 0),
            # TODO: add the rest later
        )
        for svc in pipeline_services.values()
    ]

    deployment_manager = get_deployment_manager(target, endpoint)
    deployment = Deployment(
        name=name or (pipeline if pipeline else "unnamed-deployment"),
        namespace="default",
        services=services_for_deployment,
    )
    try:
        with console.status("[bold green]Creating deployment...") as status:
            deployment_id = deployment_manager.create_deployment(
                deployment,
                wait=wait,
                timeout=timeout,
                envs=envs,
                config_file=config_file,
                pipeline=pipeline,
                dev=dev,
                args=ctx.args if hasattr(ctx, "args") else None,
            )
            status.update(
                f"[bold green]Deployment '{deployment_id}' created. Waiting for status..."
            )
            if wait:
                ready = deployment_manager.wait_until_ready(
                    deployment_id, timeout=timeout
                )
                if ready:
                    console.print(
                        Panel(
                            f"Deployment [bold]{deployment_id}[/] is [green]ready[/]",
                            title="Status",
                        )
                    )
                else:
                    console.print(
                        Panel(
                            f"Deployment [bold]{deployment_id}[/] did not become ready in time.",
                            title="Status",
                            style="red",
                        )
                    )
            display_deployment_info(deployment_manager, deployment_id)
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 409:
                console.print(
                    Panel(
                        f"Deployment already exists.\n{msg}", title="Error", style="red"
                    )
                )
            elif status in (400, 422):
                console.print(
                    Panel(f"Validation error:\n{msg}", title="Error", style="red")
                )
            elif status == 404:
                console.print(
                    Panel(
                        f"Endpoint not found: {url}\n{msg}", title="Error", style="red"
                    )
                )
            elif status == 500:
                console.print(
                    Panel(f"Internal server error:\n{msg}", title="Error", style="red")
                )
            else:
                console.print(
                    Panel(
                        f"Failed to create deployment:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command()
def create(
    ctx: typer.Context,
    pipeline: t.Optional[str] = typer.Argument(None, help="Dynamo pipeline to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
) -> None:
    """Create a deployment on Dynamo Cloud."""
    _handle_deploy_create(
        ctx, pipeline, name, config_file, wait, timeout, endpoint, envs, target, dev
    )


@app.command()
def get(
    name: str = typer.Argument(..., help="Deployment name"),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Get details for a specific deployment by name."""
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status(f"[bold green]Getting deployment '{name}'..."):
            deployment = deployment_manager.get_deployment(name)
            console.print(
                Panel(str(deployment), title="Deployment Details", style="cyan")
            )
            display_deployment_info(deployment_manager, name)
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Deployment '{name}' not found.\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to get deployment:\n{msg}", title="Error", style="red"
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command("list")
def list_deployments(
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """List all deployments."""
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status("[bold green]Listing deployments..."):
            deployments = deployment_manager.list_deployments()
            if not deployments:
                console.print(
                    Panel("No deployments found.", title="Deployments", style="yellow")
                )
            else:
                console.print(Panel("[bold]Deployments List[/]", style="blue"))
                for dep in deployments:
                    dep_id = dep.get("name") or dep.get("uid") or dep.get("id")
                    if dep_id:
                        display_deployment_info(
                            deployment_manager, dep_id, summary=True
                        )
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Endpoint not found: {url}\n{msg}", title="Error", style="red"
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to list deployments:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


@app.command()
def delete(
    name: str = typer.Argument(..., help="Deployment name"),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
) -> None:
    """Delete a deployment by name."""
    deployment_manager = get_deployment_manager(target, endpoint)
    try:
        with console.status(f"[bold green]Deleting deployment '{name}'..."):
            deployment_manager.delete_deployment(name)
            console.print(
                Panel(f"Deleted deployment {name}", title="Success", style="green")
            )
    except Exception as e:
        if isinstance(e, RuntimeError) and isinstance(e.args[0], tuple):
            status, msg, url = e.args[0]
            if status == 404:
                console.print(
                    Panel(
                        f"Deployment '{name}' not found.\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
            else:
                console.print(
                    Panel(
                        f"Failed to delete deployment:\n{msg}",
                        title="Error",
                        style="red",
                    )
                )
        else:
            console.print(Panel(str(e), title="Error", style="red"))
        raise typer.Exit(1)


def deploy(
    ctx: typer.Context,
    pipeline: t.Optional[str] = typer.Argument(None, help="Dynamo pipeline to deploy"),
    name: t.Optional[str] = typer.Option(None, "--name", "-n", help="Deployment name"),
    config_file: t.Optional[typer.FileText] = typer.Option(
        None, "--config-file", "-f", help="Configuration file path"
    ),
    wait: bool = typer.Option(
        True, "--wait/--no-wait", help="Do not wait for deployment to be ready"
    ),
    timeout: int = typer.Option(
        3600, "--timeout", help="Timeout for deployment to be ready in seconds"
    ),
    endpoint: str = typer.Option(
        ..., "--endpoint", "-e", help="Dynamo Cloud endpoint", envvar="DYNAMO_CLOUD"
    ),
    envs: t.Optional[t.List[str]] = typer.Option(
        None,
        "--env",
        help="Environment variable(s) to set (format: KEY=VALUE). Note: These environment variables will be set on ALL services in your Dynamo pipeline.",
    ),
    target: str = typer.Option(..., "--target", "-t", help="Deployment target"),
    dev: bool = typer.Option(False, "--dev", help="Development mode for deployment"),
) -> None:
    """Deploy a Dynamo pipeline (same as deployment create)."""
    _handle_deploy_create(
        ctx, pipeline, name, config_file, wait, timeout, endpoint, envs, target, dev
    )
