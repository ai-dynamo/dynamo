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
#  Modifications Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES

from __future__ import annotations

import logging
import subprocess
import typing as t
from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dynamo.sdk.cli.build import DYNAMO_IMAGE, BuildError, Package, build_package

logger = logging.getLogger(__name__)
console = Console()

DYNAMO_FIGLET = """
██████╗ ██╗   ██╗███╗   ██╗ █████╗ ███╗   ███╗ ██████╗
██╔══██╗╚██╗ ██╔╝████╗  ██║██╔══██╗████╗ ████║██╔═══██╗
██║  ██║ ╚████╔╝ ██╔██╗ ██║███████║██╔████╔██║██║   ██║
██║  ██║  ╚██╔╝  ██║╚██╗██║██╔══██║██║╚██╔╝██║██║   ██║
██████╔╝   ██║   ██║ ╚████║██║  ██║██║ ╚═╝ ██║╚██████╔╝
╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝
"""


def containerize(
    service: str = typer.Argument(
        ..., help="Service specification in the format module:ServiceClass"
    ),
    output_dir: t.Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for the build"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite of existing build"
    ),
    tag: t.Optional[str] = typer.Option(
        None, "--tag", "-t", help="Docker image tag (defaults to generated tag)"
    ),
) -> None:
    """
    Containerize a Dynamo service directly into a Docker image.

    This command combines the build and containerization steps, creating a
    Dynamo package and immediately building it into a Docker container.
    """
    try:
        if ":" not in service:
            console.print(
                "[red]Error: Service specification must be in format 'module:ServiceClass'[/]"
            )
            raise typer.Exit(1)

        console.print(DYNAMO_FIGLET)
        console.print("[bold green]Building and containerizing Dynamo service...[/]")
        console.print(f"[blue]Service:[/] {service}")

        # Calling shared build_package
        package = build_package(service, output_dir, force, tag)
        output_path = Path(package.path)

        docker_dir = output_path / "env" / "docker"
        docker_dir.mkdir(exist_ok=True, parents=True)
        docker_file = docker_dir / "Dockerfile"

        dockerfile_content = Package.get_dockerfile_template(DYNAMO_IMAGE)
        with open(docker_file, "w", encoding="utf-8") as f:
            f.write(dockerfile_content)

        image_tag = tag or f"{package.tag.name}:{package.tag.version}"

        console.print(f"[blue]Building Docker image:[/] {image_tag}")
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold green]Building Docker image {image_tag}..."),
            transient=True,
        ) as progress:
            progress.add_task("docker", total=None)
            subprocess.run(
                [
                    "docker",
                    "build",
                    "-t",
                    image_tag,
                    "-f",
                    str(docker_file),
                    str(output_path),
                ],
                check=True,
            )

        console.print(f"[green]Successfully built Docker image {image_tag}.")
        console.print(f"[green]Package location: {output_path}")
        console.print("\n[blue]Next steps:[/]")
        console.print(
            f"  • Run the containerized service: [cyan]docker run -p 8000:8000 {image_tag}[/]"
        )
        console.print(f"  • Push to registry: [cyan]docker push {image_tag}[/]")

    except BuildError as e:
        console.print(f"[red]Build error: {e}[/]")
        raise typer.Exit(1)
    except subprocess.CalledProcessError as e:
        console.print(f"[red]Error building Docker image: {e}[/]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error containerizing service: {str(e)}[/]")
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(containerize)
