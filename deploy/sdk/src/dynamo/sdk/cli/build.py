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

import datetime
import importlib
import importlib.util
import inspect
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import typing as t
import uuid
from pathlib import Path
from typing import TypeVar

import typer
import yaml
from pydantic import BaseModel, Field
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from dynamo.sdk import DYNAMO_IMAGE
from dynamo.sdk.core.protocol.interface import (
    DynamoTransport,
    LinkedServices,
    ServiceInterface,
)
from dynamo.sdk.core.runner import TargetEnum

logger = logging.getLogger(__name__)
console = Console()
T = TypeVar("T", bound=object)

DYNAMO_FIGLET = """
██████╗ ██╗   ██╗███╗   ██╗ █████╗ ███╗   ███╗ ██████╗
██╔══██╗╚██╗ ██╔╝████╗  ██║██╔══██╗████╗ ████║██╔═══██╗
██║  ██║ ╚████╔╝ ██╔██╗ ██║███████║██╔████╔██║██║   ██║
██║  ██║  ╚██╔╝  ██║╚██╗██║██╔══██║██║╚██╔╝██║██║   ██║
██████╔╝   ██║   ██║ ╚████║██║  ██║██║ ╚═╝ ██║╚██████╔╝
╚═════╝    ╚═╝   ╚═╝  ╚═══╝╚═╝  ╚═╝╚═╝     ╚═╝ ╚═════╝
"""

# --- Custom exceptions ---


class InvalidArgument(Exception):
    """Exception raised for invalid arguments."""

    pass


class BuildError(Exception):
    """Exception raised for build errors."""

    pass


# --- Data models ---


class Tag(BaseModel):
    """Tag for identifying a package."""

    name: str
    version: t.Optional[str] = None

    def __str__(self) -> str:
        if self.version:
            return f"{self.name}:{self.version}"
        return self.name

    def make_new_version(self) -> Tag:
        """Create a new version based on timestamp."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        short_uuid = uuid.uuid4().hex[:8]
        return Tag(name=self.name, version=f"{timestamp}_{short_uuid}")


class ServiceConfig(BaseModel):
    """Configuration for a service."""

    name: str
    service: str = ""  # Fully qualified service name
    models: t.List[str] = Field(default_factory=list)
    dependencies: t.List[str] = Field(default_factory=list)
    resources: t.Dict[str, t.Any] = Field(default_factory=dict)
    workers: t.Optional[int] = None
    image: str = "dynamo:latest"
    dynamo: t.Dict[str, t.Any] = Field(default_factory=dict)
    http_exposed: bool = False
    api_endpoints: t.List[str] = Field(default_factory=list)


class ServiceInfo(BaseModel):
    """Information about a service."""

    name: str
    module_path: str
    class_name: str
    config: ServiceConfig

    @classmethod
    def from_service(cls, service: ServiceInterface[T]) -> ServiceInfo:
        """Create ServiceInfo from a service instance."""
        service_class = service.inner
        name = getattr(service, "name", service_class.__name__)

        # Extract API endpoints if available
        api_endpoints = []
        for ep_name, endpoint in service.get_dynamo_endpoints().items():
            if DynamoTransport.HTTP in endpoint.transports:
                api_endpoints.append(f"/{ep_name}")

        image = service.config.image or DYNAMO_IMAGE
        assert (
            image is not None
        ), "Please set DYNAMO_IMAGE environment variable or image field in service config"

        # Create config
        config = ServiceConfig(
            name=name,
            service="",
            resources=service.config.resources.model_dump(),
            workers=service.config.workers,
            image=image,
            dynamo=service.config.dynamo.model_dump(),
            http_exposed=len(api_endpoints) > 0,
            api_endpoints=api_endpoints,
        )

        return cls(
            name=name,
            module_path=service.__module__,
            class_name=service_class.__name__,
            config=config,
        )


class BuildConfig(BaseModel):
    """Configuration for building a Dynamo pipeline."""

    service: str
    name: t.Optional[str] = None
    version: t.Optional[str] = None
    tag: t.Optional[str] = None
    include: t.List[str] = Field(
        default_factory=lambda: [
            "**/*.py",
            "**/*.yaml",
            "**/*.json",
            "**/*.toml",
            "**/*.md",
            "**/*.sh",
        ]
    )
    exclude: t.List[str] = Field(
        default_factory=lambda: [
            "**/__pycache__/**",
            "**/.git/**",
        ]
    )
    labels: t.Dict[str, str] = Field(default_factory=dict)
    envs: t.List[str] = Field(default_factory=list)
    docker: t.Dict[str, t.Any] = Field(default_factory=dict)

    def to_yaml(self, file_obj: t.TextIO) -> None:
        """Write config to YAML file."""
        yaml.dump(self.model_dump(), file_obj)

    def with_defaults(self) -> BuildConfig:
        """Return config with default values filled in."""
        return self


class ManifestInfo(BaseModel):
    """Information for generating a manifest file."""

    service: str
    name: str
    version: str
    creation_time: str
    labels: t.Dict[str, str]
    entry_service: str
    services: t.List[ServiceInfo]
    envs: t.List[str]

    def to_dict(self) -> t.Dict[str, t.Any]:
        """Convert to dictionary for YAML serialization."""
        result = self.model_dump()
        # Convert ServiceInfo objects to dictionaries
        services_dict = []
        for service in result["services"]:
            service_dict = {
                "name": service["name"],
                "service": service["config"]["service"],
                "config": {
                    "resources": service["config"]["resources"],
                    "workers": service["config"]["workers"],
                    "image": service["config"]["image"],
                    "dynamo": service["config"]["dynamo"],
                },
            }

            # Add HTTP configuration if exposed
            if service["config"]["http_exposed"]:
                service_dict["config"]["http_exposed"] = True
                service_dict["config"]["api_endpoints"] = service["config"][
                    "api_endpoints"
                ]
            services_dict.append(service_dict)
        result["services"] = services_dict
        return result


class PackageInfo(BaseModel):
    """Information about a built package."""

    tag: Tag
    path: str
    service: str
    services: t.List[ServiceInfo]
    entry_service: str
    labels: t.Dict[str, str]
    envs: t.List[str]

    def to_yaml(self) -> str:
        """Convert to YAML string."""
        return yaml.dump(self.model_dump())

    def to_manifest(self) -> ManifestInfo:
        """Convert to manifest information."""
        return ManifestInfo(
            service=self.service,
            name=self.tag.name,
            version=self.tag.version if self.tag.version else "",
            creation_time=datetime.datetime.now().strftime(
                "%Y-%m-%dT%H:%M:%S.%f+00:00"
            ),
            labels=self.labels,
            entry_service=self.entry_service,
            services=self.services,
            envs=self.envs,
        )


class Package:
    """Dynamo package that bundles services for deployment."""

    def __init__(self, tag: Tag, path: str, info: PackageInfo):
        self.tag = tag
        self.path = path
        self.info = info

    def __str__(self) -> str:
        return str(self.tag)

    @classmethod
    def dynamo_service(
        cls,
        build_config: BuildConfig,
        build_ctx: t.Optional[str] = None,
    ) -> t.Any:
        """Get a dynamo service from config."""
        build_ctx = (
            os.getcwd()
            if build_ctx is None
            else os.path.realpath(os.path.expanduser(build_ctx))
        )

        if not os.path.isdir(build_ctx):
            raise InvalidArgument(
                f"Build context {build_ctx} does not exist or is not a directory."
            )

        # Load the service
        from dynamo.sdk.lib.loader import find_and_load_service

        dyn_svc = find_and_load_service(build_config.service, working_dir=build_ctx)
        # Clean up unused edges
        LinkedServices.remove_unused_edges()
        dyn_svc.inject_config()
        return dyn_svc

    @classmethod
    def create(
        cls,
        build_config: BuildConfig,
        build_ctx: str,
        version: t.Optional[str] = None,
    ) -> Package:
        dyn_svc = cls.dynamo_service(build_config, build_ctx)

        # Get service name for package
        package_name = cls.to_package_name(build_config.service)
        # image: str = dyn_svc.image

        # Use provided version or create new one
        if version is None:
            version = build_config.version

        # Create tag with version
        tag = Tag(name=package_name, version=version)
        if version is None:
            tag = tag.make_new_version()

        logger.debug(
            f'Building Dynamo package "{tag}" from build context "{build_ctx}".'
        )

        # Create temporary directory for package
        package_dir = tempfile.mkdtemp(prefix=f"dynamo_package_{package_name}_")

        # Copy files based on include/exclude patterns
        cls.copy_files(
            build_ctx, package_dir, build_config.include, build_config.exclude
        )

        # Get info about all services
        all_services = list(dyn_svc.all_services().values())
        services_info = [ServiceInfo.from_service(s) for s in all_services]

        # Create package info
        package_info = PackageInfo(
            tag=tag,
            service=build_config.service,
            path=package_dir,
            services=services_info,
            entry_service=dyn_svc.name,
            labels=build_config.labels,
            envs=build_config.envs,
        )

        # Create the package
        package = cls(tag, package_dir, package_info)
        # Write package info and manifests
        return package

    def generate_manifests(self) -> None:
        """Generate manifest files for the package."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Generating manifests..."),
            transient=True,
        ) as progress:
            progress.add_task("generate", total=None)

            manifest = self.info.to_manifest()
            manifest_dict = manifest.to_dict()
            with open(os.path.join(self.path, "dynamo.yaml"), "w") as f:
                yaml.dump(manifest_dict, f, default_flow_style=False)

    @staticmethod
    def load_service(service_path: str, working_dir: str) -> t.Any:
        """Load a service from a path."""
        logger.info(f"Loading service from: {service_path}")

        # Add working directory to sys.path
        sys.path.insert(0, working_dir)

        try:
            # Handle module:class format
            if ":" in service_path:
                module_path, class_name = service_path.split(":", 1)
                module = importlib.import_module(module_path)
                return getattr(module, class_name)

            # Handle direct Python file
            elif service_path.endswith(".py"):
                module_name = os.path.basename(service_path)[:-3]
                spec = importlib.util.spec_from_file_location(
                    module_name, os.path.join(working_dir, service_path)
                )
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load {service_path}")

                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)

                # Find the service class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isclass(attr) and attr.__module__ == module.__name__:
                        # Simple heuristic - find a class defined in this module
                        return attr

                raise ImportError(f"No service class found in {service_path}")

            # Handle Python module
            else:
                module = importlib.import_module(service_path)
                # Find the service class
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if inspect.isclass(attr) and attr.__module__ == module.__name__:
                        # Simple heuristic - find a class defined in this module
                        return attr

                raise ImportError(f"No service class found in {service_path}")

        finally:
            # Remove working directory from sys.path
            sys.path.pop(0)

    @staticmethod
    def to_package_name(name: str) -> str:
        """Convert CamelCase to snake_case."""
        import re

        name = name.split(":")[1].lower()
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1)
        ret = s2.replace(":", "_")
        return ret

    @staticmethod
    def _get_dockerfile_template(base_image: str = "dynamo:latest") -> str:
        """Get the Dockerfile template content with configurable base image."""
        # Try to load the Dockerfile.template template from the CLI directory
        cli_template_path = Path(__file__).parent / "Dockerfile.template"
        if not cli_template_path.exists():
            raise FileNotFoundError(
                f"Dockerfile template not found at {cli_template_path}"
            )
        with open(cli_template_path, "r") as f:
            template_content = f.read()
            # Replace the base image placeholder with the actual base image
            template_content = template_content.replace("__BASE_IMAGE__", base_image)
            return template_content

    @staticmethod
    def copy_files(
        source_dir: str,
        target_dir: str,
        include_patterns: t.List[str],
        exclude_patterns: t.List[str],
    ) -> None:
        """Copy files based on include/exclude patterns."""
        import glob

        # Create set of all files to include
        all_files = set()
        for pattern in include_patterns:
            pattern_path = os.path.join(source_dir, pattern)
            matched_files = glob.glob(pattern_path, recursive=True)
            all_files.update(matched_files)

        # Remove excluded files
        for pattern in exclude_patterns:
            pattern_path = os.path.join(source_dir, pattern)
            excluded_files = glob.glob(pattern_path, recursive=True)
            all_files.difference_update(excluded_files)

        # Copy each file preserving relative path
        for file_path in all_files:
            if os.path.isfile(file_path):
                rel_path = os.path.relpath(file_path, source_dir)
                target_path = os.path.join(target_dir, rel_path)
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                shutil.copy2(file_path, target_path)


def build(
    service: str = typer.Argument(
        ..., help="Service specification in the format module:ServiceClass"
    ),
    output_dir: t.Optional[str] = typer.Option(
        None, "--output-dir", "-o", help="Output directory for the build"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force overwrite of existing build"
    ),
    containerize: bool = typer.Option(
        False,
        "--containerize",
        help="Containerize the dynamo pipeline after building.",
    ),
) -> None:
    """Packages Dynamo service for deployment. Optionally builds a docker container."""
    from dynamo.sdk.cli.utils import configure_target_environment

    configure_target_environment(TargetEnum.DYNAMO)

    # Determine output directory
    if output_dir is None:
        # Default to ~/.dynamo/packages/service_name
        graph_name = service.rsplit(":", 1)[-1].lower()
        dynamo_tag = generate_random_tag()
        output_dir = str(Path.home() / ".dynamo" / "packages" / graph_name / dynamo_tag)

    output_path = Path(output_dir)

    # Check if output directory exists
    if output_path.exists() and not force:
        console.print(
            f"[bold red]Output directory {output_dir} already exists. Use --force to overwrite.[/]"
        )
        raise typer.Exit(1)

    source_dir = output_path / "src"
    source_dir.mkdir(exist_ok=True, parents=True)

    build_ctx = "."
    build_config = BuildConfig(
        service=service,
        tag=dynamo_tag,
    )

    try:
        # Create the package
        package = Package.create(
            build_config=build_config,
            version=dynamo_tag,
            build_ctx=build_ctx,
        )

        # Copy to output directory
        with Progress(
            SpinnerColumn(),
            TextColumn(f"[bold green]Copying package to {output_dir}..."),
            transient=True,
        ) as progress:
            progress.add_task("copy", total=None)

            for item in os.listdir(package.path):
                s = os.path.join(package.path, item)
                d = os.path.join(source_dir, item)
                if os.path.isdir(s):
                    shutil.copytree(s, d, dirs_exist_ok=True)
                else:
                    shutil.copy2(s, d)

        # Update package path
        package.path = output_dir
        package.generate_manifests()

        console.print(DYNAMO_FIGLET)
        console.print(f"[green]Successfully built {package.tag}.")
        console.print(f"[green]Output directory: {output_dir}")

        next_steps = []
        if not containerize:
            next_steps.append(
                "\n\n* Containerize your Dynamo pipeline with "
                "`dynamo build --containerize <service_name>`:\n"
                f"    $ dynamo build --containerize {service}"
            )

        if next_steps:
            console.print(f"\n[blue]Next steps: {''.join(next_steps)}[/]")

        docker_dir = output_path / "env" / "docker"
        docker_dir.mkdir(exist_ok=True, parents=True)
        docker_file = docker_dir / "Dockerfile"

        dockerfile_content = Package._get_dockerfile_template(DYNAMO_IMAGE)
        with open(docker_file, "w") as f:
            f.write(dockerfile_content)

        if containerize:
            # Generate Dockerfile next to dynamo.yaml using template
            # Build Docker image
            image_name = f"{package.tag.name}:{package.tag.version}"
            with Progress(
                SpinnerColumn(),
                TextColumn(f"[bold green]Building Docker image {image_name}..."),
                transient=True,
            ) as progress:
                progress.add_task("docker", total=None)
                subprocess.run(
                    [
                        "docker",
                        "build",
                        "-t",
                        image_name,
                        "-f",
                        str(docker_file),
                        output_path,
                    ],
                    check=True,
                )
            console.print(f"[green]Successfully built Docker image {image_name}.")
    except Exception as e:
        console.print(f"[red]Error building package: {str(e)}")
        raise


def generate_random_tag() -> str:
    """Generate a random tag for the Dynamo pipeline."""
    return f"{uuid.uuid4().hex[:8]}"


if __name__ == "__main__":
    typer.run(build)
