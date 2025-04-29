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

import collections
import contextlib
import json
import logging
import os
import pathlib
import random
import socket
from typing import Any, DefaultDict, Dict, Iterator, Optional, Protocol, TextIO, Union

import click
import yaml
from click import Command, Context

from dynamo.runtime.logging import configure_dynamo_logging

configure_dynamo_logging()

logger = logging.getLogger(__name__)

DYN_LOCAL_STATE_DIR = "DYN_LOCAL_STATE_DIR"


# Define a Protocol for services to ensure type safety
class ServiceProtocol(Protocol):
    name: str
    inner: Any
    models: list[Any]
    bento: Any

    def is_dynamo_component(self) -> bool:
        ...

    def dynamo_address(self) -> tuple[str, str]:
        ...


class DynamoCommandGroup(click.Group):
    """Simplified version of BentoMLCommandGroup for Dynamo CLI"""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.aliases = kwargs.pop("aliases", [])
        super().__init__(*args, **kwargs)
        self._commands: dict[str, list[str]] = {}
        self._aliases: dict[str, str] = {}

    def add_command(self, cmd: Command, name: str | None = None) -> None:
        assert cmd.callback is not None
        callback = cmd.callback
        cmd.callback = callback
        cmd.context_settings["max_content_width"] = 120
        aliases = getattr(cmd, "aliases", None)
        if aliases:
            assert cmd.name
            self._commands[cmd.name] = aliases
            self._aliases.update({alias: cmd.name for alias in aliases})
        return super().add_command(cmd, name)

    def add_subcommands(self, group: click.Group) -> None:
        if not isinstance(group, click.MultiCommand):
            raise TypeError(
                "DynamoCommandGroup.add_subcommands only accepts click.MultiCommand"
            )
        if isinstance(group, DynamoCommandGroup):
            # Common wrappers are already applied, call the super() method
            for name, cmd in group.commands.items():
                super().add_command(cmd, name)
            self._commands.update(group._commands)
            self._aliases.update(group._aliases)
        else:
            for name, cmd in group.commands.items():
                self.add_command(cmd, name)

    def resolve_alias(self, cmd_name: str):
        return self._aliases[cmd_name] if cmd_name in self._aliases else cmd_name

    def get_command(self, ctx: Context, cmd_name: str) -> Command | None:
        cmd_name = self.resolve_alias(cmd_name)
        return super().get_command(ctx, cmd_name)

    def add_single_command(self, group: click.Group, command_name: str) -> None:
        """Add a single command from a group by name."""
        if not isinstance(group, click.MultiCommand):
            raise TypeError("Only accepts click.MultiCommand")

        ctx = click.Context(group)
        cmd = group.get_command(ctx, command_name)
        if cmd is None:
            raise ValueError(f"Command '{command_name}' not found in group")

        self.add_command(cmd, command_name)


@contextlib.contextmanager
def reserve_free_port(
    host: str = "localhost",
    port: int | None = None,
    prefix: Optional[str] = None,
    max_retry: int = 50,
    enable_so_reuseport: bool = False,
) -> Iterator[int]:
    """
    detect free port and reserve until exit the context
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if enable_so_reuseport:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
        if sock.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT) == 0:
            raise RuntimeError("Failed to set SO_REUSEPORT.") from None

    if prefix is not None:
        prefix_num = int(prefix) * 10 ** (5 - len(prefix))
        suffix_range = min(65535 - prefix_num, 10 ** (5 - len(prefix)))
        for _ in range(max_retry):
            suffix = random.randint(0, suffix_range)
            port = int(f"{prefix_num + suffix}")
            try:
                sock.bind((host, port))
                break
            except OSError:
                continue
        else:
            raise RuntimeError(
                f"Cannot find free port with prefix {prefix} after {max_retry} retries."
            ) from None
    else:
        if port:
            sock.bind((host, port))
        else:
            sock.bind((host, 0))
    try:
        yield sock.getsockname()[1]
    finally:
        sock.close()


def save_dynamo_state(
    namespace: str,
    circus_endpoint: str,
    components: dict[str, Any],
    environment: dict[str, Any],
):
    state_dir = os.environ.get(
        DYN_LOCAL_STATE_DIR, os.path.expanduser("~/.dynamo/state")
    )
    os.makedirs(state_dir, exist_ok=True)

    # create the state object
    state = {
        "namespace": namespace,
        "circus_endpoint": circus_endpoint,
        "components": components,
        "environment": environment,
    }

    # save the state object to a file
    state_file = os.path.join(state_dir, f"{namespace}.json")
    with open(state_file, "w") as f:
        json.dump(state, f)

    logger.warning(f"Saved state to {state_file}")


def append_dynamo_state(namespace: str, component_name: str, data: dict) -> None:
    """Append additional data to an existing component's state"""
    state_dir = os.environ.get(
        DYN_LOCAL_STATE_DIR, os.path.expanduser("~/.dynamo/state")
    )
    state_file = os.path.join(state_dir, f"{namespace}.json")

    if not os.path.exists(state_file):
        logger.warning(
            f"Skipping append to state file {state_file} because it doesn't exist"
        )
        return

    with open(state_file, "r") as f:
        state = json.load(f)

    if "components" not in state:
        state["components"] = {}
    if component_name not in state["components"]:
        state["components"][component_name] = {}

    state["components"][component_name].update(data)

    logger.warning(f"Appending {data} to {component_name} in {state_file}")

    with open(state_file, "w") as f:
        json.dump(state, f)


def _parse_service_arg(arg_name: str, arg_value: str) -> tuple[str, str, Any]:
    """Parse a single CLI argument into service name, key, and value."""

    parts = arg_name.split(".")
    service = parts[0]
    nested_keys = parts[1:]

    # Special case: if this is a ServiceArgs.envs.* path, keep value as string
    if (
        len(nested_keys) >= 2
        and nested_keys[0] == "ServiceArgs"
        and nested_keys[1] == "envs"
    ):
        value: Union[str, int, float, bool, dict, list] = arg_value
    else:
        # Parse value based on type for non-env vars
        try:
            value = json.loads(arg_value)
        except json.JSONDecodeError:
            if arg_value.isdigit():
                value = int(arg_value)
            elif arg_value.replace(".", "", 1).isdigit() and arg_value.count(".") <= 1:
                value = float(arg_value)
            elif arg_value.lower() in ("true", "false"):
                value = arg_value.lower() == "true"
            else:
                value = arg_value

    # Build nested dict structure
    result = value
    for key in reversed(nested_keys[1:]):
        result = {key: result}

    return service, nested_keys[0], result


def _parse_service_args(args: list[str]) -> Dict[str, Any]:
    service_configs: DefaultDict[str, Dict[str, Any]] = collections.defaultdict(dict)

    def deep_update(d: dict, key: str, value: Any):
        """
        Recursively updates nested dictionaries. We use this to process arguments like

        ---Worker.ServiceArgs.env.CUDA_VISIBLE_DEVICES="0,1"

        The _parse_service_arg function will parse this into:
        service = "Worker"
        nested_keys = ["ServiceArgs", "envs", "CUDA_VISIBLE_DEVICES"]

        And returns: ("VllmWorker", "ServiceArgs", {"envs": {"CUDA_VISIBLE_DEVICES": "0,1"}})

        We then use deep_update to update the service_configs dictionary with this nested value.
        """
        if isinstance(value, dict) and key in d and isinstance(d[key], dict):
            for k, v in value.items():
                deep_update(d[key], k, v)
        else:
            d[key] = value

    index = 0
    while index < len(args):
        next_arg = args[index]

        if not (next_arg.startswith("--") or "." not in next_arg):
            continue
        try:
            if "=" in next_arg:
                arg_name, arg_value = next_arg.split("=", 1)
                index += 1
            elif args[index + 1] == "=":
                arg_name = next_arg
                arg_value = args[index + 2]
                index += 3
            else:
                arg_name = next_arg
                arg_value = args[index + 1]
                index += 2
            if arg_value.startswith("-"):
                raise ValueError("Service arg value can not start with -")
            arg_name = arg_name[2:]
            service, key, value = _parse_service_arg(arg_name, arg_value)
            deep_update(service_configs[service], key, value)
        except Exception:
            raise ValueError(f"Error parsing service arg: {args[index]}")

    return service_configs


def resolve_service_config(
    config_file: pathlib.Path | TextIO | None = None,
    args: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Resolve service configuration from file and command line arguments.

    Args:
        config_file: Path to YAML config file or file object
        args: List of command line arguments

    Returns:
        Dictionary mapping service names to their configurations
    """
    service_configs: dict[str, dict[str, Any]] = {}

    # Check for deployment config first
    if "DYN_DEPLOYMENT_CONFIG" in os.environ:
        try:
            deployment_config = yaml.safe_load(os.environ["DYN_DEPLOYMENT_CONFIG"])
            # Use deployment config directly
            service_configs = deployment_config
            logger.info(f"Successfully loaded deployment config: {service_configs}")
            logger.warning(
                "DYN_DEPLOYMENT_CONFIG found in environment - ignoring configuration file and command line arguments"
            )
        except Exception as e:
            logger.warning(f"Failed to parse DYN_DEPLOYMENT_CONFIG: {e}")
    else:
        if config_file:
            with open(config_file) if isinstance(
                config_file, (str, pathlib.Path)
            ) else contextlib.nullcontext(config_file) as f:
                yaml_configs = yaml.safe_load(f)
                logger.debug(f"Loaded config from file: {yaml_configs}")
                # Initialize service_configs as empty dict if it's None
                # Convert nested YAML structure to flat dict with dot notation
                for service, configs in yaml_configs.items():
                    if service not in service_configs:
                        service_configs[service] = {}
                    for key, value in configs.items():
                        service_configs[service][key] = value

        # Process service-specific options
        if args:
            cmdline_overrides = _parse_service_args(args)
            logger.debug(f"Applying command line overrides: {cmdline_overrides}")
            for service, configs in cmdline_overrides.items():
                if service not in service_configs:
                    service_configs[service] = {}
                for key, value in configs.items():
                    service_configs[service][key] = value

    logger.debug(f"Final resolved config: {service_configs}")
    return service_configs
