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

import contextlib
import os
import pathlib
import random
import socket
import typing as t

import click
import psutil
from click import Command, Context


class DynamoCommandGroup(click.Group):
    """Simplified version of BentoMLCommandGroup for Dynamo CLI"""

    def __init__(self, *args: t.Any, **kwargs: t.Any) -> None:
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
    prefix: t.Optional[str] = None,
    max_retry: int = 50,
    enable_so_reuseport: bool = False,
) -> t.Iterator[int]:
    """
    detect free port and reserve until exit the context
    """
    import psutil

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    if enable_so_reuseport:
        if psutil.WINDOWS:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        elif psutil.MACOS or psutil.FREEBSD:
            sock.setsockopt(socket.SOL_SOCKET, 0x10000, 1)  # SO_REUSEPORT_LB
        else:
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


def path_to_uri(path: str) -> str:
    """
    Convert a path to a URI.

    Args:
        path: Path to convert to URI.

    Returns:
        URI string. (quoted, absolute)
    """
    path = os.path.abspath(path)
    if psutil.WINDOWS:
        return pathlib.PureWindowsPath(path).as_uri()
    if psutil.POSIX:
        return pathlib.PurePosixPath(path).as_uri()
    raise ValueError("Unsupported OS")
