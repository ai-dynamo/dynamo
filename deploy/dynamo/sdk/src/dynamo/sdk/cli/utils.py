# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import click
import difflib
import functools
import typing as t
from click import Command, Context, HelpFormatter, UsageError

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