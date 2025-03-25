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

import shutil
import subprocess
import sys

import click


def build_run_command() -> click.Group:
    @click.group(name="run")
    def cli():
        pass

    @cli.command(
        context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,
        ),
    )
    def run() -> None:
        """Call dynamo-run with remaining arguments"""
        # Check if dynamo-run is available in PATH
        if shutil.which("dynamo-run") is None:
            click.echo(
                "Error: 'dynamo-run' is needed but not found.\n"
                "Please install it using: cargo install dynamo-run",
                err=True,
            )
            sys.exit(1)

        command = ["dynamo-run"] + sys.argv[2:]
        try:
            subprocess.run(command)
        except Exception as e:
            click.echo(f"Error executing dynamo-run: {str(e)}", err=True)
            sys.exit(1)

    return cli


run_command = build_run_command()
