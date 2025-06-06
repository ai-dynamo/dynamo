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

import typer
from rich.console import Console

console = Console()


def run(ctx: typer.Context):
    """Execute dynamo-run with any additional arguments."""
    # Check if dynamo-run is available in PATH
    if shutil.which("dynamo-run") is None:
        console.print(
            "[bold red]Error:[/bold red] 'dynamo-run' is needed but not found.\n"
            "Please install it using: [bold cyan]cargo install dynamo-run[/bold cyan]",
            style="red",
        )
        raise typer.Exit(code=1)

    # Extract all arguments after 'run'
    args = sys.argv[sys.argv.index("run") + 1 :] if "run" in sys.argv else []

    command = ["dynamo-run"] + args

    try:
        subprocess.run(command)
    except Exception as e:
        console.print(f"[bold red]Error executing dynamo-run:[/bold red] {str(e)}")
        raise typer.Exit(code=1)
