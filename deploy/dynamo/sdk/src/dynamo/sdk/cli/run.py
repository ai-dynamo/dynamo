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

import sys
import subprocess
import click


def build_run_command() -> click.Group:
    from bentoml_cli.utils import BentoMLCommandGroup
    
    @click.group(name="run", cls=BentoMLCommandGroup)
    def cli():
        pass

    @cli.command(
        context_settings=dict(
            ignore_unknown_options=True,
            allow_extra_args=True,
        ),
    )
    def run() -> None:
        """Call dynamo-run under the hood to execute whatever options passed in"""
        command = ["dynamo-run"] + sys.argv[2:]
        subprocess.run(command)
       
    return cli


run_command = build_run_command()
