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

import os
import re

from typer.testing import CliRunner

from dynamo.sdk.cli.cli import cli

runner = CliRunner()


def test_app():
    print(os.getcwd())
    result = runner.invoke(
        cli,
        [
            "serve",
            "pipeline:Frontend",
            "--working-dir",
            "deploy/dynamo/sdk/src/dynamo/sdk/tests",
            "--Frontend.model=qwentastic",
            "--Middle.bias=0.5",
            "--dry-run",
        ],
    )
    print(result.output)

    # Remove color codes printed by rich - TODO add a CI/Test flag that doesnt use rich
    clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)

    # Assert successful exit code
    assert result.exit_code == 0

    # With clean output we can do more precise checks
    assert "Service Configuration:" in clean_output
    assert '"Frontend": {' in clean_output
    assert '"model": "qwentastic"' in clean_output
    assert '"Middle": {' in clean_output
    assert '"bias": 0.5' in clean_output
    assert "Environment Variable that would be set:" in clean_output
    assert "DYNAMO_SERVICE_CONFIG=" in clean_output
    assert (
        '{"Frontend": {"model": "qwentastic"}, "Middle": {"bias": 0.5}}' in clean_output
    )
