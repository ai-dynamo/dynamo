# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.parallel,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]


def test_public_mocker_module_cli_is_available() -> None:
    assert importlib.util.find_spec("dynamo.mocker.__main__") is not None


@pytest.mark.parametrize("option", ["--help", "--version"])
def test_public_mocker_module_cli_executes(option: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dynamo.mocker", option],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout
