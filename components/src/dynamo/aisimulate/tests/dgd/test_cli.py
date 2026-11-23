# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from dynamo.aisimulate.tests.dgd import cli

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
    pytest.mark.planner,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    ("config_path", "expected"),
    [
        (Path("/workspace/config.yaml"), "config.yaml"),
        (Path("/tmp/config.yaml"), "/tmp/config.yaml"),
    ],
)
def test_run_dgd_recommend_uses_workspace_relative_config(
    monkeypatch, config_path: Path, expected: str
) -> None:
    observed = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(cli, "repository_root", lambda: Path("/workspace"))
    monkeypatch.setattr(cli.subprocess, "run", run)

    command, result = cli.run_dgd_recommend(config_path, Path("/tmp/output"))

    assert result is not None
    assert command[command.index("--config") + 1] == expected
    assert observed["command"] == command
    assert observed["kwargs"]["cwd"] == Path("/workspace")
