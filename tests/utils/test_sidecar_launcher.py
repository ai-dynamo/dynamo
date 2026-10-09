# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize("mode", ["path", "python", "pinned", "missing", "empty"])
def test_sidecar_resolution_preserves_native_selection(
    tmp_path: Path, mode: str
) -> None:
    # Regression: a missing CI artifact or failing native command must not
    # silently switch execution to Python bindings or another command on PATH.
    launch_utils = Path(__file__).parents[2] / "examples/common/launch_utils.sh"
    native = tmp_path / "dynamo-vllm-sidecar"
    python = tmp_path / "python3"
    pinned = tmp_path / "pinned binary"
    for path, label, code in (
        (native, "native", 17),
        (python, "python", 19),
        (pinned, "pinned", 23),
    ):
        path.write_text(f"#!/bin/bash\nprintf '%s\\0' {label} \"$@\"\nexit {code}\n")
        path.chmod(0o755)

    env = {"PATH": str(tmp_path)}
    expected_code = 17
    expected_args = ["native", "argument with spaces"]
    if mode == "python":
        native.unlink()
        expected_code = 19
        expected_args = [
            "python",
            "-m",
            "dynamo.vllm.sidecar",
            "argument with spaces",
        ]
    elif mode == "pinned":
        env["DYNAMO_SIDECAR_BIN"] = str(pinned)
        expected_code = 23
        expected_args = ["pinned", "argument with spaces"]
    elif mode in ("missing", "empty"):
        env["DYNAMO_SIDECAR_BIN"] = (
            str(tmp_path / "missing") if mode == "missing" else ""
        )
        expected_code = 1
        expected_args = []

    result = subprocess.run(
        [
            "/bin/bash",
            "-e",
            "-c",
            'source "$1"; resolve_sidecar vllm SIDECAR_CMD; '
            'exec "${SIDECAR_CMD[@]}" "argument with spaces"',
            "sidecar-test",
            str(launch_utils),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )
    assert result.returncode == expected_code, result.stderr
    assert result.stdout.split("\0")[:-1] == expected_args, result
    if mode == "python":
        assert "WARNING:" in result.stderr, result.stderr
