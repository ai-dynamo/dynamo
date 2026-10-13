# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check that single-GPU launchers cannot silently select an unassigned GPU."""

import subprocess
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.multimodal,
    pytest.mark.timeout(10),
]


def _assign_gpu(devices: str | None) -> subprocess.CompletedProcess[str]:
    gpu_utils = Path(__file__).resolve().parents[2] / "examples/common/gpu_utils.sh"
    env = {} if devices is None else {"CUDA_VISIBLE_DEVICES": devices}
    return subprocess.run(
        [
            "bash",
            "-eu",
            "-c",
            'source "$1"; GPU_ID=$(inherited_gpu_id); printf "%s\\n" "$GPU_ID"',
            "bash",
            str(gpu_utils),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=5,
    )


@pytest.mark.parametrize("devices", [None, "", ",1"])
def test_missing_gpu_assignment_fails(devices: str | None) -> None:
    """Stop the caller when no first device is assigned, without printing GPU 0."""
    result = _assign_gpu(devices)
    assert result.returncode != 0
    assert result.stdout == ""
    assert "CUDA_VISIBLE_DEVICES must name a GPU" in result.stderr


@pytest.mark.parametrize(
    ("devices", "expected"),
    [
        ("0", "0"),
        ("1,2", "1"),
        ("GPU-abcdef,GPU-fedcba", "GPU-abcdef"),
        ("MIG-GPU-abcdef/1/0", "MIG-GPU-abcdef/1/0"),
        ("-1", "-1"),
    ],
)
def test_inherited_gpu_assignment_is_preserved(devices: str, expected: str) -> None:
    """Keep the first device token, including UUIDs and the no-device sentinel."""
    result = _assign_gpu(devices)
    assert result.returncode == 0, result.stderr
    assert result.stdout == f"{expected}\n"
