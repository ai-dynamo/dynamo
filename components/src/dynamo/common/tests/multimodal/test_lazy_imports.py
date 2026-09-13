# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The package must not drag `torch` in behind the lightweight media loaders."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]

# A fresh interpreter is the only honest way to ask this: once any earlier test
# has imported torch, an eager import here would go unnoticed.
#
# The answer is tagged and parsed off its own line rather than read from the
# whole of stdout. Imports here are free to print (a CuPy fallback notice from
# `dynamo.nixl_connect` is one that really happens), and a bare
# `stdout.strip() == "True"` would read that extra output as False, turning the
# regression this test exists to catch into a pass.
_PROBE = """
import sys
from dynamo.common.multimodal import {names}
print("RESULT:" + str("torch" in sys.modules))
"""


def _torch_imported_by(names: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(names=names)],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Checked here rather than with check=True: that raises CalledProcessError,
    # whose message names the command and the exit status only. The reason the
    # child died is in stderr, which pytest never shows, so a missing torch or
    # PIL or a raise during package import reads as a bare non-zero exit.
    assert (
        result.returncode == 0
    ), f"probe for {names} exited {result.returncode}\nstderr:\n{result.stderr}"
    answers = [
        line.removeprefix("RESULT:")
        for line in result.stdout.splitlines()
        if line.startswith("RESULT:")
    ]
    assert len(answers) == 1, (
        f"probe for {names} did not report exactly one result; "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert answers[0] in ("True", "False"), f"unexpected probe answer {answers[0]!r}"
    return answers[0] == "True"


@pytest.mark.parametrize("name", ["ImageLoader", "AudioLoader", "VideoLoader"])
def test_media_loader_import_does_not_pull_torch(name: str) -> None:
    assert not _torch_imported_by(
        name
    ), f"importing {name} pulled in torch; the loaders must stay usable without the ML stack"


def test_heavy_members_still_import() -> None:
    assert _torch_imported_by(
        "AsyncEncoderCache"
    ), "AsyncEncoderCache should still resolve, and it legitimately needs torch"
