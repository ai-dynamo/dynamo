# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The package must not drag `torch` in behind the lightweight media loaders."""

from __future__ import annotations

import subprocess
import sys

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

# A fresh interpreter is the only honest way to ask this: once any earlier test
# has imported torch, an eager import here would go unnoticed.
_PROBE = """
import sys
from dynamo.common.multimodal import {names}
print("torch" in sys.modules)
"""


def _torch_imported_by(names: str) -> bool:
    result = subprocess.run(
        [sys.executable, "-c", _PROBE.format(names=names)],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip() == "True"


@pytest.mark.parametrize("name", ["ImageLoader", "AudioLoader", "VideoLoader"])
def test_media_loader_import_does_not_pull_torch(name: str) -> None:
    assert not _torch_imported_by(
        name
    ), f"importing {name} pulled in torch; the loaders must stay usable without the ML stack"


def test_heavy_members_still_import() -> None:
    assert _torch_imported_by(
        "AsyncEncoderCache"
    ), "AsyncEncoderCache should still resolve, and it legitimately needs torch"
