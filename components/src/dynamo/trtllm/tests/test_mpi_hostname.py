# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PMIX_HOSTNAME rule that dynamo.trtllm applies at import."""

import os
import subprocess
import sys

import pytest

from dynamo.trtllm.mpi_hostname import short_pmix_hostname

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

_LONG = "trtllm-agg-example-trtllmworker-e34726f8-7d9c5b6f4-x2k9q"


@pytest.mark.parametrize(
    "hostname, launched, preset, shortened",
    [
        pytest.param("h" * 31, False, False, True, id="31-chars"),
        pytest.param("h" * 30, False, False, False, id="30-chars"),
        pytest.param(_LONG, True, False, False, id="launched-rank"),
        pytest.param(_LONG, False, True, False, id="preset"),
    ],
)
def test_short_pmix_hostname_rule(hostname, launched, preset, shortened):
    name = short_pmix_hostname(hostname, launched=launched, preset=preset)
    if not shortened:
        assert name is None
        return
    assert name is not None
    assert name.startswith("pmix-")
    assert len(name) == 13
    # Same host, same name, so every process on a pod agrees on it.
    assert name == short_pmix_hostname(hostname, launched=launched, preset=preset)


def test_short_pmix_hostname_differs_per_host():
    a = short_pmix_hostname(_LONG, launched=False, preset=False)
    b = short_pmix_hostname(_LONG + "z", launched=False, preset=False)
    assert a != b


def test_importing_dynamo_trtllm_sets_short_pmix_hostname():
    # A fresh interpreter, so the package __init__ really runs, with a fake
    # long nodename installed before the import.
    code = (
        "import os\n"
        "class _U:\n"
        f"    nodename = {_LONG!r}\n"
        "os.uname = lambda: _U()\n"
        # Drop any name a startup .pth hook (the arm64 image) already set.
        "os.environ.pop('PMIX_HOSTNAME', None)\n"
        "import dynamo.trtllm\n"
        "print(os.environ.get('PMIX_HOSTNAME', ''))\n"
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("PMIX_")}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout.strip()
    assert out == short_pmix_hostname(_LONG, launched=False, preset=False)
