# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard: MPI_Init must not pin a TRT-LLM process to one core.

`import tensorrt_llm` runs MPI_Init (tensorrt_llm/_utils.py imports mpi4py.MPI).
When the selected Open MPI's etc/openmpi-mca-params.conf says
`hwloc_base_binding_policy = core`, that singleton MPI_Init binds the calling
process to one core, and every thread and child process it starts inherits the
mask. trtllm_runtime.Dockerfile edits that file for the Open MPI it selects
(/opt/dynamo/mpi); this test keeps a base-image bump or a change of selection
from silently bringing the binding back.
"""

import importlib.util
import os
import subprocess
import sys

import pytest

pytestmark = [
    # Let the 30s subprocess timeout report its failure before pytest interrupts.
    pytest.mark.timeout(60),
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None
        or importlib.util.find_spec("mpi4py") is None,
        reason="TRT-LLM images only (the binding comes from their Open MPI)",
    ),
    pytest.mark.skipif(not hasattr(os, "sched_getaffinity"), reason="Linux only"),
]

# Start from every CPU the cgroup allows, so a caller that is already pinned
# (for example a pytest worker that imported tensorrt_llm) cannot hide the bind.
_PROBE = (
    "import os\n"
    "os.sched_setaffinity(0, range(os.cpu_count()))\n"
    "before = len(os.sched_getaffinity(0))\n"
    "from mpi4py import MPI\n"
    "after = sorted(os.sched_getaffinity(0))\n"
    "print(before, len(after), after[:8])\n"
)

# Variables a running MPI singleton exports to its environment. A child that
# inherits them joins the parent's MPI job instead of starting its own.
_RUNTIME_PREFIXES = ("PMIX_", "OMPI_MCA_orte_", "OMPI_MCA_ess", "OMPI_COMM_WORLD_")


def test_mpi_init_keeps_cpu_affinity():
    env = {k: v for k, v in os.environ.items() if not k.startswith(_RUNTIME_PREFIXES)}
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    # MPI can print warnings to stdout; the probe's own line is the last one.
    before, after, cpus = proc.stdout.strip().splitlines()[-1].split(maxsplit=2)
    if int(before) < 4:
        pytest.skip(f"only {before} CPUs available; a one-core bind is not observable")
    assert int(after) == int(before), (
        f"MPI_Init narrowed CPU affinity from {before} to {after} CPUs {cpus}; "
        "check hwloc_base_binding_policy in "
        "/opt/dynamo/mpi/etc/openmpi-mca-params.conf "
        "(see trtllm_runtime.Dockerfile)"
    )
