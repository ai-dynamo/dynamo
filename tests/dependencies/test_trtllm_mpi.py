# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guards for the Open MPI that TRT-LLM images run on.

trtllm_runtime.Dockerfile links /opt/dynamo/mpi to Open MPI 4 on amd64, where
Open MPI 5 failed to spawn TRT-LLM's workers in CI, and to Open MPI 5 on arm64,
where the base image's libtorch needs a symbol only Open MPI 5 defines. It points
PATH, LD_LIBRARY_PATH and OPAL_PREFIX at the link and edits the selected tree's
etc/openmpi-mca-params.conf. Where it selects Open MPI 5, it also installs a
startup hook (_dynamo_pmix_hostname) that shortens PMIX_HOSTNAME on hosts whose
names are too long for an Open MPI 5 singleton to spawn. Each test checks one
part of that in the built image, so no part can regress while the others stay
green.
"""

import contextlib
import importlib.util
import os
import platform
import shutil
import signal
import subprocess
import sys

import pytest

pytestmark = [
    # Let the 60s subprocess timeouts report their failure before pytest interrupts.
    pytest.mark.timeout(90),
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None
        or importlib.util.find_spec("mpi4py") is None,
        reason="TRT-LLM images only (the selection is in trtllm_runtime.Dockerfile)",
    ),
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only"),
]

_MPI_LINK = "/opt/dynamo/mpi"

# Keep in step with the /opt/dynamo/mpi RUN in trtllm_runtime.Dockerfile.
_EXPECTED_MPI = {"x86_64": "/opt/hpcx/ompi4", "aarch64": "/opt/hpcx/ompi5"}

# Installed with a .pth line only where the image selects Open MPI 5. Delete the
# hook tests together with the hook (see trtllm_runtime.Dockerfile).
_HOSTNAME_HOOK = "_dynamo_pmix_hostname"

# Variables a running MPI singleton exports to its environment. A child that
# inherits them joins the parent's MPI job instead of starting its own.
_RUNTIME_PREFIXES = ("PMIX_", "OMPI_MCA_orte_", "OMPI_MCA_ess", "OMPI_COMM_WORLD_")

_LOADED_LIBMPI = (
    "from mpi4py import MPI\n"
    "print(next(line.split()[-1] for line in open('/proc/self/maps')"
    " if 'libmpi.so' in line))\n"
)

# How TRT-LLM's MpiPoolSession starts its workers: a singleton MPI.COMM_SELF.Spawn.
_SPAWN = (
    "import sys\n"
    "from mpi4py import MPI\n"
    "child = ('from mpi4py import MPI; parent = MPI.Comm.Get_parent(); '\n"
    "         'parent.send(7, dest=0); parent.Disconnect()')\n"
    "comm = MPI.COMM_SELF.Spawn(sys.executable, args=['-c', child], maxprocs=1)\n"
    "print('spawned child sent', comm.recv(source=0))\n"
    "comm.Disconnect()\n"
)

_ALLREDUCE = "from mpi4py import MPI; print('allreduce', MPI.COMM_WORLD.allreduce(1))"

# Start from every CPU the cgroup allows, so a caller that is already pinned
# (for example a pytest worker that imported tensorrt_llm) cannot hide the bind.
_AFFINITY = (
    "import os\n"
    "os.sched_setaffinity(0, range(os.cpu_count()))\n"
    "before = len(os.sched_getaffinity(0))\n"
    "from mpi4py import MPI\n"
    "after = sorted(os.sched_getaffinity(0))\n"
    "print(before, len(after), after[:8])\n"
)


def _run(args: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
    """Run `args` in its own session, and kill the whole group on timeout.

    Killing the group, not only `args[0]`, keeps a hung spawn from leaving its
    child or Open MPI's helper processes running after the test.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(_RUNTIME_PREFIXES)}
    proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        try:
            _, stderr = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            stderr = "(unavailable: a process outside the group still holds the pipe)"
        pytest.fail(
            f"{args[0]} did not finish in {timeout}s; stderr tail:\n{stderr[-2000:]}"
        )
    return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)


def test_selected_mpi_matches_the_architecture():
    expected = _EXPECTED_MPI.get(platform.machine())
    if expected is None:
        pytest.skip(f"no Open MPI selection is defined for {platform.machine()}")
    assert os.path.realpath(_MPI_LINK) == expected, (
        f"{_MPI_LINK} resolves to {os.path.realpath(_MPI_LINK)}, "
        f"expected {expected} on {platform.machine()}"
    )
    mpirun = shutil.which("mpirun")
    assert mpirun and os.path.realpath(mpirun).startswith(expected + "/"), (
        f"mpirun on PATH is {mpirun} -> {mpirun and os.path.realpath(mpirun)}, "
        f"not from {expected}"
    )


def test_mpi4py_loads_the_selected_mpi():
    proc = _run([sys.executable, "-c", _LOADED_LIBMPI])
    assert proc.returncode == 0, proc.stderr[-2000:]
    loaded = os.path.realpath(proc.stdout.strip().splitlines()[-1])
    selected = os.path.realpath(_MPI_LINK)
    assert loaded.startswith(selected + "/"), (
        f"mpi4py loaded {loaded}, not the Open MPI under {_MPI_LINK} ({selected}); "
        "check LD_LIBRARY_PATH in trtllm_runtime.Dockerfile"
    )


def test_comm_self_spawn_reaches_its_child():
    """A singleton spawn, the way TRT-LLM starts its workers.

    `_run` drops PMIX_* from the child's environment, and the child's own
    interpreter applies the PMIX_HOSTNAME hook again at startup. So on a host
    whose name is too long for Open MPI 5, this also checks the hook.
    """
    proc = _run([sys.executable, "-c", _SPAWN])
    assert proc.returncode == 0 and "spawned child sent 7" in proc.stdout, (
        f"MPI.COMM_SELF.Spawn failed (exit {proc.returncode}), so TRT-LLM "
        f"cannot start its workers; stderr tail:\n{proc.stderr[-2000:]}"
    )


def test_pmix_hostname_hook_follows_the_selection():
    installed = importlib.util.find_spec(_HOSTNAME_HOOK) is not None
    selected = os.path.realpath(_MPI_LINK)
    assert installed == selected.endswith("/ompi5"), (
        f"{_HOSTNAME_HOOK} installed={installed}, but {_MPI_LINK} is {selected}; "
        "the hook belongs exactly where trtllm_runtime.Dockerfile selects Open MPI 5"
    )


@pytest.mark.parametrize(
    ("hostname", "launched", "preset", "shortened"),
    [
        ("h" * 46, False, False, True),
        ("h" * 31, False, False, True),
        ("h" * 30, False, False, False),
        ("h" * 46, True, False, False),
        ("h" * 46, False, True, False),
    ],
    ids=["long", "just-over", "at-limit", "launched-rank", "already-set"],
)
def test_pmix_hostname_hook_rule(hostname, launched, preset, shortened):
    hook = pytest.importorskip(_HOSTNAME_HOOK, reason="installed with Open MPI 5 only")
    name = hook.short_pmix_hostname(hostname, launched=launched, preset=preset)
    if not shortened:
        assert name is None
        return
    # 13 characters plus the 7 digits of the largest pid stays within 37.
    assert name is not None and name.startswith("pmix-") and len(name) == 13
    again = hook.short_pmix_hostname(hostname, launched=False, preset=False)
    other = hook.short_pmix_hostname(hostname + "x", launched=False, preset=False)
    assert name == again, "not deterministic"
    assert name != other, "two hostnames map to one name"


def test_two_ranks_connect_over_ob1():
    """The operator's multi-node launch forces the ob1 PML.

    ob1 needs a byte transport besides `self` to reach another rank, which
    `btl = self` in etc/openmpi-mca-params.conf takes away.
    """
    mpirun = shutil.which("mpirun")
    assert mpirun, "mpirun is not on PATH"
    root = ["--allow-run-as-root"] if os.geteuid() == 0 else []
    args = [mpirun, *root, "--oversubscribe", "--mca", "pml", "ob1", "-n", "2"]
    proc = _run([*args, sys.executable, "-c", _ALLREDUCE])
    assert proc.stdout.count("allreduce 2") == 2, (
        f"two ob1 ranks did not connect (exit {proc.returncode}); "
        f"stderr tail:\n{proc.stderr[-2000:]}"
    )


@pytest.mark.skipif(
    not hasattr(os, "sched_getaffinity"), reason="needs sched_getaffinity"
)
def test_mpi_init_keeps_cpu_affinity():
    """MPI_Init must not pin the process to one core.

    `import tensorrt_llm` runs MPI_Init (tensorrt_llm/_utils.py imports
    mpi4py.MPI). With `hwloc_base_binding_policy = core`, that singleton
    MPI_Init binds the process to one core, and every thread and child process
    it starts inherits the mask.
    """
    proc = _run([sys.executable, "-c", _AFFINITY])
    assert proc.returncode == 0, proc.stderr[-2000:]
    # MPI can print warnings to stdout; the probe's own line is the last one.
    before, after, cpus = proc.stdout.strip().splitlines()[-1].split(maxsplit=2)
    if int(before) < 4:
        pytest.skip(f"only {before} CPUs available; a one-core bind is not observable")
    assert int(after) == int(before), (
        f"MPI_Init narrowed CPU affinity from {before} to {after} CPUs {cpus}; "
        f"check hwloc_base_binding_policy in {_MPI_LINK}/etc/openmpi-mca-params.conf "
        "(see trtllm_runtime.Dockerfile)"
    )
