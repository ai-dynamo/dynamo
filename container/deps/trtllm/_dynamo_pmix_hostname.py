# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Give an Open MPI 5 singleton a hostname short enough for MPI_Comm_spawn.

trtllm_runtime.Dockerfile installs this module, and a .pth line that imports
it when the interpreter starts, only where it selects Open MPI 5 (arm64).

Open MPI 5.0.10rc2 cannot spawn from a singleton when the hostname length
plus the number of digits in the process ID exceeds 37. Open MPI passes the
singleton's name, "singleton.<hostname>.<pid>.0", to the prte daemon through a
50-byte print buffer. A longer name is cut short, prte registers the wrong
name, and MPI.COMM_SELF.Spawn raises MPI_ERR_UNKNOWN. TRT-LLM starts its
workers that way, and Kubernetes pod names are often longer than that.

PMIx takes the hostname from PMIX_HOSTNAME when that is set. So a process that
no MPI launcher started, on a host whose name is longer than 30 characters,
gets a short name derived from the real one: 30 plus the 7 digits of the
largest process ID is 37. A process that mpirun or prte started carries
PMIX_NAMESPACE and a PMIX_HOSTNAME of its own, and is left alone, so ranks that
mpirun or prte start keep their real hostnames.

A PMIX_HOSTNAME that is already set is left alone, even when it is too long.

This runs in every Python process, so it imports nothing beyond os unless it
has to set the name.

Remove this module, its install step in trtllm_runtime.Dockerfile and its
tests in tests/dependencies/test_trtllm_mpi.py when the image's Open MPI has
the upstream fix, open-mpi/ompi#14398 (v5.0.x backport: open-mpi/ompi#14409).
Open MPI 5.0.11 and HPC-X v2.50 do not have it. Tracked in
NVIDIA/TensorRT-LLM#19607.
"""

import os

_MAX_SAFE_HOSTNAME = 30


def short_pmix_hostname(hostname: str, *, launched: bool, preset: bool) -> str | None:
    """Return the PMIX_HOSTNAME to set, or None to leave the environment alone.

    launched: an MPI launcher started this process (PMIX_NAMESPACE is set).
    preset: PMIX_HOSTNAME is already set.
    """
    if launched or preset or len(hostname) <= _MAX_SAFE_HOSTNAME:
        return None
    import zlib

    return f"pmix-{zlib.crc32(hostname.encode()):08x}"


_name = short_pmix_hostname(
    os.uname().nodename,
    launched="PMIX_NAMESPACE" in os.environ,
    preset="PMIX_HOSTNAME" in os.environ,
)
if _name is not None:
    os.environ["PMIX_HOSTNAME"] = _name
