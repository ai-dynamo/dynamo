# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Give an Open MPI 5 singleton a hostname short enough for MPI_Comm_spawn.

TensorRT-LLM starts its executor ranks with MPI.COMM_SELF.Spawn from a
singleton. Open MPI 5.0.10rc2 cannot do that when the hostname length plus the
number of digits in the process ID exceeds 37: the singleton's name,
"singleton.<hostname>.<pid>.0", goes to the prte daemon through a 50-byte
buffer, gets cut short, and Spawn raises MPI_ERR_UNKNOWN. Kubernetes pod names
are often longer than that.

PMIx takes the hostname from PMIX_HOSTNAME when it is set. So a process that no
MPI launcher started (no PMIX_NAMESPACE), on a host whose name is longer than
30 characters, gets a short name derived from the real one. 30 plus the 7
digits of the largest process ID is 37. Launched ranks and a PMIX_HOSTNAME that
is already set are left alone.

dynamo.trtllm applies this in its package __init__, before anything imports
tensorrt_llm (whose import runs MPI_Init), so the rule holds in every image that
runs dynamo.trtllm, not only images built from trtllm_runtime.Dockerfile.

This is the in-package copy of container/deps/trtllm/_dynamo_pmix_hostname.py,
which covers other Python processes in the arm64 runtime image. Keep the two
rules identical. Remove both, along with this module's tests and the import and
call in __init__.py, when the image's Open MPI has the upstream fix,
open-mpi/ompi#14398 (v5.0.x backport: open-mpi/ompi#14409). Open MPI 5.0.11
and HPC-X v2.50 do not have it.
"""

import os
import zlib

_MAX_SAFE_HOSTNAME = 30


def short_pmix_hostname(hostname: str, *, launched: bool, preset: bool) -> str | None:
    """Return the PMIX_HOSTNAME to set, or None to leave the environment alone.

    launched: an MPI launcher started this process (PMIX_NAMESPACE is set).
    preset: PMIX_HOSTNAME is already set.
    """
    if launched or preset or len(hostname) <= _MAX_SAFE_HOSTNAME:
        return None
    return f"pmix-{zlib.crc32(hostname.encode()):08x}"


def apply_short_pmix_hostname() -> None:
    """Set PMIX_HOSTNAME in os.environ when short_pmix_hostname asks for it."""
    name = short_pmix_hostname(
        os.uname().nodename,
        launched="PMIX_NAMESPACE" in os.environ,
        preset="PMIX_HOSTNAME" in os.environ,
    )
    if name is not None:
        os.environ["PMIX_HOSTNAME"] = name
