#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

# Runs before any dynamo.trtllm submodule, and so before tensorrt_llm's import
# calls MPI_Init. See mpi_hostname.py.
from .mpi_hostname import apply_short_pmix_hostname

apply_short_pmix_hostname()

try:
    from ._version import __version__
except Exception:
    try:
        from importlib.metadata import version as _pkg_version

        __version__ = _pkg_version("ai-dynamo")
    except Exception:
        __version__ = "0.0.0+unknown"
