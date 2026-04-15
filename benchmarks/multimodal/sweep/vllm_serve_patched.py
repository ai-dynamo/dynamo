# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Wrapper for ``vllm serve`` that applies [PERF] monkey-patches first.

Usage:
    python -m benchmarks.multimodal.sweep.vllm_serve_patched serve <model> [args...]
"""

from __future__ import annotations

# Apply patches BEFORE vllm CLI parses/starts anything heavy
import benchmarks.multimodal.sweep.vllm_perf_patches  # noqa: F401

if __name__ == "__main__":
    # Delegate to vllm's CLI (guard prevents re-execution in spawned workers)
    try:
        from vllm.scripts import main
    except ImportError:
        from vllm.entrypoints.cli.main import main  # type: ignore[no-redef]

    main()
