# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
CLI entry point for the multimodal benchmark sweep.

Usage:
    python -m benchmarks.multimodal.sweep --config experiment.yaml
    python -m benchmarks.multimodal.sweep --config experiment.yaml --output-dir /tmp/results
    python -m benchmarks.multimodal.sweep --config experiment.yaml --model MyModel --osl 200
"""

from __future__ import annotations

from .args import parse_args
from .config import load_config, resolve_repo_root
from .orchestrator import run_sweep


def main(argv=None) -> None:
    args = parse_args(argv)

    overrides = {k: v for k, v in vars(args).items() if k != "config" and v is not None}

    config = load_config(args.config, cli_overrides=overrides or None)

    repo_root = resolve_repo_root()
    config.validate(repo_root=repo_root)

    run_sweep(config, repo_root=repo_root)


if __name__ == "__main__":
    main()
