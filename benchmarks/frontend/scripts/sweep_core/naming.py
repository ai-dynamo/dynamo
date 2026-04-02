# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run ID and directory naming conventions for sweep runs."""

from __future__ import annotations

from sweep_core.models import AiperfDimension, DeployDimension


def build_run_id(deploy: DeployDimension, aiperf: AiperfDimension) -> str:
    """Build a human-readable run ID from deploy + aiperf dimensions.

    Format: {tokenizer}_c{concurrency}_isl{isl}_w{workers}[_m{models}][_rps{rate}]

    This matches the naming convention from the original sweep_runner.py.
    """
    base = f"{deploy.tokenizer}_c{aiperf.concurrency}_isl{aiperf.isl}_w{deploy.workers}"
    if deploy.num_models > 1:
        base += f"_m{deploy.num_models}"
    if aiperf.request_rate is not None:
        base += f"_rps{aiperf.request_rate}"
    return base
