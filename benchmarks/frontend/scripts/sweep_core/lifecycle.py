# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle management -- deploy-dimension delta detection and reset strategy."""

from __future__ import annotations

from typing import Optional

from sweep_core.models import IsolationPolicy, RunSpec


def needs_deploy_or_reset(
    current: RunSpec,
    previous: Optional[RunSpec],
    isolation_policy: IsolationPolicy,
) -> bool:
    """Determine if the current run needs a deploy/reset before execution.

    Args:
        current: The run about to execute.
        previous: The run that just completed (None for the first run).
        isolation_policy: The sweep-level isolation policy.

    Returns:
        True if a deploy/reset is needed before this run.
    """
    if previous is None:
        # First run always needs deployment
        return True

    if isolation_policy == "fresh_per_run":
        # Every run gets its own deploy/reset cycle
        return True

    # reuse_by_deploy_key: only reset when the deploy key changes
    return current.deploy_key != previous.deploy_key


def deploy_key_changed(
    current: RunSpec,
    previous: Optional[RunSpec],
) -> bool:
    """Check if the deploy key has changed between consecutive runs."""
    if previous is None:
        return True
    return current.deploy_key != previous.deploy_key
