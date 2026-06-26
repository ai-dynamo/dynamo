# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
SweepExecutor protocol -- the run-level extensibility seam.

Each executor implements this protocol. The orchestrator calls these methods
without knowing whether runs execute locally, in k8s, or elsewhere.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

from sweep_core.models import DeployDimension, RunResult, RunSpec, SweepConfig


@runtime_checkable
class SweepExecutor(Protocol):
    """Protocol for sweep executors."""

    def prepare(self, config: SweepConfig) -> None:
        """One-time setup before the sweep begins (e.g., start infra)."""
        ...

    def apply_deploy(
        self,
        deploy: DeployDimension,
        prev: Optional[DeployDimension],
    ) -> None:
        """Apply a deployment change (e.g., restart frontend, switch backend).

        Args:
            deploy: The deployment configuration to apply.
            prev: The previous deployment configuration (None for first run).
        """
        ...

    def execute_run(self, run_spec: RunSpec, run_dir: Path) -> RunResult:
        """Execute a single run and return results.

        Args:
            run_spec: The run specification.
            run_dir: Directory where artifacts should be written.

        Returns:
            RunResult with status and metrics.
        """
        ...

    def cleanup(self) -> None:
        """Cleanup after the sweep completes (e.g., stop infra)."""
        ...
