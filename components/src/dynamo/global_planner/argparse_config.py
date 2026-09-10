# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for GlobalPlanner.

The component takes a single ``--config`` argument -- an inline JSON string or a
path to a JSON/YAML file -- matching how ``dynamo.planner`` is configured. Every
setting lives in :class:`~dynamo.global_planner.config.GlobalPlannerConfig` and
has exactly one place it can be set, so there is no flag-versus-file precedence
to reason about.
"""

import argparse

from dynamo.global_planner.config import GlobalPlannerConfig


def create_global_planner_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser for GlobalPlanner.

    Returns:
        argparse.ArgumentParser: Configured argument parser for GlobalPlanner
    """
    parser = argparse.ArgumentParser(
        description="GlobalPlanner - Centralized Scaling Execution Service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Defaults: accept all namespaces, no GPU budget
  DYN_NAMESPACE=global-infra python -m dynamo.global_planner

  # From a config file
  DYN_NAMESPACE=global-infra python -m dynamo.global_planner \\
    --config /etc/global-planner/config.yaml

  # From an inline JSON string
  DYN_NAMESPACE=global-infra python -m dynamo.global_planner \\
    --config '{"min_total_gpus": 16, "max_total_gpus": 16}'
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Inline JSON string or path to a JSON/YAML file holding GlobalPlanner "
            "configuration. Omit to run with defaults."
        ),
    )

    return parser


def resolve_config(args: argparse.Namespace) -> GlobalPlannerConfig:
    """Build a validated config from parsed arguments.

    Returns the default configuration when no ``--config`` was supplied.
    """
    config_arg = getattr(args, "config", None)
    if not config_arg:
        return GlobalPlannerConfig()
    return GlobalPlannerConfig.from_config_arg(config_arg)
