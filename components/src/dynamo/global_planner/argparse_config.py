# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argument parsing for GlobalPlanner."""

import argparse


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
  # Simple deployment (accept all namespaces)
  DYN_NAMESPACE=global-infra python -m dynamo.global_planner

  # With authorization
  DYN_NAMESPACE=global-infra python -m dynamo.global_planner \\
    --managed-namespaces app-ns-1 app-ns-2 app-ns-3

  # Custom environment
  DYN_NAMESPACE=global-infra python -m dynamo.global_planner \\
    --environment=kubernetes
        """,
    )

    parser.add_argument(
        "--managed-namespaces",
        type=str,
        nargs="+",
        default=None,
        help="Optional: List of namespaces authorized to use this GlobalPlanner (default: accept all)",
    )

    parser.add_argument(
        "--environment",
        default="kubernetes",
        choices=["kubernetes"],
        help="Environment type (currently only kubernetes supported)",
    )

    return parser
