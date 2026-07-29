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

    parser.add_argument(
        "--no-operation",
        action="store_true",
        default=False,
        dest="no_operation",
        help="Log incoming scale requests without executing them (useful for testing the e2e flow without actual K8s scaling)",
    )

    parser.add_argument(
        "--max-total-gpus",
        type=int,
        default=-1,
        dest="max_total_gpus",
        help="Maximum total GPUs across all managed pools. Requests that would exceed this limit are rejected. 0 means no GPU scaling is allowed. -1 (default) disables enforcement entirely.",
    )

    parser.add_argument(
        "--min-total-gpus",
        type=int,
        default=-1,
        dest="min_total_gpus",
        help="Minimum total GPUs across all managed pools. Scale-down requests that would drop below this floor are denied unless a pending scale-up on another pool can be paired with them. -1 (default) disables the floor.",
    )

    parser.add_argument(
        "--intent-cache-ttl-seconds",
        type=float,
        default=360.0,
        dest="intent_cache_ttl_seconds",
        help="Cached scale-intent from a pool is considered fresh for this many seconds (default: 360). This should be at least 2x the local planner's slowest tick interval so opposite-direction intents can overlap. Default throughput-based scaling ticks every 180s, so 360 covers two ticks.",
    )

    return parser
