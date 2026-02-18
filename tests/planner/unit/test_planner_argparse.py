# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for planner argument parsing and validation."""

import pytest

from dynamo.planner.utils.planner_argparse import (
    create_sla_planner_parser,
    validate_planner_args,
)

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.planner,
]


def test_parser_global_planner_mode():
    """Test parser accepts global-planner environment mode arguments."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        [
            "--namespace",
            "test-ns",
            "--environment",
            "global-planner",
            "--global-planner-namespace",
            "global-ns",
        ]
    )

    assert args.environment == "global-planner"
    assert args.global_planner_namespace == "global-ns"


def test_validate_global_planner_mode_without_namespace():
    """Test validation fails for global-planner environment without GlobalPlanner namespace."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        ["--namespace", "test-ns", "--environment", "global-planner"]
    )

    with pytest.raises(ValueError, match="global-planner-namespace required"):
        validate_planner_args(args)


def test_parser_invalid_environment():
    """Test parser rejects invalid environment."""
    parser = create_sla_planner_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["--namespace", "test-ns", "--environment", "invalid-environment"]
        )


def test_parser_all_existing_args_still_work():
    """Test that existing planner arguments still work."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        [
            "--namespace",
            "test-ns",
            "--backend",
            "vllm",
            "--environment",
            "kubernetes",
            "--ttft",
            "200",
            "--itl",
            "50",
            "--max-gpu-budget",
            "16",
            "--adjustment-interval",
            "60",
        ]
    )

    assert args.namespace == "test-ns"
    assert args.backend == "vllm"
    assert args.environment == "kubernetes"
    assert args.ttft == 200
    assert args.itl == 50
    assert args.max_gpu_budget == 16
    assert args.adjustment_interval == 60
