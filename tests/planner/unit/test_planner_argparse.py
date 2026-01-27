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


def test_parser_default_mode():
    """Test parser sets default planner mode to local."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(["--namespace", "test-ns"])

    assert args.planner_mode == "local"


def test_parser_delegating_mode():
    """Test parser accepts delegating mode arguments."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        [
            "--namespace",
            "test-ns",
            "--planner-mode",
            "delegating",
            "--global-planner-namespace",
            "global-ns",
        ]
    )

    assert args.planner_mode == "delegating"
    assert args.global_planner_namespace == "global-ns"
    assert args.global_planner_component == "GlobalPlanner"  # default


def test_parser_custom_global_component():
    """Test parser accepts custom GlobalPlanner component name."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        [
            "--namespace",
            "test-ns",
            "--planner-mode",
            "delegating",
            "--global-planner-namespace",
            "global-ns",
            "--global-planner-component",
            "CustomGlobalPlanner",
        ]
    )

    assert args.global_planner_component == "CustomGlobalPlanner"


def test_validate_delegating_mode_without_namespace():
    """Test validation fails for delegating mode without GlobalPlanner namespace."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(["--namespace", "test-ns", "--planner-mode", "delegating"])

    with pytest.raises(ValueError, match="global-planner-namespace required"):
        validate_planner_args(args)


def test_validate_delegating_mode_success():
    """Test validation succeeds for delegating mode with namespace."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        [
            "--namespace",
            "test-ns",
            "--planner-mode",
            "delegating",
            "--global-planner-namespace",
            "global-ns",
        ]
    )

    # Should not raise
    validate_planner_args(args)


def test_validate_local_mode():
    """Test validation succeeds for local mode without extra args."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(["--namespace", "test-ns", "--planner-mode", "local"])

    # Should not raise
    validate_planner_args(args)


def test_parser_invalid_mode():
    """Test parser rejects invalid planner mode."""
    parser = create_sla_planner_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--namespace", "test-ns", "--planner-mode", "invalid-mode"])


def test_parser_all_existing_args_still_work():
    """Test that existing planner arguments still work with new mode args."""
    parser = create_sla_planner_parser()
    args = parser.parse_args(
        [
            "--namespace",
            "test-ns",
            "--planner-mode",
            "local",
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
    assert args.planner_mode == "local"
    assert args.backend == "vllm"
    assert args.environment == "kubernetes"
    assert args.ttft == 200
    assert args.itl == 50
    assert args.max_gpu_budget == 16
    assert args.adjustment_interval == 60
