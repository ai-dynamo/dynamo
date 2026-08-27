# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the prioritized Dynamo RL product-gap register."""

from __future__ import annotations

import check_rl_product_gaps
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]


def _registry() -> dict:
    return check_rl_product_gaps.load_registry(check_rl_product_gaps.DEFAULT_REGISTRY)


def test_repository_gap_register_is_issue_ready() -> None:
    assert check_rl_product_gaps.validate_registry(_registry()) == []


def test_required_source_text_drift_is_detected() -> None:
    registry = _registry()
    registry["gaps"][0]["source_assertions"][0]["contains"].append(
        "field_that_does_not_exist"
    )
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any("no longer contains 'field_that_does_not_exist'" in item for item in findings)


def test_gap_closing_source_signal_expires_the_register() -> None:
    registry = _registry()
    registry["gaps"][0]["source_assertions"][0]["not_contains"].append(
        "pub session_id: String"
    )
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any("now contains gap-closing signal 'pub session_id: String'" in item for item in findings)


def test_documented_boundary_drift_is_detected() -> None:
    registry = _registry()
    registry["gaps"][2]["affected_docs"][0]["contains"] = [
        "boundary that is not documented"
    ]
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any("no longer contains 'boundary that is not documented'" in item for item in findings)


def test_gap_ids_must_be_unique() -> None:
    registry = _registry()
    registry["gaps"][1]["id"] = registry["gaps"][0]["id"]
    findings = check_rl_product_gaps.validate_registry(registry)
    assert "gap IDs must be unique" in findings


def test_unknown_dependency_is_rejected() -> None:
    registry = _registry()
    registry["gaps"][3]["depends_on"].append("DYN-RL-GAP-999")
    registry["gaps"][3]["depends_on"].sort()
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any("references unknown gap DYN-RL-GAP-999" in item for item in findings)


def test_dependency_cycle_is_rejected() -> None:
    registry = _registry()
    registry["gaps"][0]["depends_on"] = ["DYN-RL-GAP-002"]
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any(item.startswith("dependency cycle:") for item in findings)


def test_priority_cannot_outrun_a_dependency() -> None:
    registry = _registry()
    registry["gaps"][0]["depends_on"] = ["DYN-RL-GAP-004"]
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any("cannot have higher urgency than dependency DYN-RL-GAP-004" in item for item in findings)


def test_acceptance_evidence_must_be_actionable() -> None:
    registry = _registry()
    registry["gaps"][2]["acceptance_evidence"] = ["one vague check"]
    findings = check_rl_product_gaps.validate_registry(registry)
    assert "gaps[2].acceptance_evidence must contain at least 4 items" in findings


def test_closed_loop_prerequisites_must_match_the_gap_graph() -> None:
    registry = _registry()
    registry["closed_loop_decision"]["prerequisite_gap_ids"].pop()
    findings = check_rl_product_gaps.validate_registry(registry)
    assert (
        "closed_loop_decision prerequisites must equal DYN-RL-GAP-005 dependencies"
        in findings
    )


def test_closed_loop_package_owner_cannot_be_preassigned() -> None:
    registry = _registry()
    registry["closed_loop_decision"]["package_owner"] = "DynoSim"
    findings = check_rl_product_gaps.validate_registry(registry)
    assert (
        "closed_loop_decision.package_owner must remain unassigned until DEP approval"
        in findings
    )


def test_registry_baseline_must_match_the_evidence_manifest() -> None:
    registry = _registry()
    registry["baseline_dynamo_commit"] = "0" * 40
    findings = check_rl_product_gaps.validate_registry(registry)
    assert (
        "baseline_dynamo_commit must match rl_evidence.json baseline.dynamo_commit"
        in findings
    )


def test_source_paths_must_be_repository_relative() -> None:
    registry = _registry()
    registry["gaps"][0]["source_assertions"][0]["path"] = "../outside.rs"
    findings = check_rl_product_gaps.validate_registry(registry)
    assert any("path must be repository-relative" in item for item in findings)


def test_cli_passes_the_checked_registry() -> None:
    assert check_rl_product_gaps.main([]) == 0
