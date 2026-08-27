# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the independent RL clean-room review publication gate."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import check_rl_clean_room_record
import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

TEMPLATE = Path(__file__).resolve().parents[1] / "rl_clean_room_record.template.json"


def _template() -> dict:
    return check_rl_clean_room_record.load_record(TEMPLATE)


def _passing_record() -> dict:
    record = _template()
    record.update(
        {
            "record_state": "passed",
            "review_id": "verl-clean-room-2026-08-27-1",
            "last_validated": "2026-08-27T16:00:00Z",
        }
    )
    record["scope"].update(
        {
            "guide_path": "docs/fern/pages/use-cases/reinforcement-learning/verl.md",
            "framework_name": "verl",
            "maturity_target": "experimental",
            "integration_artifact": "https://github.com/verl-project/verl-recipe/tree/"
            + "a" * 40,
            "recipe_commit": "a" * 40,
            "core_commit": "b" * 40,
            "dynamo_commit": "c" * 40,
        }
    )
    for name, linked in record["linked_records"].items():
        checker = (
            "check_rl_validation_record.py"
            if name == "framework_validation"
            else "check_rl_program_record.py"
        )
        linked.update(
            {
                "record_id": f"{name}-record-1",
                "uri": f"artifact://clean-room/linked/{name}.json",
                "sha256": "d" * 64,
                "checker_command": f"python3 docs/fern/scripts/{checker} record.json --publication-gate",
                "checker_output_artifact": f"artifact://clean-room/linked/{name}-checker.txt",
                "publication_gate_passed": True,
            }
        )
    record["reviewer"].update(
        {
            "name": "Independent Reviewer",
            "github": "@independent-reviewer",
            "organization": "docs-team",
            "independence_attested": True,
        }
    )
    for index, (role, owner) in enumerate(record["owners"].items(), start=1):
        owner.update(
            {
                "name": f"Accepted Owner {index}",
                "github": f"@{role}-owner",
                "accepted": True,
            }
        )
    record["environment"].update(
        {
            "fresh_workspace": True,
            "base_image": "registry.example/dynamo-rl:v1",
            "base_image_digest": "sha256:" + "e" * 64,
            "model_name": "example/model",
            "model_revision": "model-revision",
            "hardware_summary": "one node with eight H100 SXM GPUs and NVLink",
            "preexisting_dependencies": ["CUDA driver 590.00"],
        }
    )
    record["run"].update(
        {
            "started_at": "2026-08-27T10:00:00Z",
            "completed_at": "2026-08-27T15:00:00Z",
            "entry_page": "docs/fern/pages/use-cases/reinforcement-learning/verl.md",
            "navigation_clicks": 1,
            "commands_executed": [
                "git clone https://github.com/verl-project/verl-recipe.git",
                "python3 -m recipe.dynamo.main_dynamo ...",
            ],
            "undocumented_steps": [],
            "artifact_root": "artifact://clean-room/",
        }
    )
    for gate_name, gate in record["journey"].items():
        gate.update(
            {
                "status": "passed",
                "conclusion": f"{gate_name} completed from the documented path",
                "artifacts": [f"artifact://clean-room/journey/{gate_name}.json"],
            }
        )
    record["findings"] = [
        {
            "id": "DOC-1",
            "severity": "minor",
            "status": "resolved",
            "description": "One expected-output label was initially ambiguous.",
            "resolution": "The guide now names the exact success marker.",
            "owner": "Accepted Owner 1",
            "artifact": "artifact://clean-room/findings/DOC-1.json",
        }
    ]
    record["broken_links"].update(
        {
            "command": "fern docs broken-links",
            "rl_errors": 0,
            "unrelated_errors": 4,
            "baseline_decision": "waived_with_owner",
            "waiver_owner": "repository-docs-owner",
            "waiver_expires_at": "2026-09-27T00:00:00Z",
            "artifact": "artifact://clean-room/broken-links.txt",
        }
    )
    record["decision"].update(
        {
            "outcome": "approved",
            "summary": "The pinned experimental verl guide was usable without undocumented steps.",
            "limitations": [
                "The unrelated Qwen link baseline remains under a dated waiver."
            ],
            "signed_at": "2026-08-27T15:30:00Z",
            "artifact": "artifact://clean-room/decision.json",
        }
    )
    return record


def test_planned_template_has_valid_structure_but_cannot_publish() -> None:
    record = _template()
    assert check_rl_clean_room_record.validate_structure(record) == []
    findings = check_rl_clean_room_record.publication_findings(record)
    assert "record_state must be passed for publication" in findings
    assert (
        "linked_records.framework_validation.publication_gate_passed must be true"
        in findings
    )
    assert "journey.navigation_and_pin.status must be passed" in findings


def test_complete_record_passes_the_publication_gate() -> None:
    assert check_rl_clean_room_record.publication_findings(_passing_record()) == []


@pytest.mark.parametrize(
    "mutation,expected",
    [
        (
            lambda record: record["linked_records"]["framework_validation"].update(
                {"publication_gate_passed": False}
            ),
            "framework_validation.publication_gate_passed must be true",
        ),
        (
            lambda record: record["linked_records"]["program_evidence"].update(
                {"sha256": "unpinned"}
            ),
            "program_evidence.sha256 must be a full lowercase digest",
        ),
        (
            lambda record: record["reviewer"].update(
                {
                    "name": record["owners"]["program_dri"]["name"],
                    "github": record["owners"]["program_dri"]["github"],
                }
            ),
            "reviewer must be independent of owners.program_dri",
        ),
        (
            lambda record: record["owners"]["routing"].update({"accepted": False}),
            "owners.routing.accepted must be true",
        ),
        (
            lambda record: record["run"].update({"navigation_clicks": 3}),
            "navigation_clicks must be between 0 and 2",
        ),
        (
            lambda record: record["run"].update(
                {"undocumented_steps": ["apply a local patch"]}
            ),
            "run.undocumented_steps must be empty",
        ),
        (
            lambda record: record["journey"]["generation_and_training"].update(
                {"status": "failed"}
            ),
            "journey.generation_and_training.status must be passed",
        ),
        (
            lambda record: record["findings"][0].update(
                {"severity": "major", "status": "open"}
            ),
            "findings[0].status must not be open",
        ),
        (
            lambda record: record["findings"][0].update(
                {"severity": "major", "status": "waived"}
            ),
            "blocking/major finding must be resolved",
        ),
        (
            lambda record: record["broken_links"].update({"rl_errors": 1}),
            "broken_links.rl_errors must be zero",
        ),
        (
            lambda record: record["broken_links"].update(
                {"baseline_decision": "resolved"}
            ),
            "baseline_decision must be waived_with_owner",
        ),
        (
            lambda record: record["broken_links"].update(
                {"waiver_expires_at": "2026-08-27T15:00:00Z"}
            ),
            "broken_links waiver must expire after decision.signed_at",
        ),
        (
            lambda record: record["run"].update({"started_at": "2026-08-27T10:00:00"}),
            "run.started_at must be ISO-8601 with a UTC offset",
        ),
        (
            lambda record: record["decision"].update({"outcome": "changes_required"}),
            "decision.outcome must be approved",
        ),
    ],
)
def test_publication_gate_rejects_incomplete_or_nonindependent_review(
    mutation, expected: str
) -> None:
    record = copy.deepcopy(_passing_record())
    mutation(record)
    findings = check_rl_clean_room_record.publication_findings(record)
    assert any(expected in finding for finding in findings)


def test_cli_distinguishes_structure_from_publication(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    planned = tmp_path / "planned.json"
    planned.write_text(json.dumps(_template()), encoding="utf-8")
    assert check_rl_clean_room_record.main([str(planned)]) == 0
    assert check_rl_clean_room_record.main([str(planned), "--publication-gate"]) == 1
    captured = capsys.readouterr()
    assert "RL clean-room review passed (structure" in captured.out
    assert "record_state must be passed for publication" in captured.err
