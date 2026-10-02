# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import yaml

from dynamo.aisimulate.output.dgdr.kube_status import (
    STATUS_FILE_NAME,
    CandidateOutcome,
    CandidateStatusEntry,
    SweeperStatusSnapshot,
    SweepRunStatus,
    write_sweeper_status,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def test_materialized_entry_requires_manifest() -> None:
    with pytest.raises(ValueError, match="requires manifest"):
        CandidateStatusEntry(candidate_id="c1", outcome=CandidateOutcome.MATERIALIZED)


def test_failed_entry_requires_error() -> None:
    with pytest.raises(ValueError, match="requires error"):
        CandidateStatusEntry(
            candidate_id="c1", outcome=CandidateOutcome.MATERIALIZATION_FAILED
        )


def test_failed_entry_truncates_an_overlong_error_message() -> None:
    entry = CandidateStatusEntry(
        candidate_id="c1",
        outcome=CandidateOutcome.MATERIALIZATION_FAILED,
        error="x" * 10_000,
    )
    # Bounded regardless of how verbose the renderer's exception text is --
    # this is the ConfigMap-size safeguard, not an incidental limit.
    assert len(entry.error) <= 500
    assert entry.error.endswith("...[truncated]")


def test_failed_entry_leaves_a_short_error_message_untouched() -> None:
    entry = CandidateStatusEntry(
        candidate_id="c1",
        outcome=CandidateOutcome.MATERIALIZATION_FAILED,
        error="renderer output must contain exactly one DynamoGraphDeployment",
    )
    assert entry.error == (
        "renderer output must contain exactly one DynamoGraphDeployment"
    )


def test_snapshot_round_trips_through_yaml(tmp_path) -> None:
    snapshot = SweeperStatusSnapshot(
        status=SweepRunStatus.RUNNING,
        round_no=3,
        cumulative_evaluated=42,
        candidates=(
            CandidateStatusEntry(
                candidate_id="candidate-000001",
                outcome=CandidateOutcome.MATERIALIZED,
                manifest="kind: DynamoGraphDeployment\nmetadata:\n  name: candidate-000001\n",
                path="candidates/candidate-000001.yaml",
            ),
            CandidateStatusEntry(
                candidate_id="candidate-000002",
                outcome=CandidateOutcome.MATERIALIZATION_FAILED,
                error="renderer output must contain exactly one DynamoGraphDeployment",
            ),
        ),
        message="round 3: 2 candidates retained",
    )

    write_sweeper_status(tmp_path, snapshot)

    on_disk = yaml.safe_load((tmp_path / STATUS_FILE_NAME).read_text())
    assert on_disk["status"] == "running"
    assert on_disk["round_no"] == 3
    assert on_disk["cumulative_evaluated"] == 42
    assert on_disk["message"] == "round 3: 2 candidates retained"
    assert "error" not in on_disk  # top-level error only set for a failed run
    assert on_disk["candidates"] == [
        {
            "id": "candidate-000001",
            "outcome": "materialized",
            "manifest": "kind: DynamoGraphDeployment\nmetadata:\n  name: candidate-000001\n",
            "path": "candidates/candidate-000001.yaml",
        },
        {
            "id": "candidate-000002",
            "outcome": "materialization_failed",
            "error": "renderer output must contain exactly one DynamoGraphDeployment",
        },
    ]


def test_snapshot_omits_candidates_key_when_empty(tmp_path) -> None:
    write_sweeper_status(
        tmp_path,
        SweeperStatusSnapshot(
            status=SweepRunStatus.RUNNING, round_no=0, cumulative_evaluated=0
        ),
    )
    on_disk = yaml.safe_load((tmp_path / STATUS_FILE_NAME).read_text())
    assert "candidates" not in on_disk


def test_write_is_atomic_no_partial_file_left_behind(tmp_path) -> None:
    write_sweeper_status(
        tmp_path,
        SweeperStatusSnapshot(
            status=SweepRunStatus.SUCCESS, round_no=5, cumulative_evaluated=80
        ),
    )
    # replace_text's temp file is named .{name}.*.tmp -- none should remain.
    leftovers = list(tmp_path.glob(f".{STATUS_FILE_NAME}.*.tmp"))
    assert leftovers == []
    assert (tmp_path / STATUS_FILE_NAME).exists()
