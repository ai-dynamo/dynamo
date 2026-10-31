# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
import yaml

from dynamo.aisimulate.output.dgdr_run.snapshot import (
    SCHEMA_VERSION,
    SNAPSHOT_FILE_NAME,
    CandidateOutcome,
    DGDRRunSnapshot,
    RunPhase,
    SnapshotCandidate,
    write_snapshot,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def materialized(candidate_id: str = "evaluated-point-a") -> SnapshotCandidate:
    return SnapshotCandidate(
        id=candidate_id,
        outcome=CandidateOutcome.MATERIALIZED,
        parameters={"tp": 4},
        metrics={"score": 1.5, "usedGpus": 4},
        manifest="kind: DynamoGraphDeployment\n",
    )


def test_wire_format_of_a_running_snapshot() -> None:
    payload = DGDRRunSnapshot(
        phase=RunPhase.RUNNING,
        round_no=7,
        evaluated=48,
        candidates=(materialized(),),
    ).to_dict()

    assert payload["schemaVersion"] == SCHEMA_VERSION
    assert payload["run"] == {"phase": "Running", "terminal": False}
    assert payload["progress"] == {"round": 7, "evaluated": 48}
    assert payload["candidates"] == [
        {
            "id": "evaluated-point-a",
            "outcome": "materialized",
            "parameters": {"tp": 4},
            "metrics": {"score": 1.5, "usedGpus": 4},
            "manifest": "kind: DynamoGraphDeployment\n",
        }
    ]
    assert payload["timestamp"].endswith("Z")


def test_terminal_snapshot_carries_the_outcome() -> None:
    payload = DGDRRunSnapshot(
        phase=RunPhase.FAILED,
        round_no=2,
        evaluated=9,
        message="stopped",
        error="OOM",
    ).to_dict()
    assert payload["run"] == {
        "phase": "Failed",
        "terminal": True,
        "message": "stopped",
        "error": "OOM",
    }
    assert payload["candidates"] == []


def test_a_failed_candidate_carries_an_error_and_no_manifest() -> None:
    failed = SnapshotCandidate(
        id="evaluated-point-b",
        outcome=CandidateOutcome.MATERIALIZATION_FAILED,
        error="cannot render",
    )
    assert failed.to_dict() == {
        "id": "evaluated-point-b",
        "outcome": "materialization_failed",
        "parameters": {},
        "metrics": {},
        "error": "cannot render",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"id": "", "outcome": CandidateOutcome.MATERIALIZED, "manifest": "x"},
        {"id": "a", "outcome": CandidateOutcome.MATERIALIZED},
        {
            "id": "a",
            "outcome": CandidateOutcome.MATERIALIZED,
            "manifest": "x",
            "error": "y",
        },
        {"id": "a", "outcome": CandidateOutcome.MATERIALIZATION_FAILED},
        {
            "id": "a",
            "outcome": CandidateOutcome.MATERIALIZATION_FAILED,
            "error": "y",
            "manifest": "x",
        },
    ],
)
def test_inconsistent_candidates_are_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        SnapshotCandidate(**kwargs)


def test_write_replaces_the_file_atomically_and_leaves_no_temporary_files(
    tmp_path: Path,
) -> None:
    for round_no in range(2):
        path = write_snapshot(
            tmp_path,
            DGDRRunSnapshot(phase=RunPhase.RUNNING, round_no=round_no, evaluated=0),
        )
        assert path == tmp_path / SNAPSHOT_FILE_NAME
        assert yaml.safe_load(path.read_text())["progress"]["round"] == round_no

    assert [p.name for p in tmp_path.iterdir()] == [SNAPSHOT_FILE_NAME]


def test_the_manifest_survives_a_yaml_round_trip_byte_for_byte(
    tmp_path: Path,
) -> None:
    manifest = "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\nspec:\n  a: 'x: y'\n"
    candidate = SnapshotCandidate(
        id="evaluated-point-c",
        outcome=CandidateOutcome.MATERIALIZED,
        manifest=manifest,
    )
    path = write_snapshot(
        tmp_path,
        DGDRRunSnapshot(
            phase=RunPhase.RUNNING, round_no=0, evaluated=1, candidates=(candidate,)
        ),
    )
    (loaded,) = yaml.safe_load(path.read_text())["candidates"]
    assert loaded["manifest"] == manifest
