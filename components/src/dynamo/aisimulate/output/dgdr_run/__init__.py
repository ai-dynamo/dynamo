# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Live DynamoGraphDeploymentRun output adapter for AISimulate (``dgdr_run``)."""

from dynamo.aisimulate.output.dgdr_run.adapter import (
    DGDRRunOutputAdapter,
    DGDRRunOutputConfig,
    DGDRRunOutputPlugin,
    candidate_id_for,
    create_adapter,
)
from dynamo.aisimulate.output.dgdr_run.snapshot import (
    SNAPSHOT_FILE_NAME,
    CandidateOutcome,
    DGDRRunSnapshot,
    RunPhase,
    SnapshotCandidate,
    write_snapshot,
)

__all__ = [
    "SNAPSHOT_FILE_NAME",
    "CandidateOutcome",
    "DGDRRunOutputAdapter",
    "DGDRRunOutputConfig",
    "DGDRRunOutputPlugin",
    "DGDRRunSnapshot",
    "RunPhase",
    "SnapshotCandidate",
    "candidate_id_for",
    "create_adapter",
    "write_snapshot",
]
