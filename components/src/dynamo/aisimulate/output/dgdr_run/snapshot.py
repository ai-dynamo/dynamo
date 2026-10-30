# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Desired-state snapshot of one DGDRRun, written to the pod-local volume.

The Sweeper container atomically replaces a single YAML file with the complete
*current* projection of one ``DynamoGraphDeploymentRun``; the publisher
sidecar in the same pod consumes it directly (shared ``emptyDir``). Nothing
here involves a ConfigMap, the Dynamo event plane, or any cross-pod RPC.

The snapshot is desired state, not a command list and not history:

* ``run`` carries progress and, once ``terminal`` is true, the final outcome;
* ``candidates`` is the bounded set of evaluated points that should exist as
  ``DynamoGraphDeploymentCandidate`` objects, best-first for scalar searches.

Each candidate has a stable ``id`` that depends only on the evaluated point
(its resolved parameters), never on its list position or rank. Reordering
the list must therefore never change an identity or a rendered manifest; it
only changes the ordered references the publisher writes to the run status.

Each published snapshot is complete. The publisher only needs the latest one,
so intermediate states may be coalesced away, but a terminal snapshot must
always be written (see ``DGDRRunOutputAdapter.close``).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from dynamo.aisimulate.output.dgd.writers.atomic import replace_text

SNAPSHOT_FILE_NAME = "dgdr_run_snapshot.yaml"
SCHEMA_VERSION = 1


class RunPhase(str, Enum):
    """Run lifecycle as reported by the producer. Anything but RUNNING is terminal."""

    RUNNING = "Running"
    SUCCEEDED = "Succeeded"
    FAILED = "Failed"

    @property
    def terminal(self) -> bool:
        return self is not RunPhase.RUNNING


class CandidateOutcome(str, Enum):
    MATERIALIZED = "materialized"
    MATERIALIZATION_FAILED = "materialization_failed"


@dataclass(frozen=True)
class SnapshotCandidate:
    """One evaluated point in the bounded projection.

    ``parameters`` and ``metrics`` are the immutable evaluation facts the
    publisher needs to build the candidate's resource: a DGD manifest alone
    does not identify an evaluated point, because two evaluations can render
    the same DGD under different parameters and still have different metrics.
    """

    id: str
    outcome: CandidateOutcome
    parameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    manifest: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("candidate id must not be empty")
        if self.outcome is CandidateOutcome.MATERIALIZED:
            if not self.manifest:
                raise ValueError("a materialized candidate must carry its manifest")
            if self.error is not None:
                raise ValueError("a materialized candidate must not carry an error")
        elif not self.error:
            raise ValueError("a failed candidate must carry its error")
        elif self.manifest is not None:
            raise ValueError("a failed candidate must not carry a manifest")

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "outcome": self.outcome.value,
            "parameters": self.parameters,
            "metrics": self.metrics,
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class DGDRRunSnapshot:
    """The complete current projection of one run."""

    phase: RunPhase
    round_no: int
    evaluated: int
    candidates: tuple[SnapshotCandidate, ...] = ()
    message: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        run: dict[str, Any] = {
            "phase": self.phase.value,
            "terminal": self.phase.terminal,
        }
        if self.message:
            run["message"] = self.message
        if self.error:
            run["error"] = self.error
        return {
            "schemaVersion": SCHEMA_VERSION,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "run": run,
            "progress": {"round": self.round_no, "evaluated": self.evaluated},
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


def write_snapshot(directory: Path, snapshot: DGDRRunSnapshot) -> Path:
    """Atomically replace the snapshot file; readers never see a partial file."""
    path = directory / SNAPSHOT_FILE_NAME
    replace_text(path, yaml.safe_dump(snapshot.to_dict(), sort_keys=False))
    return path


__all__ = [
    "SCHEMA_VERSION",
    "SNAPSHOT_FILE_NAME",
    "CandidateOutcome",
    "DGDRRunSnapshot",
    "RunPhase",
    "SnapshotCandidate",
    "write_snapshot",
]
