# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Status-snapshot schema for the in-cluster (DGDR v2) Sweeper status adapter.

Extends the pattern already shipped for DGDR v1
(``dynamo.profiler.utils.profiler_status``): the Sweeper container writes a
single, atomically-replaced YAML file describing *current* state; a thin
polling sidecar (unchanged from v1) relays its contents into a ConfigMap; the
DGDR(v2) controller reconciles ``DGDRRun.status`` and ``DGDC`` objects from
that ConfigMap.

This is deliberately a snapshot, not an event log: each write overwrites the
previous one in full. There is nothing to replay and nothing to resume from
on a sidecar restart -- the next successful poll simply picks up whatever the
file currently says. See the DEP for the reasoning (supersedes #15002,
#15073).

Each materialized candidate's full rendered DGD manifest is carried inline
(``CandidateStatusEntry.manifest``), not just a path reference. A path-only
payload would require the DGDR(v2) controller to read the Sweeper pod's own
local filesystem to resolve ``DynamoGraphDeploymentCandidate.Spec`` content,
which it has no access to -- the controller only ever sees the relayed
ConfigMap. Measured against real rendered DGD manifests (1.3KiB-11.5KiB
depending on agg vs. disagg and node count), inlining at the adapter's
default ``top_n=5`` costs at most ~5.5% of the 1MiB ConfigMap cap even in the
worst observed case (every retained slot a large multi-node disaggregated
deployment); see ``_MAX_TOP_N`` in ``dgdradapter.py`` for the ceiling this
implies at the adapter's upper bound.

Complementary to, not a replacement for, ``dynamo.aisimulate.output.dgd.adapter``
(DEP #14282): that adapter is AISimulate's own output-adapter plugin, called
once after a sweep finishes with the final selected candidate(s). It cannot
report anything if the Sweeper process dies mid-run -- it simply never runs.
This module exists for the complementary in-cluster need: live progress
during the run, and a durable "last known state" if the process crashes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import yaml

from dynamo.aisimulate.output.dgd.writers.atomic import replace_text

STATUS_FILE_NAME = "sweeper_status.yaml"

# Back-of-envelope budget (see the DEP's "size" discussion): a ConfigMap is
# capped at 1MiB by the API server. This snapshot never grows with how many
# candidates have been *evaluated* -- only with how many are *retained*
# (bounded by the adapter's top_n). With manifests now carried inline rather
# than as a path reference, the dominant cost per retained slot is the
# manifest itself (1.3KiB-11.5KiB observed); a materialization-failure
# error message is comparatively small, but still uncapped without this
# limit, so one verbose renderer exception could still dominate an
# otherwise-small snapshot of all-failed candidates.
_MAX_ERROR_LENGTH = 500
_TRUNCATION_SUFFIX = "...[truncated]"


class SweepRunStatus(str, Enum):
    """Overall status of the Sweeper run, mirroring ProfilerStatus."""

    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"


class CandidateOutcome(str, Enum):
    """Outcome of one candidate's materialization attempt."""

    MATERIALIZED = "materialized"
    MATERIALIZATION_FAILED = "materialization_failed"


@dataclass(frozen=True)
class CandidateStatusEntry:
    """One retained candidate's current reporting state.

    ``manifest`` carries the candidate's full rendered DGD manifest inline,
    so the controller can build ``DynamoGraphDeploymentCandidate.Spec``
    straight from the relayed ConfigMap, with no dependency on the Sweeper
    pod's own filesystem. ``path`` is kept alongside it only as a
    human-readable pointer to where the same content also landed on disk
    (relative to the status file's own directory, matching the existing
    ``outputs: {final_config: path}`` convention from ``profiler_status.py``)
    -- it is not required by, and not read by, the controller.
    """

    candidate_id: str
    outcome: CandidateOutcome
    manifest: str | None = None
    path: str | None = None
    error: str | None = None

    def __post_init__(self) -> None:
        if self.outcome is CandidateOutcome.MATERIALIZED and not self.manifest:
            raise ValueError("a materialized candidate entry requires manifest")
        if self.outcome is CandidateOutcome.MATERIALIZATION_FAILED and not self.error:
            raise ValueError("a materialization_failed entry requires error")
        if self.error and len(self.error) > _MAX_ERROR_LENGTH:
            # frozen dataclass: normalize via object.__setattr__ rather than
            # rejecting -- a renderer's full exception text is diagnostic
            # detail, not something the caller should have to pre-truncate.
            truncated = self.error[: _MAX_ERROR_LENGTH - len(_TRUNCATION_SUFFIX)]
            object.__setattr__(self, "error", truncated + _TRUNCATION_SUFFIX)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.candidate_id,
            "outcome": self.outcome.value,
        }
        if self.manifest is not None:
            payload["manifest"] = self.manifest
        if self.path is not None:
            payload["path"] = self.path
        if self.error is not None:
            payload["error"] = self.error
        return payload


@dataclass(frozen=True)
class SweeperStatusSnapshot:
    """The complete current-state payload written to ``sweeper_status.yaml``.

    ``candidates`` holds only the currently retained bounded set (at most
    ``maxCandidates``), not every candidate ever evaluated -- round-level
    search activity beyond that is the two cheap scalars below, not
    manifests. This keeps the relayed ConfigMap small regardless of how long
    the search runs or how many candidates it has rejected.
    """

    status: SweepRunStatus
    round_no: int
    cumulative_evaluated: int
    candidates: tuple[CandidateStatusEntry, ...] = ()
    message: str = ""
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status.value,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "round_no": self.round_no,
            "cumulative_evaluated": self.cumulative_evaluated,
        }
        if self.message:
            payload["message"] = self.message
        if self.error:
            payload["error"] = self.error
        if self.candidates:
            payload["candidates"] = [c.to_dict() for c in self.candidates]
        return payload


def write_sweeper_status(output_dir: Path, snapshot: SweeperStatusSnapshot) -> None:
    """Atomically replace ``sweeper_status.yaml`` with the current snapshot."""
    replace_text(
        output_dir / STATUS_FILE_NAME,
        yaml.safe_dump(snapshot.to_dict(), sort_keys=False),
    )


__all__ = [
    "STATUS_FILE_NAME",
    "CandidateOutcome",
    "CandidateStatusEntry",
    "SweepRunStatus",
    "SweeperStatusSnapshot",
    "write_sweeper_status",
]
