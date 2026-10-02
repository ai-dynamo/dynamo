# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DynamoGraphDeploymentRequest(v2) Sweeper live-status reporting, owned by Dynamo.

Reports live Sweeper progress (``on_round``/``on_candidate``) as a snapshot
YAML file, complementary to the pull-once ``DGDOutputAdapter`` in
``dynamo.aisimulate.output.dgd``. Renders candidates through that package's
``render_dgd`` and writes through its ``writers.atomic`` module, so the two
adapters share one rendering/writing implementation even though they live
in separate packages.
"""

from dynamo.aisimulate.output.dgdr.dgdradapter import DGDRAdapter
from dynamo.aisimulate.output.dgdr.kube_status import (
    STATUS_FILE_NAME,
    CandidateOutcome,
    CandidateStatusEntry,
    SweeperStatusSnapshot,
    SweepRunStatus,
    write_sweeper_status,
)

__all__ = [
    "STATUS_FILE_NAME",
    "CandidateOutcome",
    "CandidateStatusEntry",
    "DGDRAdapter",
    "SweeperStatusSnapshot",
    "SweepRunStatus",
    "write_sweeper_status",
]
