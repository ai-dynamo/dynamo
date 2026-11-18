# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend benchmark report reads Tokio health from non-deprecated metrics."""

from __future__ import annotations

import pytest

from benchmarks.frontend.scripts.analysis.create_report import section_tokio
from benchmarks.frontend.scripts.analysis.parsing_util import parse_prometheus_text

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]

# The deprecated stall counter (99) and busy ratios (saturated) disagree with
# the replacement metrics on purpose, so a revert to them fails the test.
_EXPOSITION = """\
dynamo_frontend_event_loop_delay_seconds_bucket{le="0"} 0
dynamo_frontend_event_loop_delay_seconds_bucket{le="0.001"} 40
dynamo_frontend_event_loop_delay_seconds_bucket{le="0.005"} 45
dynamo_frontend_event_loop_delay_seconds_bucket{le="0.01"} 47
dynamo_frontend_event_loop_delay_seconds_bucket{le="+Inf"} 48
dynamo_frontend_event_loop_delay_seconds_sum 0.2
dynamo_frontend_event_loop_delay_seconds_count 48
dynamo_frontend_event_loop_stall_total 99
dynamo_tokio_worker_busy_ratio{worker="0"} 1000
dynamo_tokio_worker_busy_ratio{worker="1"} 1000
dynamo_tokio_worker_mean_poll_time_ns{worker="0"} 20000
dynamo_tokio_worker_mean_poll_time_ns{worker="1"} 30000
"""


def test_tokio_section_uses_replacement_metrics(tmp_path):
    path = tmp_path / "final_snapshot.txt"
    path.write_text(_EXPOSITION)

    snap = parse_prometheus_text(path)

    assert snap is not None
    assert snap.tokio_event_loop_stall_total == 3  # 48 - 45
    assert sorted(snap.tokio_worker_mean_poll_time_ns) == [20000, 30000]

    section = section_tokio(snap)
    assert section is not None
    assert "Worker Poll Time" in section
    assert "| Event loop stalls (timer delay > 5 ms) | 3 |" in section
    for word in ("Busy Ratio", "busy ratio", "idle", "hot", "saturated"):
        assert word not in section
