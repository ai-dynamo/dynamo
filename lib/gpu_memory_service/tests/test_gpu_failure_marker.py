# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import stat

from gpu_memory_service.common.gpu_failure_marker import (
    gpu_failure_marker_path,
    publish_gpu_failure_marker,
    read_gpu_failure_marker,
)


def test_failure_marker_is_private_and_first_report_wins(tmp_path):
    cohort = tmp_path / "cohort"
    cohort.touch(mode=0o600)

    marker = publish_gpu_failure_marker(
        cohort,
        rank=7,
        pid=1234,
        source="signal-11",
    )

    assert stat.S_IMODE(marker.stat().st_mode) == 0o600
    assert read_gpu_failure_marker(marker) == (7, 1234, "signal-11")

    publish_gpu_failure_marker(cohort, rank=3, pid=5678, source="later-report")
    assert read_gpu_failure_marker(marker) == (7, 1234, "signal-11")


def test_incomplete_or_malformed_marker_is_not_actionable(tmp_path):
    marker = gpu_failure_marker_path(tmp_path / "cohort")
    marker.write_text("")
    assert read_gpu_failure_marker(marker) is None
    marker.write_text("not a valid marker")
    assert read_gpu_failure_marker(marker) is None
