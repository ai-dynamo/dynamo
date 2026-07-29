# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest

from tests.fault_tolerance.deploy.test_trtllm_cancel_burst_recovery import (
    _assert_fresh_probe_records,
    _inject_debug_fresh_probe_timeout_records,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


def test_debug_fresh_probe_timeout_records_fail_serving_oracle(tmp_path):
    raw_path = tmp_path / "profile_export.jsonl"
    raw_path.write_text(
        json.dumps({"status_code": 200, "metadata": {"was_cancelled": False}}) + "\n"
    )

    _inject_debug_fresh_probe_timeout_records(tmp_path, record_count=3)

    assert (tmp_path / "profile_export.before_debug_injection.jsonl").exists()
    with pytest.raises(
        AssertionError, match="Fresh probe did not make forward progress"
    ):
        _assert_fresh_probe_records(tmp_path, expected_successes=3)
