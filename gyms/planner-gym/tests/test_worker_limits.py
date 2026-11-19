# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from autoscaling_arena.worker_limits import WorkerLimits, resident_workers

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


def test_all_asymmetric_disagg_splits_are_allowed_with_unequal_gpu_costs():
    targets = WorkerLimits(2, 16).targets(
        topology="disagg", gpu_budget=124, prefill_gpus=4, decode_gpus=8
    )
    assert len(targets) == 120
    assert (1, 15) in targets and (15, 1) in targets and (1, 1) in targets
    assert (8, 9) not in targets


def test_gpu_budget_still_constrains_the_total_worker_range():
    targets = WorkerLimits(2, 16).targets(
        topology="disagg", gpu_budget=16, prefill_gpus=8, decode_gpus=8
    )
    assert targets == ((1, 1),)
    agg = WorkerLimits(2, 16).targets(
        topology="agg", gpu_budget=128, prefill_gpus=0, decode_gpus=8
    )
    assert agg == tuple((0, n) for n in range(2, 17))


@pytest.mark.parametrize("bounds", [(True, 16), (17, 16), (0, 16)])
def test_invalid_limits_are_rejected(bounds):
    with pytest.raises(ValueError):
        WorkerLimits(*bounds)


def test_resident_workers_count_starting_and_draining_workers():
    snapshot = {
        "active_decode_ids": [0, 1],
        "starting_decode_ids": [2],
        "draining_decode_ids": [3],
    }
    assert resident_workers(snapshot, "decode") == 4
    snapshot["draining_decode_ids"] = [1]
    with pytest.raises(ValueError, match="Overlapping"):
        resident_workers(snapshot, "decode")
