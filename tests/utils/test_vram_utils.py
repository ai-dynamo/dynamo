# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from tests.utils.vram_utils import (
    DEFAULT_GPU_PARALLEL_PROCESS_CAP,
    auto_worker_count,
    print_gpu_plan,
)


def _gpu(total_gib: int = 22) -> dict:
    return {"index": 0, "name": "Test GPU", "total_mib": total_gib * 1024}


def test_auto_worker_count_uses_bounded_process_cap_for_mixed_profiles():
    assert (
        auto_worker_count([_gpu()], 24, test_profiled_gibs=[0, 3.7, 14.2])
        == DEFAULT_GPU_PARALLEL_PROCESS_CAP
    )


def test_auto_worker_count_respects_explicit_process_cap():
    assert (
        auto_worker_count(
            [_gpu()],
            24,
            test_profiled_gibs=[0, 3.7, 14.2],
            max_process_slots=12,
        )
        == 12
    )


def test_auto_worker_count_is_bounded_across_multiple_gpus():
    gpus = [_gpu(), {"index": 1, "name": "Test GPU 2", "total_mib": 22 * 1024}]

    assert auto_worker_count(gpus, 24, test_profiled_gibs=[0, 3.7]) == 16


def test_auto_worker_count_falls_back_without_usable_vram_metadata():
    assert auto_worker_count([], 24) == 1
    assert auto_worker_count([_gpu()], 0) == 1


def test_print_gpu_plan_describes_process_cap_and_vram_gates(capsys):
    print_gpu_plan([_gpu()], 24, [("zero", 0), ("gpu", 3.7)])

    out = capsys.readouterr().out
    assert "-n auto  : up to 16 process slots" in out
    assert "VRAM gates still apply per GPU" in out
    assert "Smallest nonzero profiled test: 3.7 GiB" in out
