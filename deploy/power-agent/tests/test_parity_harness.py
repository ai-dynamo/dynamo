# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for the GPU-less logic of the e2e parity harness.

`e2e_actuator_parity.py` is a script (not collected as tests and normally run
only on a real dual-actuator GPU rig), but two pieces of its logic are pure and
must be pinned without hardware:

  1. Ground truth is read via ``nvidia-smi -i <UUID>`` — NOT by an
     actuator-native index. DCGM and NVML index spaces can differ, so an
     index-based read could sample a different physical GPU than the actuator
     probed and report phantom parity failures.
  2. ``parity_check`` joins NVML and DCGM results BY UUID, not by positional
     ``zip`` — so the two actuators enumerating the same GPUs under different
     index orders does not create false diffs, and a UUID present on only one
     path is surfaced as a failure.
"""

import os
import sys
import unittest
from unittest.mock import patch

# `e2e_actuator_parity.py` is a sibling script in this tests/ dir; ensure the
# directory is importable regardless of pytest's import mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2e_actuator_parity as parity  # noqa: E402


def _gpu(
    uuid,
    idx,
    *,
    min_w=100,
    max_w=700,
    apply_cap_returned=None,
    ns_after_apply=None,
    ns_after_restore=None,
    ns_default=700.0,
):
    """Build a probe-result GPU dict in the shape `parity_check` consumes."""
    return {
        "idx": idx,
        "uuid": uuid,
        "min_w": min_w,
        "max_w": max_w,
        "pid_count": 0,
        "ns_before": ns_default,
        "ns_default": ns_default,
        "apply_cap_returned": apply_cap_returned,
        "ns_after_apply": ns_after_apply,
        "ns_after_restore": ns_after_restore,
        "write_skipped_reason": "read-only mode",
    }


class TestNvidiaSmiQueriesByUuid(unittest.TestCase):
    def test_power_limit_passes_uuid_to_dash_i(self):
        with patch("subprocess.check_output", return_value="250.5\n") as co:
            val = parity.nvidia_smi_power_limit("GPU-abc123")
        self.assertEqual(val, 250.5)
        argv = co.call_args.args[0]
        self.assertEqual(argv[0], "nvidia-smi")
        self.assertIn("-i", argv)
        # The token right after -i is the UUID, not an index.
        self.assertEqual(argv[argv.index("-i") + 1], "GPU-abc123")
        self.assertIn("--query-gpu=power.limit", argv)

    def test_default_limit_passes_uuid_to_dash_i(self):
        with patch("subprocess.check_output", return_value="700\n") as co:
            val = parity.nvidia_smi_default_limit("GPU-xyz")
        self.assertEqual(val, 700.0)
        argv = co.call_args.args[0]
        self.assertEqual(argv[argv.index("-i") + 1], "GPU-xyz")
        self.assertIn("--query-gpu=power.default_limit", argv)


class TestParityCheckJoinsByUuid(unittest.TestCase):
    def test_reordered_indices_same_uuids_is_parity(self):
        """The actuators enumerate the SAME GPUs under SWAPPED indices. A
        positional zip would compare A-vs-B and fail; a UUID join passes."""
        nvml = {
            "device_count": 2,
            "gpus": [_gpu("GPU-A", 0), _gpu("GPU-B", 1)],
        }
        dcgm = {
            "device_count": 2,
            "gpus": [_gpu("GPU-B", 0), _gpu("GPU-A", 1)],  # swapped index order
        }
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 0)

    def test_uuid_present_on_one_side_only_is_failure(self):
        nvml = {"device_count": 2, "gpus": [_gpu("GPU-A", 0), _gpu("GPU-B", 1)]}
        dcgm = {"device_count": 2, "gpus": [_gpu("GPU-A", 0), _gpu("GPU-C", 1)]}
        # B (nvml-only) and C (dcgm-only) each fail.
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 2)

    def test_value_diff_on_matched_uuid_is_failure(self):
        nvml = {"device_count": 1, "gpus": [_gpu("GPU-A", 0, max_w=700)]}
        dcgm = {"device_count": 1, "gpus": [_gpu("GPU-A", 1, max_w=650)]}
        # Same UUID, max_w differs by 50 W (> tolerance) → one failure.
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 1)

    def test_device_count_mismatch_bails_before_join(self):
        nvml = {"device_count": 2, "gpus": [_gpu("GPU-A", 0), _gpu("GPU-B", 1)]}
        dcgm = {"device_count": 1, "gpus": [_gpu("GPU-A", 0)]}
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 1)


if __name__ == "__main__":
    unittest.main()
