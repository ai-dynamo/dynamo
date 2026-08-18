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
from datetime import datetime, timezone
from unittest.mock import patch

# `e2e_actuator_parity.py` is a sibling script in this tests/ dir; ensure the
# directory is importable regardless of pytest's import mode.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import e2e_actuator_parity as parity  # noqa: E402
from actuator import ApplyResult  # noqa: E402


def _gpu(
    uuid,
    idx,
    *,
    min_w=100,
    max_w=700,
    apply_result=None,
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
        "apply_result": apply_result,
        "ns_after_apply": ns_after_apply,
        "ns_after_restore": ns_after_restore,
        "ns_after_cleanup": ns_default,
        "write_skipped_reason": "read-only mode",
    }


class _FakeActuator:
    name = "nvml"

    def __init__(self, states):
        self.states = dict(states)
        self.defaults = {uuid: 700.0 for uuid in states}
        self.uuids = list(states)
        self.apply_calls = []
        self.shutdown_called = False

    def init(self):
        pass

    def shutdown(self):
        self.shutdown_called = True

    def device_count(self):
        return len(self.uuids)

    def get_uuid(self, idx):
        return self.uuids[idx]

    def constraints_w(self, idx):
        return 100, 700

    def list_running_pids(self, idx, expected_uuid=None):
        return []

    def apply_cap(self, idx, watts, expected_uuid=None):
        uuid = self.uuids[idx]
        self.states[uuid] = float(watts)
        self.apply_calls.append((uuid, watts))
        return ApplyResult(
            gpu_uuid=uuid,
            requested_watts=watts,
            target_watts=watts,
            constraint_min_watts=100,
            constraint_max_watts=700,
            policy_outcome="annotated",
            write_outcome="succeeded",
            readback_outcome="succeeded",
            enforced_cap_watts=watts,
            actuator=self.name,
            observed_at=datetime.now(timezone.utc),
        )

    def restore_default(self, idx):
        uuid = self.uuids[idx]
        self.states[uuid] = self.defaults[uuid]
        return True


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


class TestSingleActuatorQualification(unittest.TestCase):
    @staticmethod
    def _result(**overrides):
        values = {
            "gpu_uuid": "GPU-A",
            "requested_watts": 300,
            "target_watts": 300,
            "constraint_min_watts": 100,
            "constraint_max_watts": 700,
            "policy_outcome": "annotated",
            "write_outcome": "succeeded",
            "readback_outcome": "succeeded",
            "enforced_cap_watts": 300,
            "actuator": "nvml",
            "observed_at": datetime.now(timezone.utc),
        }
        values.update(overrides)
        return ApplyResult(**values)

    def test_nvml_only_requires_typed_write_and_independent_readback(self):
        result = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=self._result(),
                    ns_after_apply=300,
                    ns_after_restore=700,
                )
            ],
        }

        self.assertEqual(
            parity.actuator_check(
                result,
                actuator_name="nvml",
                tolerance_w=2.0,
                require_writes=True,
                expected_request_watts=300,
            ),
            0,
        )

    def test_nvml_only_failed_write_cannot_false_pass(self):
        result = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=self._result(
                        write_outcome="failed",
                        readback_outcome="skipped",
                        enforced_cap_watts=None,
                    ),
                    ns_after_apply=700,
                    ns_after_restore=700,
                )
            ],
        }

        self.assertGreater(
            parity.actuator_check(
                result,
                actuator_name="nvml",
                tolerance_w=2.0,
                require_writes=True,
                expected_request_watts=300,
            ),
            0,
        )

    def test_nvml_only_skipped_write_cannot_false_pass(self):
        result = {"device_count": 1, "gpus": [_gpu("GPU-A", 0)]}

        self.assertGreater(
            parity.actuator_check(
                result,
                actuator_name="nvml",
                tolerance_w=2.0,
                require_writes=True,
                expected_request_watts=300,
            ),
            0,
        )

    def test_nvml_only_read_only_discovery_remains_available(self):
        result = {"device_count": 1, "gpus": [_gpu("GPU-A", 0)]}

        self.assertEqual(
            parity.actuator_check(
                result,
                actuator_name="nvml",
                tolerance_w=2.0,
            ),
            0,
        )


class TestParityCheckJoinsByUuid(unittest.TestCase):
    @staticmethod
    def _apply_result(actuator: str) -> ApplyResult:
        return ApplyResult(
            gpu_uuid="GPU-A",
            requested_watts=300,
            target_watts=300,
            constraint_min_watts=100,
            constraint_max_watts=700,
            policy_outcome="annotated",
            write_outcome="succeeded",
            readback_outcome="succeeded",
            enforced_cap_watts=300,
            actuator=actuator,
            observed_at=datetime.now(timezone.utc),
        )

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

    def test_constraint_drift_inside_readback_tolerance_is_failure(self):
        nvml = {"device_count": 1, "gpus": [_gpu("GPU-A", 0, max_w=700)]}
        dcgm = {"device_count": 1, "gpus": [_gpu("GPU-A", 1, max_w=699)]}
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 1)

    def test_device_count_mismatch_bails_before_join(self):
        nvml = {"device_count": 2, "gpus": [_gpu("GPU-A", 0), _gpu("GPU-B", 1)]}
        dcgm = {"device_count": 1, "gpus": [_gpu("GPU-A", 0)]}
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 1)

    def test_typed_apply_evidence_is_compared_without_dataclass_arithmetic(self):
        nvml = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=self._apply_result("nvml"),
                    ns_after_apply=300,
                )
            ],
        }
        dcgm = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=self._apply_result("dcgm"),
                    ns_after_apply=300,
                )
            ],
        }
        self.assertEqual(parity.parity_check(nvml, dcgm, tolerance_w=2.0), 0)

    def test_wrong_matching_clamp_contract_cannot_qualify(self):
        def wrong_result(actuator: str) -> ApplyResult:
            result = self._apply_result(actuator)
            return ApplyResult(
                gpu_uuid=result.gpu_uuid,
                requested_watts=50,
                target_watts=300,
                constraint_min_watts=result.constraint_min_watts,
                constraint_max_watts=result.constraint_max_watts,
                policy_outcome=result.policy_outcome,
                write_outcome=result.write_outcome,
                readback_outcome=result.readback_outcome,
                enforced_cap_watts=300,
                actuator=result.actuator,
                observed_at=result.observed_at,
            )

        nvml = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=wrong_result("nvml"),
                    ns_after_apply=300,
                )
            ],
        }
        dcgm = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=wrong_result("dcgm"),
                    ns_after_apply=300,
                )
            ],
        }
        self.assertGreater(
            parity.parity_check(
                nvml,
                dcgm,
                tolerance_w=2.0,
                require_writes=True,
                expected_request_watts=50,
            ),
            0,
        )

    def test_matching_failed_apply_results_cannot_qualify(self):
        def failed_result(actuator: str) -> ApplyResult:
            return ApplyResult(
                gpu_uuid="GPU-A",
                requested_watts=300,
                target_watts=300,
                constraint_min_watts=100,
                constraint_max_watts=700,
                policy_outcome="annotated",
                write_outcome="failed",
                readback_outcome="not_attempted",
                enforced_cap_watts=None,
                actuator=actuator,
                observed_at=datetime.now(timezone.utc),
            )

        nvml = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=failed_result("nvml"),
                    ns_after_apply=700,
                )
            ],
        }
        dcgm = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=failed_result("dcgm"),
                    ns_after_apply=700,
                )
            ],
        }
        self.assertGreater(
            parity.parity_check(
                nvml,
                dcgm,
                tolerance_w=2.0,
                require_writes=True,
            ),
            0,
        )

    def test_matching_wrong_independent_readbacks_cannot_qualify(self):
        nvml = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=self._apply_result("nvml"),
                    ns_after_apply=350,
                )
            ],
        }
        dcgm = {
            "device_count": 1,
            "gpus": [
                _gpu(
                    "GPU-A",
                    0,
                    apply_result=self._apply_result("dcgm"),
                    ns_after_apply=350,
                )
            ],
        }
        self.assertGreater(
            parity.parity_check(
                nvml,
                dcgm,
                tolerance_w=2.0,
                require_writes=True,
            ),
            0,
        )

    def test_all_skipped_write_matrix_cannot_qualify(self):
        nvml = {"device_count": 1, "gpus": [_gpu("GPU-A", 0)]}
        dcgm = {"device_count": 1, "gpus": [_gpu("GPU-A", 0)]}
        self.assertGreater(
            parity.parity_check(
                nvml,
                dcgm,
                tolerance_w=2.0,
                require_writes=True,
            ),
            0,
        )


class TestProbeSafety(unittest.TestCase):
    def test_require_default_refuses_all_writes_before_first_apply(self):
        actuator = _FakeActuator({"GPU-A": 700.0, "GPU-B": 650.0})
        with (
            patch.object(
                parity,
                "nvidia_smi_power_limit",
                side_effect=lambda uuid: actuator.states[uuid],
            ),
            patch.object(
                parity,
                "nvidia_smi_default_limit",
                side_effect=lambda uuid: actuator.defaults[uuid],
            ),
            self.assertRaisesRegex(RuntimeError, "GPU-B=650.0W"),
        ):
            parity.probe(
                actuator,
                250,
                verify_writes=True,
                sleep_s=0,
                require_default_before_write=True,
            )
        self.assertEqual(actuator.apply_calls, [])
        self.assertTrue(actuator.shutdown_called)

    def test_failed_independent_read_restores_exact_entry_cap(self):
        actuator = _FakeActuator({"GPU-A": 700.0})
        reads = 0

        def read_power(uuid):
            nonlocal reads
            reads += 1
            if reads == 2:
                raise RuntimeError("independent read failed")
            return actuator.states[uuid]

        with (
            patch.object(parity, "nvidia_smi_power_limit", side_effect=read_power),
            patch.object(
                parity,
                "nvidia_smi_default_limit",
                side_effect=lambda uuid: actuator.defaults[uuid],
            ),
            patch.object(parity.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "independent read failed"),
        ):
            parity.probe(
                actuator,
                250,
                verify_writes=True,
                sleep_s=0,
            )
        self.assertEqual(actuator.states["GPU-A"], 700.0)
        self.assertEqual(
            actuator.apply_calls,
            [("GPU-A", 250), ("GPU-A", 700)],
        )
        self.assertTrue(actuator.shutdown_called)

    def test_cleanup_write_runs_even_when_independent_reads_stay_failed(self):
        actuator = _FakeActuator({"GPU-A": 700.0})
        reads = 0

        def read_power(uuid):
            nonlocal reads
            reads += 1
            if reads > 1:
                raise RuntimeError("independent read unavailable")
            return actuator.states[uuid]

        with (
            patch.object(parity, "nvidia_smi_power_limit", side_effect=read_power),
            patch.object(
                parity,
                "nvidia_smi_default_limit",
                side_effect=lambda uuid: actuator.defaults[uuid],
            ),
            patch.object(parity.time, "sleep"),
            self.assertRaisesRegex(RuntimeError, "independent read unavailable"),
        ):
            parity.probe(
                actuator,
                250,
                verify_writes=True,
                sleep_s=0,
            )
        self.assertEqual(actuator.states["GPU-A"], 700.0)
        self.assertEqual(
            actuator.apply_calls,
            [("GPU-A", 250), ("GPU-A", 700)],
        )
        self.assertTrue(actuator.shutdown_called)


if __name__ == "__main__":
    unittest.main()
