# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR A — Actuator Protocol tests.

Asserts:
  - NvmlActuator structurally satisfies the Actuator Protocol.
  - Each NvmlActuator method dispatches to the right pynvml/power_agent
    function with the expected arguments (no behaviour drift vs PR #9682).
  - apply_cap returns the effective watts (Protocol contract) and routes
    through power_agent._clamp_to_constraints + power_agent._apply_cap
    (preserving the exact code paths exercised by test_apply_cap.py).
  - The constructor's `metrics=None` default raises a clear error if
    apply_cap is called without one — protects future callers from a
    silent metrics-not-recorded bug.

Out of scope for PR A: behaviour-level tests of apply/restore — those
are covered byte-for-byte by the existing test_apply_cap.py and
test_shutdown.py through the module-level `_apply_cap` /
`_handle_sigterm` paths that NvmlActuator delegates to.
"""

import unittest
from unittest.mock import MagicMock, patch

import power_agent
from actuator import Actuator, NvmlActuator


class TestProtocolSatisfaction(unittest.TestCase):
    """NvmlActuator must structurally satisfy Actuator (@runtime_checkable)."""

    def test_nvml_actuator_is_instance_of_actuator(self):
        actuator = NvmlActuator()
        self.assertIsInstance(actuator, Actuator)

    def test_name_attribute_is_nvml(self):
        self.assertEqual(NvmlActuator.name, "nvml")
        self.assertEqual(NvmlActuator().name, "nvml")

    def test_all_protocol_methods_present(self):
        actuator = NvmlActuator()
        for method in (
            "init",
            "shutdown",
            "device_count",
            "get_uuid",
            "list_running_pids",
            "constraints_w",
            "current_w",
            "default_w",
            "apply_cap",
            "restore_default",
            "restore_default_by_uuid",
        ):
            self.assertTrue(
                callable(getattr(actuator, method, None)),
                f"NvmlActuator missing or non-callable method: {method}",
            )


class TestInitAndShutdown(unittest.TestCase):
    """init/shutdown are no-ops in PR A (PR B will move NVML init here)."""

    def test_init_is_noop(self):
        # No NVML import needed; this must not call pynvml.
        with patch.object(power_agent, "pynvml", MagicMock()) as mock_nvml:
            NvmlActuator().init()
        mock_nvml.nvmlInit.assert_not_called()
        mock_nvml.nvmlShutdown.assert_not_called()

    def test_shutdown_is_noop(self):
        with patch.object(power_agent, "pynvml", MagicMock()) as mock_nvml:
            NvmlActuator().shutdown()
        mock_nvml.nvmlInit.assert_not_called()
        mock_nvml.nvmlShutdown.assert_not_called()


class TestDeviceCount(unittest.TestCase):
    def test_device_count_calls_nvml_get_count(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetCount.return_value = 8
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = NvmlActuator().device_count()
        self.assertEqual(result, 8)
        mock_nvml.nvmlDeviceGetCount.assert_called_once_with()


class TestGetUuid(unittest.TestCase):
    def test_get_uuid_delegates_to_power_agent_nvml_uuid(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_3"
        mock_nvml.nvmlDeviceGetUUID.return_value = b"GPU-deadbeef"
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                result = NvmlActuator().get_uuid(3)
        self.assertEqual(result, "GPU-deadbeef")
        self.assertIsInstance(result, str)
        mock_nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(3)

    def test_get_uuid_handles_str_return_from_new_pynvml(self):
        """Coverage parity with TestNvmlUuid: nvidia-ml-py returns str."""
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_nvml.nvmlDeviceGetUUID.return_value = "GPU-str-uuid"
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                result = NvmlActuator().get_uuid(0)
        self.assertEqual(result, "GPU-str-uuid")


class TestListRunningPids(unittest.TestCase):
    def test_returns_pid_list_via_nvml(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        proc_a = MagicMock()
        proc_a.pid = 1234
        proc_b = MagicMock()
        proc_b.pid = 5678
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [proc_a, proc_b]
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = NvmlActuator().list_running_pids(0)
        self.assertEqual(result, [1234, 5678])
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.assert_called_once_with(
            "handle_0"
        )

    def test_returns_empty_list_when_no_processes(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = NvmlActuator().list_running_pids(0)
        self.assertEqual(result, [])


class TestConstraintsW(unittest.TestCase):
    def test_constraints_converts_mw_to_w(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_1"
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (
            100_000,
            700_000,
        )
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            min_w, max_w = NvmlActuator().constraints_w(1)
        self.assertEqual((min_w, max_w), (100, 700))
        mock_nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)


class TestCurrentWAndDefaultW(unittest.TestCase):
    """v1.5 Protocol additions — lift `current` / `default` reads onto
    the actuator surface so orphan recovery's `current<default` guard
    works through the abstraction. NVML side wraps the same two NVML
    calls the inline pre-v1.5 code used."""

    def test_current_w_returns_watts_not_milliwatts(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_2"
        mock_nvml.nvmlDeviceGetPowerManagementLimit.return_value = 423_000  # mW
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = NvmlActuator().current_w(2)
        self.assertEqual(result, 423)
        mock_nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
        mock_nvml.nvmlDeviceGetPowerManagementLimit.assert_called_once_with("handle_2")

    def test_default_w_returns_watts_not_milliwatts(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_2"
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700_000
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            result = NvmlActuator().default_w(2)
        self.assertEqual(result, 700)
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.assert_called_once_with(
            "handle_2"
        )

    def test_current_w_and_default_w_use_different_nvml_calls(self):
        """Regression guard — they MUST call different NVML APIs.
        Pre-v1.5 inline code used GetPowerManagementLimit (current)
        and GetPowerManagementDefaultLimit (default); confusing them
        breaks the orphan-recovery guard."""
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "h"
        mock_nvml.nvmlDeviceGetPowerManagementLimit.return_value = 300_000
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700_000
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            actuator = NvmlActuator()
            current = actuator.current_w(0)
            default = actuator.default_w(0)
        self.assertNotEqual(current, default)
        self.assertEqual((current, default), (300, 700))


class TestApplyCap(unittest.TestCase):
    """apply_cap must delegate to power_agent._apply_cap (preserves PR #9682 path)."""

    def setUp(self):
        power_agent._managed_gpu_indices.clear()
        power_agent._previously_managed.clear()

    def test_apply_cap_requires_metrics(self):
        actuator = NvmlActuator()  # no metrics
        with self.assertRaises(RuntimeError) as ctx:
            actuator.apply_cap(0, 300)
        self.assertIn("metrics", str(ctx.exception).lower())

    def test_apply_cap_returns_effective_watts(self):
        mock_nvml = MagicMock()
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (
            100_000,
            700_000,
        )
        mock_nvml.nvmlDeviceGetUUID.return_value = b"GPU-test-0"
        metrics = MagicMock()
        actuator = NvmlActuator(metrics=metrics)
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                with patch("power_agent._persist_managed_gpus"):
                    result = actuator.apply_cap(0, 300)
        self.assertEqual(result, 300)  # within constraints, no clamp
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            "handle_0", 300_000
        )

    def test_apply_cap_clamps_above_max_and_returns_clamped(self):
        """Requested 900 W with max 700 W → return 700 (the effective watts)."""
        mock_nvml = MagicMock()
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (
            100_000,
            700_000,
        )
        mock_nvml.nvmlDeviceGetUUID.return_value = b"GPU-test-0"
        metrics = MagicMock()
        actuator = NvmlActuator(metrics=metrics)
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                with patch("power_agent._persist_managed_gpus"):
                    result = actuator.apply_cap(0, 900)
        self.assertEqual(result, 700)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            "handle_0", 700_000
        )

    def test_apply_cap_tracks_managed_gpu(self):
        """Confirms delegation routes through power_agent._apply_cap's bookkeeping."""
        mock_nvml = MagicMock()
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_2"
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (
            100_000,
            700_000,
        )
        mock_nvml.nvmlDeviceGetUUID.return_value = b"GPU-test-2"
        metrics = MagicMock()
        actuator = NvmlActuator(metrics=metrics)
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                with patch("power_agent._persist_managed_gpus"):
                    actuator.apply_cap(2, 400)
        self.assertIn(2, power_agent._managed_gpu_indices)

    def test_apply_cap_clamp_fires_exactly_once_on_out_of_range(self):
        """Regression for v1.6 review comment #8.

        Pre-fix `NvmlActuator.apply_cap` called `_clamp_to_constraints`
        directly AND then delegated to `_apply_cap` which also calls
        `_clamp_to_constraints` internally — so an out-of-range
        request logged the clamp warning twice and incremented
        `cap_clamped_total` twice. The Prometheus metric must fire
        exactly once per actual clamp.
        """
        mock_nvml = MagicMock()
        mock_nvml.NVMLError = Exception
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_0"
        mock_nvml.nvmlDeviceGetPowerManagementLimitConstraints.return_value = (
            100_000,
            700_000,
        )
        mock_nvml.nvmlDeviceGetUUID.return_value = b"GPU-test-0"
        metrics = MagicMock()
        actuator = NvmlActuator(metrics=metrics)

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                with patch("power_agent._persist_managed_gpus"):
                    # 900 W is above max (700) → exactly one clamp.
                    result = actuator.apply_cap(0, 900)

        self.assertEqual(result, 700)
        # Single increment of cap_clamped_total{direction="max"}.
        max_clamp_increments = [
            call
            for call in metrics.cap_clamped_total.labels.call_args_list
            if call.kwargs.get("direction") == "max"
        ]
        self.assertEqual(
            len(max_clamp_increments),
            1,
            f"Expected exactly 1 cap_clamped_total{{direction=max}} "
            f"increment; got {len(max_clamp_increments)}. "
            "The double-clamp bug from the v1.6 review is back.",
        )


class TestRestoreDefault(unittest.TestCase):
    def test_restore_default_calls_set_to_default_limit(self):
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.return_value = "handle_1"
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700_000
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            NvmlActuator().restore_default(1)
        mock_nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(1)
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.assert_called_once_with(
            "handle_1"
        )
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            "handle_1", 700_000
        )


class TestRestoreDefaultByUuid(unittest.TestCase):
    """Identity-stable restore (PR9790 review). The target index is resolved
    from the UUID at write time, and the `current_w < default_w` guard is
    applied at the resolved index — mirroring DcgmActuator's contract so
    orphan recovery and the SIGTERM sweep behave the same on both paths."""

    def _nvml(self, uuids, current_mw, default_mw):
        """Build a pynvml mock where index→UUID/current/default are dicts."""
        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetCount.return_value = len(uuids)
        mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: f"h{idx}"
        mock_nvml.nvmlDeviceGetUUID.side_effect = lambda h: uuids[int(h[1:])].encode()
        mock_nvml.nvmlDeviceGetPowerManagementLimit.side_effect = lambda h: current_mw[
            int(h[1:])
        ]
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.side_effect = (
            lambda h: default_mw[int(h[1:])]
        )
        return mock_nvml

    def test_restores_at_resolved_index_when_below_default(self):
        # UUID 'GPU-b' lives at index 1; capped below default → restore there.
        mock_nvml = self._nvml(
            uuids={0: "GPU-a", 1: "GPU-b"},
            current_mw={0: 700_000, 1: 400_000},
            default_mw={0: 700_000, 1: 700_000},
        )
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                result = NvmlActuator().restore_default_by_uuid("GPU-b")
        self.assertTrue(result)
        # The default-limit write landed on index 1's handle, not index 0.
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_called_once_with(
            "h1", 700_000
        )

    def test_returns_none_when_already_at_default(self):
        mock_nvml = self._nvml(
            uuids={0: "GPU-a"},
            current_mw={0: 700_000},
            default_mw={0: 700_000},
        )
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                result = NvmlActuator().restore_default_by_uuid("GPU-a")
        self.assertIsNone(result)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()

    def test_returns_none_when_uuid_absent_on_clean_scan(self):
        mock_nvml = self._nvml(
            uuids={0: "GPU-a"},
            current_mw={0: 400_000},
            default_mw={0: 700_000},
        )
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                result = NvmlActuator().restore_default_by_uuid("GPU-missing")
        self.assertIsNone(result)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()

    def test_returns_false_when_scan_inconclusive(self):
        """A probe raising mid-scan (transient outage) before any match must
        return False (indeterminate) so the caller keeps the UUID."""
        mock_nvml = self._nvml(
            uuids={0: "GPU-a", 1: "GPU-b"},
            current_mw={0: 400_000, 1: 400_000},
            default_mw={0: 700_000, 1: 700_000},
        )

        def boom(h):
            raise RuntimeError("transient NVML failure")

        mock_nvml.nvmlDeviceGetUUID.side_effect = boom
        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                result = NvmlActuator().restore_default_by_uuid("GPU-b")
        self.assertFalse(result)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()


class TestPowerAgentBindsNvmlActuatorByDefault(unittest.TestCase):
    """PowerAgent.__init__ default actuator is NvmlActuator (PR A contract)."""

    def test_default_actuator_is_nvml(self):
        # Bypass __init__ but exercise the contract by reading the default
        # parameter via inspect — avoids the NVML + K8s dependency.
        import inspect

        sig = inspect.signature(power_agent.PowerAgent.__init__)
        actuator_param = sig.parameters["actuator"]
        # Default must be None (None → NvmlActuator inside __init__), and the
        # parameter must be optional so existing callers keep working.
        self.assertIsNone(actuator_param.default)


if __name__ == "__main__":
    unittest.main()
