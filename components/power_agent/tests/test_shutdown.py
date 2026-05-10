# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the SIGTERM handler (§6.5).

Verifies that:
  - SIGTERM handler restores default TGP for every managed GPU.
  - nvmlShutdown is called exactly once.
  - _shutdown event is set after the handler completes.
"""

import signal
import unittest
from unittest.mock import MagicMock, call, patch

import power_agent


class TestSigtermHandler(unittest.TestCase):
    def setUp(self):
        # Reset module-level state between tests
        power_agent._managed_gpu_indices.clear()
        power_agent._shutdown.clear()

    def test_restores_tgp_for_all_managed_gpus(self):
        power_agent._managed_gpu_indices.update([0, 1, 2])

        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda idx: f"handle_{idx}"
        mock_nvml.nvmlDeviceGetPowerManagementDefaultLimit.return_value = 700_000
        mock_nvml.NVMLError = Exception

        with patch.dict("sys.modules", {"pynvml": mock_nvml}):
            with patch.object(power_agent, "pynvml", mock_nvml):
                power_agent._handle_sigterm(signal.SIGTERM, None)

        # Three GPUs managed → three SetPowerManagementLimit calls
        self.assertEqual(
            mock_nvml.nvmlDeviceSetPowerManagementLimit.call_count, 3
        )
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_any_call("handle_0", 700_000)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_any_call("handle_1", 700_000)
        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_any_call("handle_2", 700_000)
        # nvmlShutdown called once
        mock_nvml.nvmlShutdown.assert_called_once()
        # _shutdown event is set
        self.assertTrue(power_agent._shutdown.is_set())

    def test_no_managed_gpus_still_shuts_down(self):
        mock_nvml = MagicMock()
        with patch.object(power_agent, "pynvml", mock_nvml):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        mock_nvml.nvmlDeviceSetPowerManagementLimit.assert_not_called()
        mock_nvml.nvmlShutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())

    def test_nvml_error_does_not_prevent_shutdown(self):
        power_agent._managed_gpu_indices.add(0)

        mock_nvml = MagicMock()
        mock_nvml.nvmlDeviceGetHandleByIndex.side_effect = Exception("nvml error")
        mock_nvml.NVMLError = Exception

        with patch.object(power_agent, "pynvml", mock_nvml):
            power_agent._handle_sigterm(signal.SIGTERM, None)

        # Despite the error, nvmlShutdown and _shutdown.set() still execute
        mock_nvml.nvmlShutdown.assert_called_once()
        self.assertTrue(power_agent._shutdown.is_set())


if __name__ == "__main__":
    unittest.main()
