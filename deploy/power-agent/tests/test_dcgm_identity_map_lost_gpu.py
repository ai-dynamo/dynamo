# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DcgmActuator identity map with one GPU that NVML cannot query.

NVML documents NVML_ERROR_GPU_IS_LOST for nvmlDeviceGetHandleByIndex and
nvmlDeviceGetUUID when a GPU has fallen off the bus.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from tests.test_dcgm_actuator import _make_initialized_actuator, _wire_identity_map


class _GpuIsLost(Exception):
    """Stands in for pynvml.NVMLError_GpuIsLost."""


class TestIdentityMapWithLostGpu(unittest.TestCase):
    def test_healthy_gpu_still_reports_pids_when_another_gpu_is_lost(self):
        actuator, modules, handle, _ = _make_initialized_actuator(gpu_ids=(0, 1))
        nvml = MagicMock()
        nvml.NVMLError = _GpuIsLost
        _wire_identity_map(
            modules,
            nvml,
            handle,
            dcgm_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
            nvml_uuids=[b"GPU-aaaa", b"GPU-bbbb"],
        )

        def get_handle(index):
            if index == 1:
                raise _GpuIsLost("GPU is lost")
            return f"nvml_handle_{index}"

        nvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [MagicMock(pid=1234)]

        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            # GPU 0 is healthy: its PIDs must still be readable, so its pods'
            # caps keep being enforced. GPU 1 may fail on its own.
            self.assertEqual(actuator.list_running_pids(0), [1234])

    def _wire_lost_gpu(self, dcgm_uuids, nvml_uuids, lost_index):
        actuator, modules, handle, _ = _make_initialized_actuator(
            gpu_ids=tuple(range(len(dcgm_uuids)))
        )
        nvml = MagicMock()
        nvml.NVMLError = _GpuIsLost
        _wire_identity_map(
            modules, nvml, handle, dcgm_uuids=dcgm_uuids, nvml_uuids=nvml_uuids
        )

        def get_handle(index):
            if index == lost_index:
                raise _GpuIsLost("GPU is lost")
            return f"nvml_handle_{index}"

        nvml.nvmlDeviceGetHandleByIndex.side_effect = get_handle
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = []
        return actuator, modules, nvml

    def test_disagreement_between_answering_libraries_still_fails(self):
        # Both libraries answer for every index but name different GPUs:
        # that is the mis-routing case the map must refuse.
        actuator, modules, nvml = self._wire_lost_gpu(
            [b"GPU-aaaa", b"GPU-bbbb"], [b"GPU-aaaa", b"GPU-cccc"], lost_index=None
        )
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertRaises(RuntimeError):
                actuator.list_running_pids(0)

    def test_more_missing_uuids_than_unreadable_indices_still_fails(self):
        actuator, modules, nvml = self._wire_lost_gpu(
            [b"GPU-aaaa", b"GPU-bbbb", b"GPU-dddd"],
            [b"GPU-aaaa", b"GPU-bbbb", b"GPU-cccc"],
            lost_index=1,
        )
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertRaises(RuntimeError):
                actuator.list_running_pids(0)

    def test_recovered_gpu_is_mapped_again(self):
        actuator, modules, nvml = self._wire_lost_gpu(
            [b"GPU-aaaa", b"GPU-bbbb"], [b"GPU-aaaa", b"GPU-bbbb"], lost_index=1
        )
        nvml.nvmlDeviceGetComputeRunningProcesses.return_value = [MagicMock(pid=42)]
        with patch.dict("sys.modules", {**modules, "pynvml": nvml}):
            with self.assertRaises(RuntimeError):
                actuator.list_running_pids(1)
            # The healthy GPU reads from the cached map: no new NVML walk.
            builds = nvml.nvmlDeviceGetCount.call_count
            self.assertEqual(actuator.list_running_pids(0), [42])
            self.assertEqual(nvml.nvmlDeviceGetCount.call_count, builds)

            # GPU 1 recovers: the next read for it rebuilds the map.
            nvml.nvmlDeviceGetHandleByIndex.side_effect = lambda i: f"nvml_handle_{i}"
            self.assertEqual(actuator.list_running_pids(1), [42])


if __name__ == "__main__":
    unittest.main()
