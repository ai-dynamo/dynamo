# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend test worker must not silently weaken its startup VRAM guard."""

import math
from types import SimpleNamespace
from unittest.mock import Mock

import pynvml
import pytest

from tests.frontend import test_vllm

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.fixture
def nvml(monkeypatch):
    """Replace driver calls so guard failures can be checked without a GPU."""
    stub = SimpleNamespace(
        NVMLError=pynvml.NVMLError,
        nvmlInit=Mock(),
        nvmlDeviceGetHandleByIndex=Mock(return_value=object()),
        nvmlDeviceGetMemoryInfo=Mock(
            return_value=SimpleNamespace(total=80 * 1024**3)
        ),
        nvmlShutdown=Mock(),
    )
    monkeypatch.setattr(test_vllm, "pynvml", stub)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    return stub


@pytest.mark.parametrize(
    "operation", ["nvmlInit", "nvmlDeviceGetHandleByIndex", "nvmlDeviceGetMemoryInfo"]
)
def test_nvml_failure_prevents_guard_calculation(nvml, operation):
    error = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
    getattr(nvml, operation).side_effect = error
    worker = SimpleNamespace(required_vram_gib=18.7)

    with pytest.raises(pynvml.NVMLError) as raised:
        test_vllm.WorkerProcess._gpu_memory_utilization(worker)

    assert raised.value is error
    assert nvml.nvmlShutdown.call_count == (operation != "nvmlInit")


@pytest.mark.parametrize("visible", ["", "-1", "GPU-test", "MIG-test"])
def test_unknown_device_prevents_guard_calculation(nvml, monkeypatch, visible):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    worker = SimpleNamespace(required_vram_gib=18.7)

    with pytest.raises(ValueError, match="expected a numeric GPU index"):
        test_vllm.WorkerProcess._gpu_memory_utilization(worker)

    nvml.nvmlInit.assert_not_called()


def test_missing_budget_prevents_guard_calculation(nvml):
    worker = SimpleNamespace(required_vram_gib=None)

    with pytest.raises(ValueError, match="profiled_vram_gib is required"):
        test_vllm.WorkerProcess._gpu_memory_utilization(worker)

    nvml.nvmlInit.assert_not_called()


def test_guard_uses_declared_budget(nvml):
    worker = SimpleNamespace(required_vram_gib=18.7)

    utilization = float(test_vllm.WorkerProcess._gpu_memory_utilization(worker))
    threshold_bytes = math.ceil(80 * 1024**3 * utilization)
    assert threshold_bytes == math.ceil(worker.required_vram_gib * 1024**3)
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
    nvml.nvmlShutdown.assert_called_once_with()
