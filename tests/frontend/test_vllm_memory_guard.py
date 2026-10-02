# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The frontend test worker must not silently weaken its startup VRAM guard."""

import math
import os
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
        nvmlDeviceGetUUID=Mock(return_value="GPU-scheduled"),
        nvmlDeviceGetMemoryInfo=Mock(
            return_value=SimpleNamespace(total=80 * 1024**3)
        ),
        nvmlShutdown=Mock(),
    )
    monkeypatch.setattr(test_vllm, "pynvml", stub)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    return stub


@pytest.mark.parametrize(
    "operation",
    [
        "nvmlInit",
        "nvmlDeviceGetHandleByIndex",
        "nvmlDeviceGetMemoryInfo",
        "nvmlDeviceGetUUID",
    ],
)
def test_nvml_failure_prevents_guard_calculation(nvml, operation):
    """Capacity or identity lookup failures must stop worker construction."""
    error = pynvml.NVMLError(pynvml.NVML_ERROR_UNKNOWN)
    getattr(nvml, operation).side_effect = error
    worker = SimpleNamespace(required_vram_gib=18.7)

    with pytest.raises(pynvml.NVMLError) as raised:
        test_vllm.WorkerProcess._gpu_memory_utilization(worker, os.environ.copy())

    assert raised.value is error
    assert nvml.nvmlShutdown.call_count == (operation != "nvmlInit")


@pytest.mark.parametrize("visible", ["", "-1", "GPU-test", "MIG-test"])
def test_unknown_device_prevents_guard_calculation(nvml, monkeypatch, visible):
    """Reject selectors outside the scheduler's numeric-index contract."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    worker = SimpleNamespace(required_vram_gib=18.7)

    with pytest.raises(ValueError, match="expected a numeric GPU index"):
        test_vllm.WorkerProcess._gpu_memory_utilization(worker, os.environ.copy())

    nvml.nvmlInit.assert_not_called()


def test_missing_budget_prevents_guard_calculation(nvml):
    """A missing budget must not restore the weak fallback guard."""
    worker = SimpleNamespace(required_vram_gib=None)

    with pytest.raises(ValueError, match="profiled_vram_gib is required"):
        test_vllm.WorkerProcess._gpu_memory_utilization(worker, os.environ.copy())

    nvml.nvmlInit.assert_not_called()


def test_guard_uses_declared_budget(nvml):
    """The startup threshold must preserve the full declared byte budget."""
    worker = SimpleNamespace(required_vram_gib=18.7)

    utilization = float(
        test_vllm.WorkerProcess._gpu_memory_utilization(worker, os.environ.copy())
    )
    threshold_bytes = math.ceil(80 * 1024**3 * utilization)
    assert threshold_bytes == math.ceil(worker.required_vram_gib * 1024**3)
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(2)
    nvml.nvmlShutdown.assert_called_once_with()


@pytest.mark.parametrize("visible, expected_index", [(None, 0), ("2", 2), ("2,1", 2)])
def test_worker_pins_guarded_gpu_by_uuid(
    nvml, monkeypatch, dynamo_dynamic_ports, visible, expected_index
):
    """The launched worker must use the guarded card even if CUDA orders GPUs differently."""
    if visible is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    monkeypatch.delenv("_PROFILE_OVERRIDE_VLLM_KV_CACHE_BYTES", raising=False)
    initialize = Mock(return_value=None)
    monkeypatch.setattr(test_vllm.ManagedProcess, "__init__", initialize)
    markers = {
        "profiled_vram_gib": SimpleNamespace(args=(18.7,)),
        "requested_vllm_kv_cache_bytes": SimpleNamespace(args=(1024,)),
    }
    request = SimpleNamespace(
        node=SimpleNamespace(name="gpu_identity", get_closest_marker=markers.get)
    )

    test_vllm.WorkerProcess(
        request,
        "worker",
        dynamo_dynamic_ports.frontend_port,
        dynamo_dynamic_ports.system_ports[0],
    )

    launch = initialize.call_args.kwargs
    assert launch["env"]["CUDA_VISIBLE_DEVICES"] == "GPU-scheduled"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == visible
    nvml.nvmlDeviceGetHandleByIndex.assert_called_once_with(expected_index)
    handle = nvml.nvmlDeviceGetHandleByIndex.return_value
    nvml.nvmlDeviceGetUUID.assert_called_once_with(handle)
    nvml.nvmlDeviceGetMemoryInfo.assert_called_once_with(handle)
    command = launch["command"]
    utilization = float(command[command.index("--gpu-memory-utilization") + 1])
    assert math.ceil(80 * 1024**3 * utilization) == math.ceil(18.7 * 1024**3)
