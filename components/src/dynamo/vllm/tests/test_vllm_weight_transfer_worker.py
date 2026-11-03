# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from dynamo.vllm.args import configure_rl_weight_transfer_worker
from dynamo.vllm.weight_transfer_worker import StrictRLWorkerExtension

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
]


@pytest.mark.parametrize(
    "enabled,existing,expected",
    [
        (True, None, "dynamo.vllm.weight_transfer_worker.StrictRLWorkerExtension"),
        (True, "", "dynamo.vllm.weight_transfer_worker.StrictRLWorkerExtension"),
        (True, "user.Worker", "user.Worker"),
        (False, None, None),
    ],
)
def test_default_strict_worker_extension_preserves_user_choice(
    enabled, existing, expected
):
    engine = SimpleNamespace(worker_extension_cls=existing)
    configure_rl_weight_transfer_worker(
        SimpleNamespace(enable_rl_weight_transfer=enabled), engine
    )
    assert engine.worker_extension_cls == expected


def test_standalone_worker_device_info_uses_assigned_device(monkeypatch):
    worker = StrictRLWorkerExtension()
    worker.device = torch.device("cuda:3")
    properties = Mock(return_value=SimpleNamespace(uuid="GPU-owned-device"))
    ambient = Mock(side_effect=AssertionError("ambient device must not be read"))
    monkeypatch.setattr(torch.cuda, "get_device_properties", properties)
    monkeypatch.setattr(torch.cuda, "current_device", ambient)
    assert worker.get_weight_transfer_device_info() == {
        "device_uuid": "GPU-owned-device"
    }
    properties.assert_called_once_with(worker.device)
    ambient.assert_not_called()


def test_standalone_worker_device_info_propagates_lookup_failure(monkeypatch):
    worker = StrictRLWorkerExtension()
    worker.device = torch.device("cuda:7")
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        Mock(side_effect=RuntimeError("device unavailable")),
    )
    with pytest.raises(RuntimeError, match="device unavailable"):
        worker.get_weight_transfer_device_info()
