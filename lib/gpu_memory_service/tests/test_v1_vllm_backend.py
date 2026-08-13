# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from contextlib import nullcontext
from unittest.mock import Mock

import pytest

pytest.importorskip("vllm", reason="vLLM is required")

import gpu_memory_service.v1.integrations.vllm.backend as backend_module  # noqa: E402
from vllm.device_allocator.sleep_mode_backend import (  # noqa: E402
    SleepModeBackendFactory,
)

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
]


def test_factory_backend_delegates_capture_and_memory_control(monkeypatch):
    client = Mock()
    client.weight_region.return_value = nullcontext()
    client.kv_cache_region.return_value = nullcontext()
    monkeypatch.setattr(
        backend_module,
        "TorchMempoolMemoryClient",
        Mock(return_value=client),
    )

    backend = SleepModeBackendFactory.create_backend(Mock(sleep_mode_backend="gms-v1"))
    model = object()
    with backend.capture_weights(lambda: model), backend.capture_kv_cache():
        pass

    client.weight_region.assert_called_once_with()
    client.publish_weights.assert_called_once_with((model,))
    client.kv_cache_region.assert_called_once_with()

    backend.suspend()

    assert backend.state() == "SUSPENDED"
    client.suspend.assert_called_once_with()

    client.resume.side_effect = RuntimeError("resume rejected")
    with pytest.raises(RuntimeError, match="resume rejected"):
        backend.resume()
    assert backend.state() == "SUSPENDED"

    client.resume.side_effect = None
    backend.resume()

    assert backend.state() == "RUNNING"
    assert client.resume.call_count == 2
