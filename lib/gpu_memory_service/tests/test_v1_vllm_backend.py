# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest
from _deps import HAS_GMS, HAS_TORCH

if not HAS_GMS:
    pytest.skip("gpu_memory_service is required", allow_module_level=True)
if not HAS_TORCH:
    pytest.skip("torch is required", allow_module_level=True)
pytest.importorskip("vllm")

from gpu_memory_service.v1.integrations.vllm.backend import (  # noqa: E402
    GMSV1SleepModeBackend,
)
from vllm.device_allocator.sleep_mode_backend import SleepModeBackend  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
]


def _backend() -> GMSV1SleepModeBackend:
    backend = GMSV1SleepModeBackend.__new__(GMSV1SleepModeBackend)
    SleepModeBackend.__init__(backend)
    backend._client = Mock()
    return backend


@pytest.fixture
def snapshot_process(monkeypatch, tmp_path):
    monkeypatch.setenv("DYN_SNAPSHOT_CONTROL_DIR", str(tmp_path))


def test_discard_kv_cache_suspends_whole_engine_and_kv_cache_wake_resumes_it(
    snapshot_process,
):
    """vLLM's process checkpoint prepare discards the KV cache and its restore
    wakes ["kv_cache"]; GMS V1 releases and brings back the whole engine."""
    backend = _backend()

    backend.discard(("kv_cache",))

    backend._client.suspend.assert_called_once_with()
    assert backend.state() == "SUSPENDED"

    backend.resume(["kv_cache"])

    backend._client.resume.assert_called_once_with()
    assert backend.state() == "RUNNING"


def test_discard_outside_a_snapshot_is_refused(monkeypatch):
    """vLLM's release_kv_cache_memory() expects the weights to stay mapped."""
    monkeypatch.delenv("DYN_SNAPSHOT_CONTROL_DIR", raising=False)
    backend = _backend()

    with pytest.raises(RuntimeError, match="Dynamo Snapshot checkpoint"):
        backend.discard(("kv_cache",))

    backend._client.suspend.assert_not_called()
    assert backend.state() == "RUNNING"


@pytest.mark.parametrize("tags", [("weights",), ("kv_cache", "weights")])
def test_discard_refuses_anything_but_the_kv_cache(snapshot_process, tags):
    backend = _backend()

    with pytest.raises(ValueError, match="only the KV cache"):
        backend.discard(tags)

    backend._client.suspend.assert_not_called()
    assert backend.state() == "RUNNING"


def test_partial_tag_resume_after_sleep_is_still_refused():
    backend = _backend()
    backend.suspend(1)

    with pytest.raises(ValueError, match="partial-tag resume"):
        backend.resume(["kv_cache"])

    backend._client.resume.assert_not_called()


def test_weights_resume_after_kv_cache_discard_is_refused(snapshot_process):
    backend = _backend()
    backend.discard(("kv_cache",))

    with pytest.raises(ValueError, match="partial-tag resume"):
        backend.resume(["weights"])

    backend.resume(None)
    backend._client.resume.assert_called_once_with()
    with pytest.raises(ValueError, match="partial-tag resume"):
        backend.resume(["kv_cache"])
