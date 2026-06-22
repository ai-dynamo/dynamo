# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from types import SimpleNamespace

import pytest

from dynamo.trtllm.constants import DisaggregationMode, Modality
from dynamo.trtllm.main import (  # noqa: E402
    _SnapshotRuntimeProxy,
    _validate_supported_snapshot_config,
)

pytest.importorskip(
    "dynamo._core",
    reason="dynamo Rust Python bindings are required for dynamo.trtllm.main",
)
pytest.importorskip(
    "dynamo.nixl_connect",
    reason="NIXL bindings are required to import TRTLLM workers",
)
pytest.importorskip("tensorrt_llm")

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Runtime:
    def __init__(self) -> None:
        self.shutdown_called = False

    def endpoint(self, name: str) -> str:
        return f"endpoint:{name}"

    def shutdown(self) -> None:
        self.shutdown_called = True


def _snapshot_config(**overrides):
    values = {
        "modality": Modality.TEXT,
        "disaggregation_mode": DisaggregationMode.AGGREGATED,
        "encode_endpoint": "",
        "frontend_decoding": False,
        "tensor_parallel_size": 1,
        "pipeline_parallel_size": 1,
        "gpus_per_node": None,
        "has_connector": lambda name: False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime_config(**overrides):
    values = {
        "namespace": "checkpoint-ns",
        "discovery_backend": "kubernetes",
        "request_plane": "nats",
        "event_plane": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_snapshot_config_accepts_single_gpu_aggregated_text_path():
    _validate_supported_snapshot_config(_snapshot_config())


@pytest.mark.parametrize(
    ("override", "expected"),
    [
        ({"modality": Modality.MULTIMODAL}, "modality=multimodal"),
        (
            {"disaggregation_mode": DisaggregationMode.PREFILL},
            "disaggregation_mode=prefill",
        ),
        ({"encode_endpoint": "dyn://ns.encode.generate"}, "--encode-endpoint"),
        ({"frontend_decoding": True}, "--frontend-decoding"),
        ({"tensor_parallel_size": 2}, "tensor_parallel_size=2"),
        ({"pipeline_parallel_size": 2}, "pipeline_parallel_size=2"),
        ({"gpus_per_node": 2}, "gpus_per_node=2"),
        ({"has_connector": lambda name: name == "kvbm"}, "--connector kvbm"),
    ],
)
def test_snapshot_config_rejects_paths_that_can_create_pre_restore_state(
    override, expected
):
    with pytest.raises(ValueError, match=expected):
        _validate_supported_snapshot_config(_snapshot_config(**override))


@pytest.mark.asyncio
async def test_snapshot_runtime_proxy_materializes_runtime_after_restore(monkeypatch):
    import dynamo.trtllm.main as main_mod

    created_runtime = _Runtime()
    lifecycle_calls = []

    class FakeSnapshotController:
        def __init__(self, engine, pause_controller, checkpoint_config):
            self.engine = engine
            self.pause_controller = pause_controller
            self.checkpoint_config = checkpoint_config

        async def wait_for_restore(self):
            lifecycle_calls.append("pause")
            assert await self.pause_controller.pause(self.engine) is True
            lifecycle_calls.append("resume")
            assert await self.pause_controller.resume() is True
            self.pause_controller.mark_resumed()
            return True

        def reload_restore_identity(self, namespace, discovery_backend):
            assert namespace == "checkpoint-ns"
            assert discovery_backend == "kubernetes"
            return "restored-ns", "kubernetes"

    def fake_create_runtime(discovery_backend, request_plane, event_plane):
        assert discovery_backend == "kubernetes"
        assert request_plane == "nats"
        assert event_plane is None
        return created_runtime, object()

    monkeypatch.setattr(main_mod, "EngineSnapshotController", FakeSnapshotController)
    monkeypatch.setattr(main_mod, "create_runtime", fake_create_runtime)

    proxy = _SnapshotRuntimeProxy(checkpoint_config=object())
    config = _runtime_config()

    with pytest.raises(RuntimeError, match="not available until"):
        proxy.endpoint("ns.component.generate")

    await proxy.snapshot_before_endpoint(engine=object(), config=config)

    assert lifecycle_calls == ["pause", "resume"]
    assert config.namespace == "restored-ns"
    assert config.discovery_backend == "kubernetes"
    assert proxy.endpoint("ns.component.generate") == "endpoint:ns.component.generate"

    proxy.shutdown()
    assert created_runtime.shutdown_called is True


@pytest.mark.asyncio
async def test_snapshot_runtime_proxy_exits_without_runtime_after_capture(monkeypatch):
    import dynamo.trtllm.main as main_mod

    class SnapshotCaptured(Exception):
        pass

    class FakeSnapshotController:
        def __init__(self, engine, pause_controller, checkpoint_config):
            pass

        async def wait_for_restore(self):
            return False

    def unexpected_create_runtime(*args, **kwargs):
        raise AssertionError("runtime must not be created for initial capture")

    def fake_exit(code):
        assert code == 0
        raise SnapshotCaptured

    monkeypatch.setattr(main_mod, "EngineSnapshotController", FakeSnapshotController)
    monkeypatch.setattr(main_mod, "create_runtime", unexpected_create_runtime)
    monkeypatch.setattr(main_mod.os, "_exit", fake_exit)

    proxy = _SnapshotRuntimeProxy(checkpoint_config=object())

    with pytest.raises(SnapshotCaptured):
        await proxy.snapshot_before_endpoint(engine=object(), config=_runtime_config())
