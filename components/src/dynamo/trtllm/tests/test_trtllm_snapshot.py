# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from dynamo.trtllm import snapshot as snapshot_mod
from dynamo.trtllm.constants import DisaggregationMode, Modality
from dynamo.trtllm.snapshot import (
    _should_prefetch_model_for_snapshot,
    _SnapshotRuntimeProxy,
    _validate_supported_snapshot_config,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.core,
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


class _GenerationResult:
    def __init__(self, events=None, *, delay=0.0, error=None):
        self._events = events if events is not None else []
        self._delay = delay
        self.finished = False
        self.error = error
        self.outputs = []
        self.abort = Mock()

    async def __aiter__(self):
        self._events.append("warmup-chunk")
        yield self
        if self._delay:
            await asyncio.sleep(self._delay)
        self._events.append("warmup-final")
        self.finished = True
        self.outputs = [SimpleNamespace(token_ids=[7, 8])]
        yield self


def _engine(result):
    llm = SimpleNamespace(generate_async=Mock(return_value=result))
    return SimpleNamespace(llm=llm), llm


@pytest.fixture
def warmup_setup(monkeypatch):
    controller = Mock()
    monkeypatch.setattr(snapshot_mod, "_create_warmup_sampling_params", object)
    monkeypatch.setattr(snapshot_mod, "_create_engine_snapshot_controller", controller)
    return controller


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


def _prefetch_config(**overrides):
    values = {
        "model": "Qwen/Qwen3-0.6B",
        "load_format": "auto",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_snapshot_config_accepts_single_gpu_aggregated_text_path():
    _validate_supported_snapshot_config(_snapshot_config())


def test_snapshot_prefetches_remote_hf_model_before_forcing_offline_mode():
    assert _should_prefetch_model_for_snapshot(_prefetch_config()) is True


def test_snapshot_prefetch_skips_local_model_path(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()

    assert (
        _should_prefetch_model_for_snapshot(
            _prefetch_config(model=str(model_path)),
        )
        is False
    )


def test_snapshot_prefetch_skips_external_model_loader():
    assert (
        _should_prefetch_model_for_snapshot(_prefetch_config(load_format="gms"))
        is False
    )


def test_create_warmup_sampling_params_uses_lazy_trtllm_import(monkeypatch):
    constructor = Mock()
    for name in ("tensorrt_llm", "tensorrt_llm.llmapi"):
        module = ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, name, module)
    llm_module = ModuleType("tensorrt_llm.llmapi.llm")
    llm_module.SamplingParams = constructor  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "tensorrt_llm.llmapi.llm", llm_module)

    snapshot_mod._create_warmup_sampling_params()

    constructor.assert_called_once_with(
        end_id=-1,
        pad_id=-1,
        max_tokens=2,
        temperature=0.0,
        ignore_eos=True,
        detokenize=False,
    )


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
async def test_snapshot_runtime_proxy_materializes_runtime_after_restore(
    monkeypatch, warmup_setup
):
    created_runtime = _Runtime()
    lifecycle_calls = []
    result = _GenerationResult(lifecycle_calls)
    engine, llm = _engine(result)

    class FakeSnapshotController:
        def __init__(self, engine, pause_controller, snapshot_config):
            self.engine = engine
            self.pause_controller = pause_controller

        async def wait_for_restore(self):
            lifecycle_calls.append("pause")
            assert await self.pause_controller.pause(self.engine) is True
            lifecycle_calls.append("resume")
            assert await self.pause_controller.resume() is True
            self.pause_controller.mark_resumed()
            return True

    def fake_create_runtime(discovery_backend, request_plane, event_plane):
        assert discovery_backend == "kubernetes"
        assert request_plane == "nats"
        assert event_plane is None
        return created_runtime, object()

    async def fake_refresh_restore_runtime_config(config, argv):
        assert config.namespace == "checkpoint-ns"
        assert config.discovery_backend == "kubernetes"
        assert argv == ["--endpoint", "dyn://checkpoint-ns.backend.generate"]
        config.namespace = "restored-ns"
        return config

    monkeypatch.setattr(
        snapshot_mod,
        "_create_engine_snapshot_controller",
        FakeSnapshotController,
    )
    monkeypatch.setattr(
        snapshot_mod,
        "_refresh_snapshot_restore_runtime_config",
        fake_refresh_restore_runtime_config,
    )
    monkeypatch.setattr(snapshot_mod, "_create_runtime", fake_create_runtime)

    proxy = _SnapshotRuntimeProxy(
        snapshot_config=object(),
        argv=["--endpoint", "dyn://checkpoint-ns.backend.generate"],
    )
    config = _runtime_config()

    with pytest.raises(RuntimeError, match="not available until"):
        proxy.endpoint("ns.component.generate")

    await proxy.snapshot_before_endpoint(engine=engine, config=config)

    assert lifecycle_calls == [
        "warmup-chunk",
        "warmup-final",
        "pause",
        "resume",
    ]
    call = llm.generate_async.call_args
    assert call.kwargs["inputs"] == [1, 2, 3]
    assert call.kwargs["streaming"] is True
    assert call.kwargs["cache_salt"].startswith("dynamo-snapshot-warmup-")
    assert config.namespace == "restored-ns"
    assert config.discovery_backend == "kubernetes"
    assert proxy.endpoint("ns.component.generate") == "endpoint:ns.component.generate"

    proxy.shutdown()
    assert created_runtime.shutdown_called is True


@pytest.mark.asyncio
async def test_snapshot_runtime_proxy_exits_without_runtime_after_capture(monkeypatch):
    class SnapshotCaptured(Exception):
        pass

    class FakeSnapshotController:
        def __init__(self, engine, pause_controller, snapshot_config):
            pass

        async def wait_for_restore(self):
            return False

    def unexpected_create_runtime(*args, **kwargs):
        raise AssertionError("runtime must not be created for initial capture")

    def fake_exit(code):
        assert code == 0
        raise SnapshotCaptured

    monkeypatch.setattr(snapshot_mod, "warmup_engine", AsyncMock())
    monkeypatch.setattr(
        snapshot_mod,
        "_create_engine_snapshot_controller",
        FakeSnapshotController,
    )
    monkeypatch.setattr(snapshot_mod, "_create_runtime", unexpected_create_runtime)
    monkeypatch.setattr(snapshot_mod.os, "_exit", fake_exit)

    proxy = _SnapshotRuntimeProxy(snapshot_config=object())

    with pytest.raises(SnapshotCaptured):
        await proxy.snapshot_before_endpoint(engine=object(), config=_runtime_config())


@pytest.mark.asyncio
async def test_snapshot_warmup_timeout_aborts_and_prevents_readiness(
    monkeypatch, warmup_setup
):
    result = _GenerationResult(delay=0.1)
    engine, _ = _engine(result)
    monkeypatch.setattr(snapshot_mod, "_WARMUP_TIMEOUT_SEC", 0.01)

    proxy = _SnapshotRuntimeProxy(snapshot_config=object())
    with pytest.raises(asyncio.TimeoutError):
        await proxy.snapshot_before_endpoint(engine=engine, config=_runtime_config())

    result.abort.assert_called_once_with()
    warmup_setup.assert_not_called()


@pytest.mark.asyncio
async def test_snapshot_warmup_terminal_error_prevents_readiness(
    warmup_setup,
):
    result = _GenerationResult(error="executor failed")
    engine, _ = _engine(result)

    proxy = _SnapshotRuntimeProxy(snapshot_config=object())
    with pytest.raises(RuntimeError, match="executor failed"):
        await proxy.snapshot_before_endpoint(engine=engine, config=_runtime_config())

    warmup_setup.assert_not_called()
