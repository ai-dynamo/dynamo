# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the TRT-LLM delayed GMS publish isolate."""

from __future__ import annotations

import sys
import types
from contextlib import contextmanager

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


@pytest.fixture(autouse=True)
def clear_delay_commit_state():
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    model_loader.set_delay_commit_until_engine_init(False)
    model_loader._pending_gms_write = None
    model_loader._last_imported_weights_bytes = 0
    yield
    model_loader.set_delay_commit_until_engine_init(False)
    model_loader._pending_gms_write = None
    model_loader._last_imported_weights_bytes = 0


def test_setup_gms_plumbs_delay_commit_flag(monkeypatch):
    import gpu_memory_service.integrations.trtllm as trtllm_gms
    from gpu_memory_service.common.locks import RequestedLockType

    captured: dict[str, object] = {}

    monkeypatch.setattr(trtllm_gms, "patch_empty_cache", lambda: None)
    monkeypatch.setattr(trtllm_gms, "patch_model_loader", lambda: None)
    monkeypatch.setattr(trtllm_gms, "set_gms_enabled", lambda enabled: None)
    monkeypatch.setattr(
        trtllm_gms,
        "set_gms_lock_mode",
        lambda mode: captured.setdefault("lock_mode", mode),
    )
    monkeypatch.setattr(
        trtllm_gms,
        "set_delay_commit_until_engine_init",
        lambda enabled: captured.setdefault("delay", enabled),
    )
    monkeypatch.setattr(
        trtllm_gms,
        "set_extra_config",
        lambda extra: captured.setdefault("extra", extra),
    )
    monkeypatch.setattr(trtllm_gms, "install_mpi_worker_bootstrap", lambda: None)

    trtllm_gms.setup_gms({"gms_delay_commit_until_engine_init": True})

    assert captured["delay"] is True
    assert captured["extra"] == {"gms_delay_commit_until_engine_init": True}
    assert captured["lock_mode"] == RequestedLockType.RW_OR_RO


def test_setup_gms_rejects_non_bool_delay_commit():
    import gpu_memory_service.integrations.trtllm as trtllm_gms

    with pytest.raises(
        ValueError, match="gms_delay_commit_until_engine_init must be a boolean"
    ):
        trtllm_gms.setup_gms({"gms_delay_commit_until_engine_init": "true"})


def test_load_rw_preserves_native_weight_split_and_delays_publish(monkeypatch):
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    call_log: list[str] = []
    fake_client = object()
    fake_model = None

    class FakeAllocatedTensor:
        def __init__(self, kind: str):
            self.kind = kind
            self.device = model_loader.torch.device("cuda")
            self.is_cuda = True

        def copy_(self, _source):
            call_log.append(f"{self.kind}.copy")
            return self

    class FakeSourceTensor:
        def __init__(self, label: str, device: str = "cpu"):
            self.label = label
            self.device = model_loader.torch.device(device)
            self.is_cuda = device == "cuda"

        def cuda(self):
            call_log.append(f"{self.label}.cuda")
            return FakeAllocatedTensor(self.label)

    class FakePostLoadModule:
        _weights_removed = False

        def post_load_weights(self):
            call_log.append("post_load_weights")

    class FakeModel:
        def _apply(self, fn):
            call_log.append("apply_weights")
            fn(FakeSourceTensor("weight"))
            return self

        def to(self, device):
            call_log.append(f"model.to:{device}")
            return self

        def modules(self):
            return [FakePostLoadModule()]

        def load_weights(self, _weights):
            call_log.append("model.load_weights")

    fake_model = FakeModel()

    class FakeCheckpointLoader:
        checkpoint_format = "fake"

        def load_weights(self, checkpoint_dir, mapping=None):
            call_log.append(f"checkpoint_loader.load_weights:{checkpoint_dir}")
            return "weights"

        def get_initialized_weight_mapper(self, model, config):
            call_log.append("checkpoint_loader.get_initialized_weight_mapper")
            return "mapper"

    class FakeMoeLoadBalancer:
        def register_weight_slots_after_to_cuda(self):
            call_log.append("moe.register")

        def finalize_model(self):
            call_log.append("moe.finalize")

    @contextmanager
    def fake_timing(_label):
        yield

    @contextmanager
    def fake_meta_init():
        yield

    @contextmanager
    def fake_moe_context(_config, _mapping):
        yield FakeMoeLoadBalancer()

    fake_trt_loader = types.ModuleType("tensorrt_llm._torch.pyexecutor.model_loader")
    fake_trt_loader.timing = fake_timing
    fake_trt_loader.maybe_create_moe_load_balancer = fake_moe_context
    fake_trt_loader.MetaInitMode = fake_meta_init
    fake_trt_loader.AutoModelForCausalLM = types.SimpleNamespace(
        from_config=lambda _config: fake_model
    )
    fake_trt_loader.LoadFormat = types.SimpleNamespace(
        AUTO="auto",
        DUMMY="dummy",
        VISION_ONLY="vision_only",
    )
    fake_trt_loader._apply_to_buffers_only = lambda _model, fn: (
        call_log.append("apply_buffers"),
        fn(FakeSourceTensor("buffer")),
    )[-1]
    fake_trt_loader.get_rank_model_storage = lambda _model: (
        call_log.append("rank_model_storage"),
        2 << 30,
    )[-1]
    fake_trt_loader.initialize_dummy_weights = lambda _model: call_log.append(
        "initialize_dummy_weights"
    )
    fake_trt_loader.MoeLoadBalancer = FakeMoeLoadBalancer
    fake_trt_loader.AutoCheckpointMapper = types.SimpleNamespace(
        get=lambda *_args: None
    )
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.pyexecutor.model_loader",
        fake_trt_loader,
    )

    @contextmanager
    def fake_gms_pool(_tag, _device):
        call_log.append("enter_gms_pool")
        yield
        call_log.append("exit_gms_pool")

    monkeypatch.setattr(model_loader, "gms_use_mem_pool", fake_gms_pool)
    monkeypatch.setattr(
        model_loader,
        "_move_untracked_params",
        lambda _model, _client, device_index: call_log.append(
            f"move_untracked:{device_index}"
        ),
    )
    monkeypatch.setattr(
        model_loader.torch,
        "empty_like",
        lambda tensor, device=None: FakeAllocatedTensor(tensor.label),
    )
    monkeypatch.setattr(
        model_loader.torch.cuda,
        "current_stream",
        lambda: types.SimpleNamespace(
            synchronize=lambda: call_log.append("synchronize")
        ),
    )
    monkeypatch.setattr(
        model_loader.torch.cuda,
        "empty_cache",
        lambda: call_log.append("empty_cache"),
    )
    monkeypatch.setattr(
        model_loader,
        "finalize_gms_write",
        lambda *_args, **_kwargs: pytest.fail("finalize_gms_write should be deferred"),
    )

    fake_self = types.SimpleNamespace(
        llm_args=types.SimpleNamespace(load_format=fake_trt_loader.LoadFormat.AUTO),
        mapping="mapping",
        spec_config=None,
        _load_and_validate_config=lambda *_args, **_kwargs: {"config": True},
        _call_load_weights=lambda load_fn, weights, mapper: (
            call_log.append("call_load_weights"),
            load_fn(weights),
        )[-1],
    )

    model_loader.set_delay_commit_until_engine_init(True)

    model, moe_load_balancer = model_loader._load_rw(
        self=fake_self,
        checkpoint_dir="checkpoint",
        checkpoint_loader=FakeCheckpointLoader(),
        gms_client=fake_client,
        device_index=2,
        original_load=lambda *_args, **_kwargs: pytest.fail(
            "_load_rw should not call original_load"
        ),
    )

    assert model is fake_model
    assert isinstance(moe_load_balancer, FakeMoeLoadBalancer)
    assert model_loader._pending_gms_write == (fake_client, fake_model)
    assert call_log == [
        "apply_buffers",
        "buffer.cuda",
        "enter_gms_pool",
        "apply_weights",
        "weight.copy",
        "exit_gms_pool",
        "model.to:cuda",
        "rank_model_storage",
        "checkpoint_loader.load_weights:checkpoint",
        "checkpoint_loader.get_initialized_weight_mapper",
        "call_load_weights",
        "model.load_weights",
        "post_load_weights",
        "moe.register",
        "moe.finalize",
        "move_untracked:2",
        "synchronize",
        "empty_cache",
    ]


def test_move_untracked_params_only_touches_cuda_parameters(monkeypatch):
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    copies: list[str] = []
    create_calls: list[tuple[int, str]] = []

    class FakeStorage:
        def __init__(self, ptr: int, size: int):
            self._ptr = ptr
            self._size = size

        def data_ptr(self):
            return self._ptr

        def nbytes(self):
            return self._size

    class FakeTensor:
        def __init__(
            self,
            label: str,
            storage_ptr: int,
            data_ptr: int,
            *,
            is_cuda: bool = True,
            storage_nbytes: int = 32,
        ):
            self.label = label
            self._storage = FakeStorage(storage_ptr, storage_nbytes)
            self._data_ptr = data_ptr
            self.is_cuda = is_cuda
            self.shape = (4,)
            self._stride = (1,)
            self.dtype = "float16"
            self.data = f"original:{label}"

        def storage(self):
            return self._storage

        def untyped_storage(self):
            return self._storage

        def data_ptr(self):
            return self._data_ptr

        def stride(self):
            return self._stride

        def numel(self):
            return 4

        def element_size(self):
            return 2

    class FakeReplacement:
        def __init__(self, ptr: int):
            self.ptr = ptr

        def copy_(self, tensor):
            copies.append(f"{tensor.label}->{self.ptr}")
            return self

    class FakeClient:
        mappings = {}

        def __init__(self):
            self._next_ptr = 10_000

        def create_mapping(self, size: int, tag: str):
            create_calls.append((size, tag))
            ptr = self._next_ptr
            self._next_ptr += 1_000
            return ptr

    parameter_a = FakeTensor(
        "parameter_a", storage_ptr=100, data_ptr=100, storage_nbytes=64
    )
    parameter_b = FakeTensor(
        "parameter_b", storage_ptr=100, data_ptr=108, storage_nbytes=64
    )
    late_parameter = FakeTensor("late_parameter", storage_ptr=200, data_ptr=200)
    buffer_tensor = FakeTensor("buffer", storage_ptr=300, data_ptr=300)
    tensor_attr = FakeTensor("tensor_attr", storage_ptr=400, data_ptr=400)
    cpu_parameter = FakeTensor(
        "cpu_parameter", storage_ptr=500, data_ptr=500, is_cuda=False
    )
    already_gms = FakeTensor("already_gms", storage_ptr=900, data_ptr=900)

    monkeypatch.setattr(
        model_loader,
        "_ptr_in_gms",
        lambda _client, ptr: ptr == 900,
    )

    fake_torch_module = types.ModuleType("gpu_memory_service.client.torch.module")
    fake_torch_module._iter_module_tensors = lambda _model: [
        ("parameter_a", parameter_a, "parameter"),
        ("parameter_b", parameter_b, "parameter"),
        ("late_parameter", late_parameter, "parameter"),
        ("buffer", buffer_tensor, "buffer"),
        ("tensor_attr", tensor_attr, "tensor_attr"),
        ("cpu_parameter", cpu_parameter, "parameter"),
        ("already_gms", already_gms, "parameter"),
    ]
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.client.torch.module",
        fake_torch_module,
    )

    fake_torch_tensor = types.ModuleType("gpu_memory_service.client.torch.tensor")
    fake_torch_tensor._tensor_from_pointer = (
        lambda ptr, *_args, **_kwargs: FakeReplacement(ptr)
    )
    monkeypatch.setitem(
        sys.modules,
        "gpu_memory_service.client.torch.tensor",
        fake_torch_tensor,
    )

    model_loader._move_untracked_params(object(), FakeClient(), device_index=7)

    assert create_calls == [(64, "weights"), (32, "weights")]
    assert copies == [
        "parameter_a->10000",
        "parameter_b->10008",
        "late_parameter->11000",
    ]
    assert parameter_a.data.ptr == 10_000
    assert parameter_b.data.ptr == 10_008
    assert late_parameter.data.ptr == 11_000
    assert buffer_tensor.data == "original:buffer"
    assert tensor_attr.data == "original:tensor_attr"
    assert cpu_parameter.data == "original:cpu_parameter"
    assert already_gms.data == "original:already_gms"


def test_finalize_pending_gms_write_commits_and_clears_slot(monkeypatch):
    import gpu_memory_service.integrations.trtllm.model_loader as model_loader

    fake_client = object()
    fake_model = object()
    calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        model_loader,
        "finalize_gms_write",
        lambda client, model: calls.append((client, model)) or 123,
    )

    model_loader._pending_gms_write = (fake_client, fake_model)

    assert model_loader.finalize_pending_gms_write() == 123
    assert calls == [(fake_client, fake_model)]
    assert model_loader._pending_gms_write is None
    assert model_loader._last_imported_weights_bytes == 123
