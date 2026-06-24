# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sys
import types
from types import SimpleNamespace

import pytest

from dynamo.vllm.flashinfer_collectives import (
    DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES,
    configure_flashinfer_only_collectives,
    install_no_nccl_snapshot_guard,
    install_strict_flashinfer_collective_guard,
    patch_vllm_distributed_backend_for_snapshot,
    patch_vllm_flashinfer_allreduce_thresholds,
)
from dynamo.vllm.snapshot_worker_config import (
    DEFAULT_NO_NCCL_ALL2ALL_BACKEND,
    DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER,
    DYN_VLLM_NO_NCCL_SNAPSHOT,
    SNAPSHOT_WORKER_CLASS,
    configure_no_nccl_snapshot_before_engine_config,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def clear_collective_env(monkeypatch):
    for name in (
        DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES,
        DYN_VLLM_NO_NCCL_SNAPSHOT,
        DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER,
        "VLLM_ALLREDUCE_USE_FLASHINFER",
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "VLLM_USE_NCCL_SYMM_MEM",
        "VLLM_DISABLE_PYNCCL",
        "VLLM_DISTRIBUTED_USE_SPLIT_GROUP",
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
        "DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP",
    ):
        monkeypatch.delenv(name, raising=False)


def _config(
    *,
    tensor_parallel_size=4,
    data_parallel_size=1,
    enable_expert_parallel=False,
    all2all_backend="flashinfer_nvlink_two_sided",
    is_moe=True,
    worker_cls=SNAPSHOT_WORKER_CLASS,
):
    engine_args = SimpleNamespace(
        disable_custom_all_reduce=False,
        disable_nccl_for_dp_synchronization=False,
        tensor_parallel_size=tensor_parallel_size,
        worker_cls=worker_cls,
    )
    parallel_config = SimpleNamespace(
        tensor_parallel_size=tensor_parallel_size,
        prefill_context_parallel_size=1,
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        all2all_backend=all2all_backend,
        disable_custom_all_reduce=False,
        disable_nccl_for_dp_synchronization=False,
        is_moe_model=is_moe,
        worker_cls=worker_cls,
    )
    vllm_config = SimpleNamespace(
        parallel_config=parallel_config,
        model_config=SimpleNamespace(is_moe=is_moe),
    )
    return engine_args, vllm_config


class _FakeTensor:
    dtype = "bfloat16"
    shape = (8192, 4096)

    def __init__(self, device="cuda:0"):
        self.device = device
        self.is_cuda = str(device).startswith("cuda")

    def is_contiguous(self):
        return True

    def numel(self):
        result = 1
        for dim in self.shape:
            result *= dim
        return result


def _install_fake_cuda_communicator(monkeypatch):
    module = types.ModuleType(
        "vllm.distributed.device_communicators.cuda_communicator"
    )

    class CudaCommunicator:
        original_calls = []

        def all_reduce(self, input_):
            self.original_calls.append(("all_reduce", input_))
            return "unsafe-fallback"

        def all_gather(self, input_, dim=-1):
            return ("all_gather", input_, dim)

        def all_gatherv(self, input_, dim=0, sizes=None):
            return ("all_gatherv", input_, dim, sizes)

        def broadcast(self, input_, src=0):
            return ("broadcast", input_, src)

        def gather(self, input_, dst=0, dim=-1):
            return ("gather", input_, dst, dim)

        def reduce_scatter(self, input_, dim=-1):
            return ("reduce_scatter", input_, dim)

        def reduce_scatterv(self, input_, dim=-1, sizes=None):
            return ("reduce_scatterv", input_, dim, sizes)

        def recv(self, tensor, src=None):
            return ("recv", tensor, src)

        def send(self, tensor, dst=None):
            return ("send", tensor, dst)

        def batch_isend_irecv(self, p2p_ops):
            return ("batch_isend_irecv", p2p_ops)

    module.CudaCommunicator = CudaCommunicator
    monkeypatch.setitem(
        sys.modules,
        "vllm.distributed.device_communicators.cuda_communicator",
        module,
    )
    _install_fake_torch_distributed(monkeypatch)
    return CudaCommunicator


def _install_fake_torch_distributed(monkeypatch):
    torch_module = types.ModuleType("torch")
    torch_module.__path__ = []
    distributed_module = types.ModuleType("torch.distributed")
    distributed_module.calls = []

    def record_call(name):
        def fn(*args, **kwargs):
            distributed_module.calls.append((name, args, kwargs))
            return f"{name}-ok"

        return fn

    for name in (
        "init_process_group",
        "new_group",
        "split_group",
        "all_reduce",
        "broadcast",
        "gather",
        "send",
        "recv",
        "isend",
        "irecv",
        "all_gather",
        "all_gather_into_tensor",
        "all_to_all",
        "all_to_all_single",
        "reduce_scatter",
        "reduce_scatter_tensor",
        "scatter",
        "reduce",
        "batch_isend_irecv",
    ):
        setattr(distributed_module, name, record_call(name))

    def get_backend(group=None):
        return getattr(group, "backend", "gloo")

    distributed_module.get_backend = get_backend
    c10d_module = types.ModuleType("torch.distributed.distributed_c10d")
    c10d_module.calls = distributed_module.calls
    for name in ("init_process_group", "new_group", "split_group"):
        setattr(c10d_module, name, record_call(f"c10d.{name}"))
    distributed_module.distributed_c10d = c10d_module
    torch_module.distributed = distributed_module
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.distributed", distributed_module)
    monkeypatch.setitem(
        sys.modules,
        "torch.distributed.distributed_c10d",
        c10d_module,
    )
    return distributed_module


def _install_fake_vllm_pass_config(monkeypatch, defaults=None):
    defaults = {4: 32} if defaults is None else defaults
    compilation_module = types.ModuleType("vllm.config.compilation")

    class PassConfig:
        @staticmethod
        def default_fi_allreduce_fusion_max_size_mb():
            return defaults

    compilation_module.PassConfig = PassConfig
    monkeypatch.setitem(sys.modules, "vllm.config.compilation", compilation_module)
    return PassConfig


def _install_fake_vllm_platform(monkeypatch, backend="nccl"):
    platforms_module = types.ModuleType("vllm.platforms")
    current_platform = SimpleNamespace(dist_backend=backend)
    platforms_module.current_platform = current_platform
    monkeypatch.setitem(sys.modules, "vllm.platforms", platforms_module)
    return current_platform


def _install_fake_snapshot_worker(monkeypatch):
    base_cls = type("Worker", (), {})
    snapshot_cls = type("SnapshotWorker", (base_cls,), {})
    gpu_worker_module = types.ModuleType("vllm.v1.worker.gpu_worker")
    gpu_worker_module.Worker = base_cls
    snapshot_worker_module = types.ModuleType("dynamo.vllm.snapshot_worker")
    snapshot_worker_module.SnapshotWorker = snapshot_cls
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_worker", gpu_worker_module)
    monkeypatch.setitem(
        sys.modules, "dynamo.vllm.snapshot_worker", snapshot_worker_module
    )


def test_flashinfer_collectives_env_unset_is_noop(monkeypatch):
    monkeypatch.delenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, raising=False)
    engine_args, vllm_config = _config()

    assert configure_flashinfer_only_collectives(engine_args, vllm_config) is False
    assert vllm_config.parallel_config.disable_custom_all_reduce is False
    assert (
        vllm_config.parallel_config.disable_nccl_for_dp_synchronization is False
    )


def test_flashinfer_collectives_requires_snapshot_worker_env(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config()

    with pytest.raises(ValueError, match=DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER):
        configure_flashinfer_only_collectives(engine_args, vllm_config)


def test_flashinfer_collectives_requires_configured_snapshot_worker(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(worker_cls="vllm.v1.worker.gpu_worker.Worker")

    with pytest.raises(ValueError, match="SnapshotWorker"):
        configure_flashinfer_only_collectives(engine_args, vllm_config)


def test_flashinfer_collectives_before_engine_config_sets_safe_flags(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_snapshot_worker(monkeypatch)
    monkeypatch.setattr(
        "dynamo.vllm.flashinfer_collectives.patch_vllm_distributed_backend_for_snapshot",
        lambda: True,
    )
    engine_args = SimpleNamespace(
        disable_custom_all_reduce=False,
        disable_nccl_for_dp_synchronization=False,
        tensor_parallel_size=4,
        all2all_backend="allgather_reducescatter",
        worker_cls="auto",
        load_format="auto",
    )

    config = SimpleNamespace(engine_args=engine_args)

    assert configure_no_nccl_snapshot_before_engine_config(config)

    assert engine_args.disable_custom_all_reduce is True
    assert engine_args.disable_nccl_for_dp_synchronization is True
    assert engine_args.all2all_backend == DEFAULT_NO_NCCL_ALL2ALL_BACKEND
    assert os_environ_subset(
        {
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "VLLM_USE_NCCL_SYMM_MEM": "0",
            "VLLM_DISABLE_PYNCCL": "1",
            "VLLM_DISTRIBUTED_USE_SPLIT_GROUP": "0",
        }
    )
    assert engine_args.worker_cls == SNAPSHOT_WORKER_CLASS
    assert "VLLM_ALLREDUCE_USE_FLASHINFER" not in os.environ
    assert "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB" not in os.environ


def test_flashinfer_collectives_env_set_makes_config_safe(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(tensor_parallel_size=4)

    assert configure_flashinfer_only_collectives(engine_args, vllm_config) is True

    assert vllm_config.parallel_config.disable_custom_all_reduce is True
    assert (
        vllm_config.parallel_config.disable_nccl_for_dp_synchronization is True
    )
    assert engine_args.disable_custom_all_reduce is True
    assert engine_args.disable_nccl_for_dp_synchronization is True
    assert os_environ_subset(
        {
            "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "VLLM_USE_NCCL_SYMM_MEM": "0",
            "VLLM_DISABLE_PYNCCL": "1",
        }
    )
    assert json.loads(
        os_environ("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB")
    ) == {"4": 128}


@pytest.mark.parametrize(
    "all2all_backend",
    (
        "flashinfer_all2allv",
        "flashinfer_nvlink_one_sided",
        "flashinfer_nvlink_two_sided",
    ),
)
def test_flashinfer_collectives_preserves_explicit_threshold_and_backend(
    monkeypatch, all2all_backend
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv(
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
        '{"4": 256}',
    )
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(
        data_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend=all2all_backend,
    )

    assert configure_flashinfer_only_collectives(engine_args, vllm_config) is True

    assert vllm_config.parallel_config.all2all_backend == all2all_backend
    assert os_environ("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB") == '{"4": 256}'


def test_flashinfer_collectives_forces_unsafe_vllm_env(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "0")
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "1")
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "1")
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "0")
    monkeypatch.setenv(
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
        '{"4": 512}',
    )
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(tensor_parallel_size=4)

    assert configure_flashinfer_only_collectives(engine_args, vllm_config) is True

    assert os_environ_subset(
        {
            "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "VLLM_USE_NCCL_SYMM_MEM": "0",
            "VLLM_DISABLE_PYNCCL": "1",
        }
    )
    assert (
        os_environ("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB")
        == '{"4": 512}'
    )


def test_flashinfer_collectives_keeps_already_safe_vllm_env(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_FLASHINFER", "1")
    monkeypatch.setenv("VLLM_ALLREDUCE_USE_SYMM_MEM", "0")
    monkeypatch.setenv("VLLM_USE_NCCL_SYMM_MEM", "0")
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "1")
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(tensor_parallel_size=4)

    assert configure_flashinfer_only_collectives(engine_args, vllm_config) is True

    assert os_environ_subset(
        {
            "VLLM_ALLREDUCE_USE_FLASHINFER": "1",
            "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
            "VLLM_USE_NCCL_SYMM_MEM": "0",
            "VLLM_DISABLE_PYNCCL": "1",
        }
    )


def test_flashinfer_threshold_patch_merges_env_override(monkeypatch):
    monkeypatch.setenv(
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
        '{"4": 128, "8": 2.5}',
    )
    PassConfig = _install_fake_vllm_pass_config(monkeypatch, defaults={2: 64, 4: 32})

    assert patch_vllm_flashinfer_allreduce_thresholds() is True
    assert PassConfig.default_fi_allreduce_fusion_max_size_mb() == {
        2: 64,
        4: 128.0,
        8: 2.5,
    }
    assert patch_vllm_flashinfer_allreduce_thresholds() is False


@pytest.mark.parametrize(
    "raw_env",
    (
        "not-json",
        "[]",
        '{"0": 128}',
        '{"4.5": 128}',
        '{"4": 0}',
        '{"4": -1}',
        '{"4": "128"}',
        '{"4": true}',
    ),
)
def test_flashinfer_threshold_patch_rejects_invalid_env(monkeypatch, raw_env):
    monkeypatch.setenv("VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB", raw_env)
    _install_fake_vllm_pass_config(monkeypatch)

    with pytest.raises(
        ValueError,
        match="positive integer|positive numeric MB threshold|JSON object",
    ):
        patch_vllm_flashinfer_allreduce_thresholds()


def test_strict_backend_patch_changes_vllm_cuda_backend_to_gloo(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    current_platform = _install_fake_vllm_platform(monkeypatch, backend="nccl")

    assert patch_vllm_distributed_backend_for_snapshot() is True
    assert current_platform.dist_backend == "gloo"
    assert patch_vllm_distributed_backend_for_snapshot() is False


def test_strict_backend_patch_can_be_disabled(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    current_platform = _install_fake_vllm_platform(monkeypatch, backend="nccl")
    monkeypatch.setenv("DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP", "0")

    assert patch_vllm_distributed_backend_for_snapshot() is False
    assert current_platform.dist_backend == "nccl"


def test_no_nccl_backend_patch_rejects_disabled_gloo_patch(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    current_platform = _install_fake_vllm_platform(monkeypatch, backend="nccl")
    monkeypatch.setenv("DYN_VLLM_FLASHINFER_AVOID_NCCL_PROCESS_GROUP", "0")

    with pytest.raises(ValueError, match="requires vLLM platform"):
        patch_vllm_distributed_backend_for_snapshot()

    assert current_platform.dist_backend == "nccl"


@pytest.mark.parametrize(
    ("function_name", "kwargs"),
    (
        ("init_process_group", {"backend": "cpu:gloo,cuda:nccl"}),
        ("new_group", {"backend": "nccl"}),
        ("split_group", {"backend": "cpu:gloo,cuda:nccl"}),
    ),
)
def test_no_nccl_torch_guard_blocks_nccl_backend_strings(
    monkeypatch, function_name, kwargs
):
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)

    assert install_no_nccl_snapshot_guard() is True

    with pytest.raises(RuntimeError, match=function_name):
        getattr(distributed, function_name)(**kwargs)

    assert distributed.calls == []


def test_no_nccl_torch_guard_blocks_c10d_split_group_nccl_backend(monkeypatch):
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)

    assert install_no_nccl_snapshot_guard() is True

    with pytest.raises(RuntimeError, match="split_group"):
        distributed.distributed_c10d.split_group(backend="cuda:nccl")

    assert distributed.calls == []


def test_no_nccl_torch_guard_blocks_default_backend_with_cuda_device_id(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_NO_NCCL_SNAPSHOT, "1")
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)

    assert install_no_nccl_snapshot_guard() is True

    with pytest.raises(RuntimeError, match="device_id='cuda:0'"):
        distributed.init_process_group(device_id="cuda:0")

    assert distributed.init_process_group(backend=None) == "init_process_group-ok"
    assert [call[0] for call in distributed.calls] == ["init_process_group"]


def test_backend_patch_is_noop_outside_strict_mode(monkeypatch):
    current_platform = _install_fake_vllm_platform(monkeypatch, backend="nccl")

    assert patch_vllm_distributed_backend_for_snapshot() is False
    assert current_platform.dist_backend == "nccl"


def test_flashinfer_collectives_rejects_unsafe_all2all_when_active(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(
        data_parallel_size=2,
        enable_expert_parallel=True,
        all2all_backend="allgather_reducescatter",
    )

    with pytest.raises(ValueError, match="FlashInfer NVLink MoE all2all"):
        configure_flashinfer_only_collectives(engine_args, vllm_config)


@pytest.mark.parametrize(
    ("enable_expert_parallel", "data_parallel_size", "is_moe"),
    (
        (False, 2, True),
        (True, 1, True),
        (True, 2, False),
    ),
)
def test_flashinfer_collectives_allows_unsafe_all2all_when_inactive(
    monkeypatch, enable_expert_parallel, data_parallel_size, is_moe
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_SNAPSHOT_WORKER, "1")
    _install_fake_cuda_communicator(monkeypatch)
    engine_args, vllm_config = _config(
        data_parallel_size=data_parallel_size,
        enable_expert_parallel=enable_expert_parallel,
        all2all_backend="allgather_reducescatter",
        is_moe=is_moe,
    )

    assert configure_flashinfer_only_collectives(engine_args, vllm_config) is True


def test_strict_all_reduce_uses_flashinfer_and_is_idempotent(monkeypatch):
    CudaCommunicator = _install_fake_cuda_communicator(monkeypatch)

    assert install_strict_flashinfer_collective_guard() is True
    patched_all_reduce = CudaCommunicator.all_reduce
    patched_all_gather = CudaCommunicator.all_gather
    assert install_strict_flashinfer_collective_guard() is False
    assert CudaCommunicator.all_reduce is patched_all_reduce
    assert CudaCommunicator.all_gather is patched_all_gather

    class FakeFlashInferAllReduce:
        disabled = False

        def __init__(self):
            self.should_use = True
            self.calls = []

        def should_use_fi_ar(self, input_):
            self.calls.append(("should", input_))
            return self.should_use

        def all_reduce(self, input_):
            self.calls.append(("all_reduce", input_))
            return "flashinfer-result"

    comm = CudaCommunicator()
    comm.world_size = 4
    comm.unique_name = "tp:0"
    comm.rank = 0
    comm.rank_in_group = 0
    comm.use_flashinfer_allreduce = True
    comm.fi_ar_comm = FakeFlashInferAllReduce()
    tensor = _FakeTensor()

    assert comm.all_reduce(tensor) == "flashinfer-result"
    assert comm.fi_ar_comm.calls == [("should", tensor), ("all_reduce", tensor)]
    assert comm.original_calls == []

    comm.fi_ar_comm.should_use = False
    with pytest.raises(RuntimeError, match="FlashInfer allreduce rejected"):
        comm.all_reduce(tensor)
    assert comm.original_calls == []


def test_strict_all_reduce_rejection_diagnostic_includes_thresholds(
    monkeypatch,
):
    monkeypatch.setenv(DYN_VLLM_FLASHINFER_ONLY_COLLECTIVES, "1")
    CudaCommunicator = _install_fake_cuda_communicator(monkeypatch)
    _install_fake_vllm_pass_config(monkeypatch, defaults={4: 32})
    _install_fake_vllm_platform(monkeypatch, backend="nccl")
    monkeypatch.setenv(
        "VLLM_FLASHINFER_ALLREDUCE_FUSION_THRESHOLDS_MB",
        '{"4": 128}',
    )
    install_strict_flashinfer_collective_guard()

    class FakeFlashInferAllReduce:
        disabled = False
        world_size = 4
        rank = 0
        max_workspace_size = 128 * 1024 * 1024
        max_num_tokens = 4096

        def should_use_fi_ar(self, input_):
            return False

    comm = CudaCommunicator()
    comm.world_size = 4
    comm.unique_name = "tp:0"
    comm.rank = 0
    comm.rank_in_group = 0
    comm.use_flashinfer_allreduce = True
    comm.fi_ar_comm = FakeFlashInferAllReduce()

    with pytest.raises(RuntimeError) as exc_info:
        comm.all_reduce(_FakeTensor())

    message = str(exc_info.value)
    assert "numel=33554432" in message
    assert "bytes=67108864" in message
    assert "MiB=64.0" in message
    assert "max_workspace_size=134217728 bytes (128.000 MiB)" in message
    assert "max_num_tokens=4096" in message
    assert 'threshold_env_raw=\'{"4": 128}\'' in message
    assert "threshold_env_effective={4: 128.0}" in message
    assert "backend='gloo'" in message


def test_strict_cuda_collectives_raise_for_multi_rank(monkeypatch):
    CudaCommunicator = _install_fake_cuda_communicator(monkeypatch)
    install_strict_flashinfer_collective_guard()

    comm = CudaCommunicator()
    comm.world_size = 2
    comm.unique_name = "ep:0"
    comm.rank = 0
    comm.rank_in_group = 0
    comm.use_flashinfer_allreduce = False
    tensor = _FakeTensor()

    for method_name, args in (
        ("all_gather", (tensor,)),
        ("all_gatherv", (tensor,)),
        ("broadcast", (tensor,)),
        ("gather", (tensor,)),
        ("reduce_scatter", (tensor,)),
        ("reduce_scatterv", (tensor,)),
        ("recv", (tensor,)),
        ("send", (tensor,)),
        ("batch_isend_irecv", ([],)),
    ):
        with pytest.raises(RuntimeError, match=method_name):
            getattr(comm, method_name)(*args)

    comm.world_size = 1
    assert comm.all_gather(tensor) is tensor
    assert comm.all_gatherv(tensor) is tensor
    assert comm.broadcast(tensor) is tensor
    assert comm.gather(tensor) is tensor
    assert comm.reduce_scatter(tensor) is tensor
    assert comm.reduce_scatterv(tensor) is tensor
    assert comm.recv(tensor) is tensor
    assert comm.send(tensor) is tensor
    assert comm.batch_isend_irecv([]) is None


def test_strict_torch_distributed_guard_allows_cpu_and_gloo(monkeypatch):
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)
    install_strict_flashinfer_collective_guard()

    cpu_group = SimpleNamespace(backend="gloo")
    cpu_tensor = _FakeTensor(device="cpu")

    assert distributed.broadcast(cpu_tensor, 0, group=cpu_group) == "broadcast-ok"
    assert [call[0] for call in distributed.calls] == ["broadcast"]


def test_strict_torch_distributed_guard_rejects_cuda_on_gloo(monkeypatch):
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)
    install_strict_flashinfer_collective_guard()

    cpu_group = SimpleNamespace(backend="gloo")

    with pytest.raises(RuntimeError, match="torch.distributed.broadcast"):
        distributed.broadcast(_FakeTensor(device="cuda:0"), 0, group=cpu_group)

    assert distributed.calls == []


@pytest.mark.parametrize(
    ("function_name", "args"),
    (
        ("broadcast", lambda tensor, group: (tensor, 0, group)),
        ("all_gather", lambda tensor, group: ([tensor], tensor, group)),
        ("all_to_all", lambda tensor, group: ([tensor], [tensor], group)),
        ("all_to_all_single", lambda tensor, group: (tensor, tensor, group)),
        ("gather", lambda tensor, group: (tensor, [tensor], 0, group)),
        ("reduce_scatter", lambda tensor, group: (tensor, [tensor], None, group)),
        ("scatter", lambda tensor, group: (tensor, [tensor], 0, group)),
        ("reduce", lambda tensor, group: (tensor, 0, None, group)),
        ("send", lambda tensor, group: (tensor, 1, group)),
        ("recv", lambda tensor, group: (tensor, 0, group)),
        ("isend", lambda tensor, group: (tensor, 1, group)),
        ("irecv", lambda tensor, group: (tensor, 0, group)),
        (
            "batch_isend_irecv",
            lambda tensor, group: ([SimpleNamespace(tensor=tensor, group=group)],),
        ),
    ),
)
def test_strict_torch_distributed_guard_rejects_cuda_nccl(
    monkeypatch, function_name, args
):
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)
    install_strict_flashinfer_collective_guard()

    nccl_group = SimpleNamespace(backend="nccl")
    tensor = _FakeTensor(device="cuda:0")

    with pytest.raises(RuntimeError, match=f"torch.distributed.{function_name}"):
        getattr(distributed, function_name)(*args(tensor, nccl_group))

    assert distributed.calls == []


def test_strict_torch_distributed_guard_rejects_device_group_with_gloo_backend(
    monkeypatch,
):
    _install_fake_cuda_communicator(monkeypatch)
    distributed = _install_fake_torch_distributed(monkeypatch)
    install_strict_flashinfer_collective_guard()

    device_group = SimpleNamespace(backend="gloo", is_device_group=True)

    with pytest.raises(RuntimeError, match="torch.distributed.broadcast"):
        distributed.broadcast(_FakeTensor(device="cuda:0"), 0, group=device_group)


def os_environ(name):
    return os.environ[name]


def os_environ_subset(expected):
    return all(os.environ.get(name) == value for name, value in expected.items())
