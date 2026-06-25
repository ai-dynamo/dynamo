# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib
import sys
import types
from types import SimpleNamespace

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class _Tensor:
    def __init__(self) -> None:
        self.zero_count = 0

    def zero_(self) -> None:
        self.zero_count += 1


def _install_fake_torch(monkeypatch):
    torch_module = types.ModuleType("torch")
    torch_module.Tensor = _Tensor
    monkeypatch.setitem(sys.modules, "torch", torch_module)


def _install_fake_gpu_model_runner(monkeypatch):
    class GPUModelRunner:
        def __init__(self, cache_dtype="fp8") -> None:
            self.cache_config = SimpleNamespace(cache_dtype=cache_dtype)
            self.observed_kv_caches = None

        def init_fp8_kv_scales(self):
            self.observed_kv_caches = self.kv_caches
            for cache_tensor in self.kv_caches:
                if cache_tensor is not None:
                    cache_tensor.zero_()
            return "original-result"

    vllm_module = types.ModuleType("vllm")
    vllm_module.__path__ = []
    vllm_v1_module = types.ModuleType("vllm.v1")
    vllm_v1_module.__path__ = []
    vllm_worker_module = types.ModuleType("vllm.v1.worker")
    vllm_worker_module.__path__ = []
    vllm_utils_module = types.ModuleType("vllm.utils")
    vllm_utils_module.__path__ = []
    torch_utils_module = types.ModuleType("vllm.utils.torch_utils")
    torch_utils_module.is_quantized_kv_cache = (
        lambda cache_dtype: cache_dtype.startswith("fp8")
        or cache_dtype.endswith("per_token_head")
        or cache_dtype == "nvfp4"
    )
    gpu_model_runner_module = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    gpu_model_runner_module.GPUModelRunner = GPUModelRunner

    for module_name in (
        "dynamo.vllm.snapshot_compat",
        "vllm.v1.worker.gpu_model_runner",
        "vllm.v1.worker",
        "vllm.v1",
        "vllm.utils.torch_utils",
        "vllm.utils",
        "vllm",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1_module)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker", vllm_worker_module)
    monkeypatch.setitem(sys.modules, "vllm.utils", vllm_utils_module)
    monkeypatch.setitem(sys.modules, "vllm.utils.torch_utils", torch_utils_module)
    monkeypatch.setitem(
        sys.modules, "vllm.v1.worker.gpu_model_runner", gpu_model_runner_module
    )
    return GPUModelRunner


def test_patch_flattens_nested_fp8_kv_caches_and_restores(monkeypatch):
    _install_fake_torch(monkeypatch)
    gpu_model_runner = _install_fake_gpu_model_runner(monkeypatch)
    snapshot_compat = importlib.import_module("dynamo.vllm.snapshot_compat")

    original_method = gpu_model_runner.init_fp8_kv_scales
    assert snapshot_compat.patch_vllm_quantized_kv_cache_wake_up() is True

    runner = gpu_model_runner()
    leaf_1 = _Tensor()
    leaf_2 = _Tensor()
    original_kv_caches = [leaf_1, [None, (leaf_2,)]]
    runner.kv_caches = original_kv_caches

    assert runner.init_fp8_kv_scales() == "original-result"

    assert runner.kv_caches is original_kv_caches
    assert runner.observed_kv_caches == [leaf_1, leaf_2]
    assert leaf_1.zero_count == 1
    assert leaf_2.zero_count == 1
    assert runner.init_fp8_kv_scales.__name__ == original_method.__name__


@pytest.mark.parametrize(
    "cache_dtype", ["fp8", "fp8_e4m3", "fp8_e4m3_per_token_head", "nvfp4"]
)
def test_patch_flattens_quantized_kv_caches(monkeypatch, cache_dtype):
    _install_fake_torch(monkeypatch)
    gpu_model_runner = _install_fake_gpu_model_runner(monkeypatch)
    snapshot_compat = importlib.import_module("dynamo.vllm.snapshot_compat")
    snapshot_compat.patch_vllm_quantized_kv_cache_wake_up()

    runner = gpu_model_runner(cache_dtype=cache_dtype)
    leaf = _Tensor()
    original_kv_caches = [[leaf]]
    runner.kv_caches = original_kv_caches

    assert runner.init_fp8_kv_scales() == "original-result"

    assert runner.kv_caches is original_kv_caches
    assert runner.observed_kv_caches == [leaf]
    assert leaf.zero_count == 1


def test_patch_is_idempotent(monkeypatch):
    _install_fake_torch(monkeypatch)
    gpu_model_runner = _install_fake_gpu_model_runner(monkeypatch)
    snapshot_compat = importlib.import_module("dynamo.vllm.snapshot_compat")

    assert snapshot_compat.patch_vllm_quantized_kv_cache_wake_up() is True
    patched = gpu_model_runner.init_fp8_kv_scales

    assert snapshot_compat.patch_vllm_quantized_kv_cache_wake_up() is True

    assert gpu_model_runner.init_fp8_kv_scales is patched


def test_patch_leaves_non_fp8_cache_path_unchanged(monkeypatch):
    _install_fake_torch(monkeypatch)
    gpu_model_runner = _install_fake_gpu_model_runner(monkeypatch)
    snapshot_compat = importlib.import_module("dynamo.vllm.snapshot_compat")
    snapshot_compat.patch_vllm_quantized_kv_cache_wake_up()

    runner = gpu_model_runner(cache_dtype="auto")
    original_kv_caches = [_Tensor(), [_Tensor()]]
    runner.kv_caches = original_kv_caches

    with pytest.raises(AttributeError, match="zero_"):
        runner.init_fp8_kv_scales()

    assert runner.kv_caches is original_kv_caches
    assert runner.observed_kv_caches is original_kv_caches


def test_patch_rejects_unsupported_nested_fp8_cache_leaf(monkeypatch):
    _install_fake_torch(monkeypatch)
    gpu_model_runner = _install_fake_gpu_model_runner(monkeypatch)
    snapshot_compat = importlib.import_module("dynamo.vllm.snapshot_compat")
    snapshot_compat.patch_vllm_quantized_kv_cache_wake_up()

    runner = gpu_model_runner()
    original_kv_caches = [_Tensor(), [object()]]
    runner.kv_caches = original_kv_caches

    with pytest.raises(TypeError, match="Unsupported KV cache entry type"):
        runner.init_fp8_kv_scales()

    assert runner.kv_caches is original_kv_caches
