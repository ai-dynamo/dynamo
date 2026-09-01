# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang plugin adapter for the process-local GMS V1 Torch MemPool client."""

from __future__ import annotations

import importlib
import os
from contextlib import nullcontext
from functools import cache

import torch
from gpu_memory_service.v1.client.mempool import TorchMempoolMemoryClient
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

_FACTORY_TARGET = (
    "sglang.srt.utils.torch_memory_saver_adapter.TorchMemorySaverAdapter.create"
)
_INITIAL_MODEL_LOAD_TARGET = (
    "sglang.srt.model_executor.model_runner_components.load_model_utils."
    "load_model_with_memory_saver"
)
_INIT_ALL_CUDA_GRAPHS_TARGET = (
    "sglang.srt.managers.scheduler.Scheduler.init_all_cuda_graphs"
)
_RELEASE_MEMORY_OCCUPATION_TARGET = (
    "sglang.srt.managers.scheduler_components.weight_updater."
    "SchedulerWeightUpdaterManager.release_memory_occupation"
)
_CREATE_DSA_INDEX_TARGETS = (
    "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool._create_index_key_cache",
    "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool._create_index_buffers",
)
_CREATE_LAYER_SPLIT_DSA_INDEX_TARGETS = (
    "sglang.srt.mem_cache.dsa_cache_layer_split."
    "LayerSplitDSATokenToKVPool._create_index_key_cache",
    "sglang.srt.mem_cache.dsa_cache_layer_split."
    "LayerSplitDSATokenToKVPool._create_index_buffers",
)


class GMSV1MemorySaverAdapter(TorchMemorySaverAdapter):
    """Share one GMS V1 client across SGLang's process-local adapters."""

    def __init__(self) -> None:
        self._client: TorchMempoolMemoryClient | None = None
        self._models: list[object] = []

    @property
    def client(self) -> TorchMempoolMemoryClient:
        if self._client is None:
            self._client = TorchMempoolMemoryClient()
        return self._client

    def configure_subprocess(self):
        return nullcontext()

    def region(self, tag: str, enable_cpu_backup: bool = False):
        if tag == "weights":
            return self.client.weight_region()
        if tag == "kv_cache":
            return self.client.kv_cache_region()
        return nullcontext()

    def cuda_graph(self, **kwargs):
        kwargs.pop("tag", None)
        kwargs.pop("enable_cpu_backup", None)
        cuda_graph = kwargs.pop("cuda_graph")
        return torch.cuda.graph(cuda_graph, **kwargs)

    def disable(self):
        return self.client.unmanaged_region()

    def pause(self, tag: str) -> None:
        if tag == "weights":
            self._publish_weights()
            self.client.suspend()

    def resume(self, tag: str) -> None:
        if tag == "weights":
            self.client.resume()

    @property
    def enabled(self) -> bool:
        return True

    def observe_model(self, model: object) -> None:
        self._models.append(model)

    def _publish_weights(self) -> None:
        self.client.publish_weights(self._models)


@cache
def _adapter() -> GMSV1MemorySaverAdapter:
    return GMSV1MemorySaverAdapter()


def _around_adapter_factory(original_factory, *args, **kwargs):
    enable = args[0] if args else kwargs["enable"]
    if not enable:
        return original_factory(*args, **kwargs)
    return _adapter()


def _after_initial_model_load(result, *args, **kwargs) -> None:
    _adapter().observe_model(result.model)


def _before_init_all_cuda_graphs(_scheduler: object) -> None:
    # Publication rebinds non-Parameter tensors, so it must precede graph capture.
    _adapter()._publish_weights()


def _after_release_memory_occupation(result, manager, *args, **kwargs):
    torch.distributed.barrier(group=manager.tp_cpu_group)
    return result


def _around_create_dsa_index_cache(original, *args, **kwargs):
    """Allocate DSA index-K storage inside the GMS KV-cache region.

    SGLang wraps the DSA pool's main latent-KV allocation in its own
    ``region(GPU_MEMORY_TYPE_KV_CACHE)``, but not the index-K storage. Without
    this hook those buffers come from the default torch allocator, inside the
    ``mem_fraction_static`` budget that was sized assuming they live in GMS.

    The return value matters: ``_create_index_key_cache`` returns the
    ``IndexKeyCache`` the caller assigns to ``self.index_key_cache``, unlike the
    pre-#28609 ``_create_index_buffers``, which returned nothing.
    """
    with _adapter().region("kv_cache"):
        return original(*args, **kwargs)


def _register_first_resolvable(targets, hook, hook_type) -> None:
    """Register the first target that resolves, or refuse to start.

    ``HookRegistry.apply_hooks`` logs and swallows a target that does not
    resolve, so an upstream rename leaves DSA index-K buffers outside GMS and
    surfaces much later as a CUDA OOM during KV allocation rather than as a
    startup error. Fail loudly instead: a missing hook is a correctness bug, not
    a degraded mode.
    """
    for target in targets:
        module_path, _, attr = target.rpartition(".")
        obj_path, _, cls_name = module_path.rpartition(".")
        try:
            cls = getattr(importlib.import_module(obj_path), cls_name)
        except (ImportError, AttributeError):
            continue
        if hasattr(cls, attr):
            HookRegistry.register(target, hook, hook_type)
            return
    raise RuntimeError(
        f"GMS V1: no DSA index-cache hook target resolved among {targets}. "
        "SGLang's DSA memory-pool API has changed. Refusing to start: skipping "
        "this hook would allocate DSA index buffers outside GMS."
    )


def register_gms_v1_plugin() -> None:
    """Register the GMS hooks in SGLang processes where Dynamo enabled them."""
    if os.environ.get("DYN_GMS_USE_V1") != "true":
        return

    HookRegistry.register(
        _INITIAL_MODEL_LOAD_TARGET,
        _after_initial_model_load,
        HookType.AFTER,
    )
    HookRegistry.register(
        _INIT_ALL_CUDA_GRAPHS_TARGET,
        _before_init_all_cuda_graphs,
        HookType.BEFORE,
    )
    HookRegistry.register(
        _FACTORY_TARGET,
        _around_adapter_factory,
        HookType.AROUND,
    )
    # TP release fence: SGLang can let the response-producing rank acknowledge
    # release_memory_occupation while a peer rank still owns its KV-cache
    # socket, which reaches CRIU with a half-external stream. Barrier on the
    # TP CPU group so every rank has released before publication proceeds.
    HookRegistry.register(
        _RELEASE_MEMORY_OCCUPATION_TARGET,
        _after_release_memory_occupation,
        HookType.AFTER,
    )
    _register_first_resolvable(
        _CREATE_DSA_INDEX_TARGETS,
        _around_create_dsa_index_cache,
        HookType.AROUND,
    )
    # Layer-split DSA overrides the factory, so the parent-class hook never runs
    # for it; it also adds LayerSplitIndexKeyCache.remote_buffer, which the one
    # AROUND on the factory covers because both allocations happen inside it.
    _register_first_resolvable(
        _CREATE_LAYER_SPLIT_DSA_INDEX_TARGETS,
        _around_create_dsa_index_cache,
        HookType.AROUND,
    )
