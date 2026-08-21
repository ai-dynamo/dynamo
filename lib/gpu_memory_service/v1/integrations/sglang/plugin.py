# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SGLang plugin adapter for the process-local GMS V1 Torch MemPool client."""

from __future__ import annotations

import os
from contextlib import nullcontext
from functools import cache

import torch
from gpu_memory_service.v1.client.mempool import TorchMempoolMemoryClient
from sglang.srt.plugins.hook_registry import HookRegistry, HookType
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter


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


def register_gms_v1_plugin() -> None:
    """Register the GMS hooks in SGLang processes where Dynamo enabled them."""
    if os.environ.get("DYN_GMS_USE_V1") != "true":
        return

    def around_create_dsa_index_buffers(original, *args, **kwargs):
        with _adapter().region("kv_cache"):
            return original(*args, **kwargs)

    HookRegistry.register(
        "sglang.srt.model_executor.model_runner_components.load_model_utils."
        "load_model_with_memory_saver",
        lambda result, *args, **kwargs: _adapter().observe_model(result.model),
        HookType.AFTER,
    )
    HookRegistry.register(
        "sglang.srt.managers.scheduler.Scheduler.init_all_cuda_graphs",
        lambda _scheduler: _adapter()._publish_weights(),
        HookType.BEFORE,
    )
    HookRegistry.register(
        "sglang.srt.utils.torch_memory_saver_adapter.TorchMemorySaverAdapter.create",
        lambda _original, *_args, **_kwargs: _adapter(),
        HookType.AROUND,
    )
    HookRegistry.register(
        "sglang.srt.mem_cache.memory_pool.DSATokenToKVPool._create_index_buffers",
        around_create_dsa_index_buffers,
        HookType.AROUND,
    )
    # Layer-split DSA overrides the parent method, so the parent hook does not
    # wrap index_k_with_scale_buffer or remote_index_k_with_scale_buffer.
    HookRegistry.register(
        "sglang.srt.mem_cache.dsa_cache_layer_split."
        "LayerSplitDSATokenToKVPool._create_index_buffers",
        around_create_dsa_index_buffers,
        HookType.AROUND,
    )
