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

_FACTORY_TARGET = (
    "sglang.srt.utils.torch_memory_saver_adapter.TorchMemorySaverAdapter.create"
)
_INITIAL_MODEL_LOAD_TARGET = (
    "sglang.srt.model_executor.model_runner_components.load_model_utils."
    "load_model_with_memory_saver"
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
            self.client.publish_weights(self._models)
            self.client.suspend()

    def resume(self, tag: str) -> None:
        if tag == "weights":
            self.client.resume()

    @property
    def enabled(self) -> bool:
        return True

    def observe_model(self, model: object) -> None:
        self._models.append(model)


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


def register_gms_v1_plugin() -> None:
    """Register the GMS hooks in SGLang processes where Dynamo enabled them."""
    if os.environ.get("DYN_SGL_ENABLE_GMS_V1") != "true":
        return
    HookRegistry.register(
        _INITIAL_MODEL_LOAD_TARGET,
        _after_initial_model_load,
        HookType.AFTER,
    )
    HookRegistry.register(
        _FACTORY_TARGET,
        _around_adapter_factory,
        HookType.AROUND,
    )
