# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ownership-based GMS V1 worker for vLLM's normal model loader.

Dynamo selects this worker when ``DYN_GMS_USE_V1=true``.
"""

from __future__ import annotations

from contextlib import AbstractContextManager

from gpu_memory_service.v1.integrations.vllm.backend import BACKEND_NAME
from vllm.v1.worker.gpu_worker import Worker


class GMSV1Worker(Worker):
    """Route vLLM allocator scopes to the selected GMS V1 backend."""

    def _get_sleep_mode_backend(self):
        # vLLM 0.30 uses a property; retained 0.29 images still use the method.
        if hasattr(Worker, "sleep_mode_backend"):
            return self.sleep_mode_backend
        return super()._get_sleep_mode_backend()

    def init_device(self) -> None:
        model_config = self.vllm_config.model_config
        if not model_config.enable_sleep_mode:
            raise RuntimeError("GMS V1 requires vLLM sleep mode")
        model_config.sleep_mode_backend = BACKEND_NAME

        super().init_device()
        self._get_sleep_mode_backend()

    def _maybe_get_memory_pool_context(self, tag: str) -> AbstractContextManager[None]:
        backend = self._get_sleep_mode_backend()
        if tag == "weights":
            return backend.capture_weights(self.model_runner.get_model)
        if tag == "kv_cache":
            return backend.capture_kv_cache()
        return super()._maybe_get_memory_pool_context(tag)

    def get_kv_recovery_state(self):
        """Report the attached physical backing and actual tensor layout."""
        from dataclasses import asdict

        from gpu_memory_service.common.persistent_pool import PersistentAllocation
        from gpu_memory_service.kv_recovery.types import KVTensorLayout
        from vllm.v1.kv_cache_interface import FullAttentionSpec

        client = self._get_sleep_mode_backend()._client
        if client._state != "RUNNING" or not client._persistent_kv:
            raise RuntimeError("KV recovery requires awake persistent GMS backing")
        specs = self.model_runner.get_kv_cache_spec()
        if not specs or any(
            not isinstance(s, FullAttentionSpec) for s in specs.values()
        ):
            raise RuntimeError("initial KV recovery supports full attention only")
        manager = client._kv_cache
        nonce, gpu_uuid = manager.identity
        mappings = manager.mappings
        allocations = [
            asdict(
                PersistentAllocation(m.allocation_id, m.aligned_size, nonce, gpu_uuid)
            )
            for m in mappings
        ]
        tensors = []
        for index, tensor in enumerate(self.model_runner.kv_caches):
            pointer = tensor.data_ptr()
            span = (
                1 + sum((d - 1) * s for d, s in zip(tensor.shape, tensor.stride()))
            ) * tensor.element_size()
            matches = [
                m
                for m in mappings
                if m.base <= pointer and pointer + span <= m.base + m.aligned_size
            ]
            if len(matches) != 1:
                raise RuntimeError("KV tensor is not contained in one GMS allocation")
            allocation = matches[0]
            tensors.append(
                asdict(
                    KVTensorLayout(
                        f"kv-{index}",
                        allocation.allocation_id,
                        pointer - allocation.base,
                        tuple(tensor.shape),
                        tuple(tensor.stride()),
                        str(tensor.dtype),
                    )
                )
            )
        return {
            "allocations": allocations,
            "tensors": tensors,
            "backing_recovered": client._kv_backing_recovered,
        }
