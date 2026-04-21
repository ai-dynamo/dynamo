# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM model loader patches for GPU Memory Service integration.

This module patches TensorRT-LLM's ModelLoader to load weights through GMS,
enabling VA-stable weight sharing and sleep/wake with memory release.

Two modes:
  - RW (first loader): loads weights from disk, allocates via GMS pool, commits.
  - RO (subsequent loaders): materializes model tensors from the committed layout.
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.common.locks import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import finalize_gms_write

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

_model_loader_patched = False
_gms_enabled = False
_gms_lock_mode = RequestedLockType.RW_OR_RO
_last_imported_weights_bytes: int = 0


def set_gms_enabled(enabled: bool) -> None:
    global _gms_enabled
    _gms_enabled = enabled


def set_gms_lock_mode(mode: RequestedLockType) -> None:
    global _gms_lock_mode
    _gms_lock_mode = mode


def get_gms_lock_mode() -> RequestedLockType:
    return _gms_lock_mode


def get_imported_weights_bytes() -> int:
    """Return total bytes of weights imported/published by the last model load."""
    return _last_imported_weights_bytes


def patch_model_loader() -> None:
    """Patch TensorRT-LLM's ModelLoader to load weights through GMS.

    Idempotent — safe to call multiple times.
    """
    global _model_loader_patched
    if _model_loader_patched:
        return

    import tensorrt_llm._torch.pyexecutor.model_loader as _trt_loader

    _original_load = _trt_loader.ModelLoader.load
    _original_get_rank_model_storage = _trt_loader.get_rank_model_storage

    def patched_get_rank_model_storage(model) -> int:
        imported = get_imported_weights_bytes()
        if imported > 0:
            return imported
        return int(_original_get_rank_model_storage(model))

    def patched_load(self, checkpoint_dir: str, checkpoint_loader):
        if not _gms_enabled:
            return _original_load(self, checkpoint_dir, checkpoint_loader)
        return _gms_load(
            self=self,
            checkpoint_dir=checkpoint_dir,
            checkpoint_loader=checkpoint_loader,
            original_load=_original_load,
        )

    _trt_loader.get_rank_model_storage = patched_get_rank_model_storage
    _trt_loader.ModelLoader.load = patched_load
    _model_loader_patched = True
    logger.info("[GMS] Patched TensorRT-LLM ModelLoader.load")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _gms_load(self, checkpoint_dir: str, checkpoint_loader, original_load):
    """Route to RW (write) or RO (read) load path based on granted lock type."""
    # Neutralize TRT-LLM's model_weights_memory_tag to prevent its VMM scope
    # from overriding the GMS allocator. When sleep_config is set, TRT-LLM
    # wraps allocation in virtual_memory_scope(model_weights) — a nested scope
    # that would steal allocations away from GMS's gms_use_mem_pool.
    saved_tag = getattr(self, "model_weights_memory_tag", None)
    self.model_weights_memory_tag = None

    device_index = torch.cuda.current_device()
    gms_client = get_or_create_gms_client_memory_manager(
        get_socket_path(device_index, "weights"),
        device_index,
        mode=_gms_lock_mode,
        tag="weights",
    )

    try:
        return _gms_load_inner(
            self,
            gms_client,
            device_index,
            checkpoint_dir,
            checkpoint_loader,
            original_load,
        )
    finally:
        self.model_weights_memory_tag = saved_tag


def _gms_load_inner(
    self, gms_client, device_index, checkpoint_dir, checkpoint_loader, original_load
):
    if gms_client.granted_lock_type == GrantedLockType.RO:
        return _load_ro(
            self=self,
            checkpoint_dir=checkpoint_dir,
            checkpoint_loader=checkpoint_loader,
            gms_client=gms_client,
            device_index=device_index,
        )

    return _load_rw(
        self=self,
        checkpoint_dir=checkpoint_dir,
        checkpoint_loader=checkpoint_loader,
        gms_client=gms_client,
        device_index=device_index,
        original_load=original_load,
    )


def _load_rw(
    self, checkpoint_dir, checkpoint_loader, gms_client, device_index, original_load
):
    """RW path: load weights from disk into the GMS memory pool, then commit."""
    global _last_imported_weights_bytes

    target_device = torch.device("cuda", device_index)

    with gms_use_mem_pool("weights", target_device):
        model, moe_load_balancer = original_load(
            self, checkpoint_dir, checkpoint_loader
        )
        _move_untracked_params(model, gms_client, target_device)
        torch.cuda.empty_cache()

    _last_imported_weights_bytes = finalize_gms_write(gms_client, model)

    logger.info(
        "[GMS] TRT-LLM RW: published %.2f GiB",
        _last_imported_weights_bytes / (1 << 30),
    )
    return model, moe_load_balancer


def _load_ro(self, checkpoint_dir, checkpoint_loader, gms_client, device_index):
    """RO path: skip disk I/O, materialize tensors from the committed GMS layout."""
    global _last_imported_weights_bytes

    from tensorrt_llm._torch.models import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode, timing
    from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
        MoeLoadBalancer,
        maybe_create_moe_load_balancer,
    )

    config = self._load_and_validate_config(checkpoint_dir, checkpoint_loader)

    with (
        timing("Model init total"),
        maybe_create_moe_load_balancer(config, self.mapping) as moe_load_balancer,
    ):
        try:
            with MetaInitMode():
                model = AutoModelForCausalLM.from_config(copy.deepcopy(config))
        except Exception as exc:
            raise RuntimeError(
                "GMS RO path requires successful MetaInitMode model construction"
            ) from exc

        # Some models register cross-layer references like next_attn here.
        if hasattr(model, "post_load_weights"):
            model.post_load_weights()

        materialize_module_from_gms(gms_client, model, device_index=device_index)
        _last_imported_weights_bytes = int(gms_client.total_bytes)

        logger.info(
            "[GMS] TRT-LLM RO: imported %.2f GiB",
            _last_imported_weights_bytes / (1 << 30),
        )

        for module in model.modules():
            if hasattr(module, "post_load_weights") and not getattr(
                module, "_weights_removed", False
            ):
                module.post_load_weights()

        if isinstance(moe_load_balancer, MoeLoadBalancer):
            moe_load_balancer.register_weight_slots_after_to_cuda()
            moe_load_balancer.finalize_model()

        torch.cuda.current_stream().synchronize()

    return model, moe_load_balancer


def _ptr_in_gms(gms_client: "GMSClientMemoryManager", ptr: int) -> bool:
    """Return True if the given CUDA VA is within any active GMS mapping."""
    for va, mapping in gms_client.mappings.items():
        if va <= ptr < va + mapping.aligned_size:
            return True
    return False


def _move_untracked_params(
    model: torch.nn.Module,
    gms_client: "GMSClientMemoryManager",
    target_device: torch.device,
) -> None:
    """Move CUDA parameters that were allocated outside the GMS pool into it.

    TRT-LLM may allocate some parameters outside the pluggable-allocator scope.
    This ensures all weight tensors end up tracked by GMS before we commit.
    """
    from gpu_memory_service.client.torch.module import _iter_module_tensors
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer

    device_index = (
        torch.cuda.current_device()
        if target_device.index is None
        else int(target_device.index)
    )
    seen: set[int] = set()

    with torch.no_grad():
        for _name, tensor, tensor_type in _iter_module_tensors(model):
            if tensor_type != "parameter" or tensor is None or not tensor.is_cuda:
                continue
            storage_ptr = tensor.storage().data_ptr()
            if storage_ptr in seen:
                continue
            seen.add(storage_ptr)

            if _ptr_in_gms(gms_client, int(tensor.data_ptr())):
                continue

            # Allocate a new mapping and copy the tensor into it
            nbytes = _storage_nbytes(tensor)
            base_va = gms_client.create_mapping(size=nbytes, tag="weights")
            replacement = _tensor_from_pointer(
                int(base_va),
                list(tensor.shape),
                list(tensor.stride()),
                tensor.dtype,
                device_index,
            )
            replacement.copy_(tensor)
            tensor.data = replacement


def _storage_nbytes(tensor: torch.Tensor) -> int:
    if tensor.numel() == 0:
        return 0
    element_size = int(tensor.element_size())
    shape = list(tensor.shape)
    stride = list(tensor.stride())
    if not shape:
        return element_size
    max_offset = sum(
        abs(int(s)) * (int(d) - 1)
        for s, d in zip(stride, shape, strict=True)
        if int(d) > 1
    )
    return int((max_offset + 1) * element_size)
