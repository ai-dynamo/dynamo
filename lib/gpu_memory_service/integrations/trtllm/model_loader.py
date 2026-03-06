# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT-LLM model loader patches for GPU Memory Service integration."""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

import torch
from gpu_memory_service import get_or_create_gms_client_memory_manager
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.common.types import GrantedLockType, RequestedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import finalize_gms_write

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

_model_loader_patched = False
_gms_enabled = False
_gms_lock_mode = RequestedLockType.RW_OR_RO
_last_imported_weights_bytes = 0


def _ptr_in_gms_mappings(gms_client: "GMSClientMemoryManager", ptr: int) -> bool:
    for va, mapping in gms_client.mappings.items():
        if va <= ptr < va + mapping.aligned_size:
            return True
    return False


def _move_untracked_parameters_to_pool(
    model: torch.nn.Module,
    gms_client: "GMSClientMemoryManager",
    target_device: torch.device,
) -> list[str]:
    """Move CUDA parameters not already mapped by GMS into the active mempool."""
    from gpu_memory_service.client.torch.module import _iter_module_tensors

    moved: list[str] = []
    seen_parameter_objects: set[int] = set()
    device_index = (
        torch.cuda.current_device()
        if target_device.index is None
        else int(target_device.index)
    )
    with torch.no_grad():
        for name, tensor, tensor_type in _iter_module_tensors(model):
            if tensor_type != "parameter":
                continue
            tensor_obj_id = id(tensor)
            if tensor_obj_id in seen_parameter_objects:
                continue
            seen_parameter_objects.add(tensor_obj_id)

            if tensor is None or not tensor.is_cuda:
                continue
            if _ptr_in_gms_mappings(gms_client, int(tensor.data_ptr())):
                continue

            replacement = torch.empty_like(tensor, device=target_device)
            replacement.copy_(tensor)
            if not _ptr_in_gms_mappings(gms_client, int(replacement.data_ptr())):
                replacement = _copy_tensor_to_gms_mapping(
                    tensor=tensor,
                    gms_client=gms_client,
                    device_index=device_index,
                )
            tensor.data = replacement
            moved.append(name)
    return moved


def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
    element_size = int(tensor.element_size())
    shape = list(tensor.shape)
    stride = list(tensor.stride())
    if shape and stride:
        max_offset = sum(
            int(s) * (int(d) - 1)
            for s, d in zip(stride, shape, strict=True)
            if int(d) > 0
        )
        return int((max_offset + 1) * element_size)
    return element_size


def _copy_tensor_to_gms_mapping(
    tensor: torch.Tensor,
    gms_client: "GMSClientMemoryManager",
    device_index: int,
) -> torch.Tensor:
    from gpu_memory_service.client.torch.tensor import _tensor_from_pointer

    base_va = gms_client.create_mapping(
        size=_tensor_storage_nbytes(tensor),
        tag="weights",
    )
    mapped_tensor = _tensor_from_pointer(
        int(base_va),
        list(tensor.shape),
        list(tensor.stride()),
        tensor.dtype,
        device_index,
    )
    mapped_tensor.copy_(tensor)
    return mapped_tensor


def set_gms_enabled(enabled: bool) -> None:
    """Enable or disable GMS mode for TensorRT-LLM patches."""
    global _gms_enabled
    _gms_enabled = enabled


def set_gms_lock_mode(mode: RequestedLockType) -> None:
    """Set lock mode used by the TensorRT-LLM GMS loader."""
    global _gms_lock_mode
    _gms_lock_mode = mode


def get_gms_lock_mode() -> RequestedLockType:
    """Get lock mode used by the TensorRT-LLM GMS loader."""
    return _gms_lock_mode


def get_imported_weights_bytes() -> int:
    """Return bytes of weights imported/published by last model load."""
    return int(_last_imported_weights_bytes)


def patch_model_loader() -> None:
    """Patch TensorRT-LLM's model loader to support GMS weights."""
    global _model_loader_patched
    if _model_loader_patched:
        return

    import tensorrt_llm._torch.pyexecutor.model_loader as trt_model_loader_module

    original_load = trt_model_loader_module.ModelLoader.load
    original_get_rank_model_storage = trt_model_loader_module.get_rank_model_storage

    def patched_get_rank_model_storage(model) -> int:
        imported_bytes = get_imported_weights_bytes()
        if imported_bytes > 0:
            return imported_bytes
        return int(original_get_rank_model_storage(model))

    def patched_load(self, checkpoint_dir: str, checkpoint_loader):
        if not _gms_enabled:
            return original_load(self, checkpoint_dir, checkpoint_loader)
        return _load_model_with_gms(
            self=self,
            checkpoint_dir=checkpoint_dir,
            checkpoint_loader=checkpoint_loader,
            original_load=original_load,
        )

    trt_model_loader_module.get_rank_model_storage = patched_get_rank_model_storage
    trt_model_loader_module.ModelLoader.load = patched_load
    _model_loader_patched = True
    logger.info("[GMS] Patched TensorRT-LLM ModelLoader.load")


def _load_model_with_gms(self, checkpoint_dir: str, checkpoint_loader, original_load):
    """Load model with GMS-backed weights for TensorRT-LLM."""
    device_index = torch.cuda.current_device()
    gms_client, pool = get_or_create_gms_client_memory_manager(
        get_socket_path(device_index),
        device_index,
        mode=_gms_lock_mode,
        tag="weights",
    )

    if gms_client.granted_lock_type == GrantedLockType.RO:
        return _load_read_mode(
            self=self,
            checkpoint_dir=checkpoint_dir,
            checkpoint_loader=checkpoint_loader,
            gms_client=gms_client,
            device_index=device_index,
        )

    if pool is None:
        raise RuntimeError("GMS pool is None in write mode")

    gms_client.clear_all_handles()

    target_device = torch.device("cuda", device_index)
    global _last_imported_weights_bytes
    with torch.cuda.use_mem_pool(pool, device=target_device):
        model, moe_load_balancer = original_load(self, checkpoint_dir, checkpoint_loader)
        moved_params = _move_untracked_parameters_to_pool(
            model=model,
            gms_client=gms_client,
            target_device=target_device,
        )
        if moved_params:
            logger.warning(
                "[GMS] TensorRT-LLM moved %d parameter(s) into GMS pool: %s",
                len(moved_params),
                moved_params[:8],
            )
        torch.cuda.current_stream().synchronize()
        _last_imported_weights_bytes = finalize_gms_write(gms_client, model)
    logger.info(
        "[GMS] TensorRT-LLM write mode: published %.2f GiB",
        _last_imported_weights_bytes / (1 << 30),
    )

    return model, moe_load_balancer


def _load_read_mode(
    self,
    checkpoint_dir: str,
    checkpoint_loader,
    gms_client: "GMSClientMemoryManager",
    device_index: int,
):
    """Load model by importing weights from GMS (RO mode)."""
    global _last_imported_weights_bytes

    from tensorrt_llm._torch.models import AutoModelForCausalLM
    from tensorrt_llm._torch.models.modeling_utils import MetaInitMode, timing
    from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
        MoeLoadBalancer,
        maybe_create_moe_load_balancer,
    )

    config = self._load_and_validate_config(checkpoint_dir, checkpoint_loader)

    with timing("Model init total"), maybe_create_moe_load_balancer(
        config, self.mapping
    ) as moe_load_balancer:
        config_copy = copy.deepcopy(config)
        try:
            with MetaInitMode():
                model = AutoModelForCausalLM.from_config(config_copy)
        except Exception as exc:
            raise RuntimeError(
                "TensorRT-LLM GMS read mode requires successful MetaInitMode model construction"
            ) from exc

        materialize_module_from_gms(gms_client, model, device_index=device_index)
        _last_imported_weights_bytes = int(gms_client.total_bytes)
        logger.info(
            "[GMS] TensorRT-LLM read mode: imported %.2f GiB",
            _last_imported_weights_bytes / (1 << 30),
        )

        for module in model.modules():
            if hasattr(module, "post_load_weights") and not getattr(
                module, "_weights_removed", False
            ):
                module.post_load_weights()

        if isinstance(moe_load_balancer, MoeLoadBalancer):
            moe_load_balancer.register_weight_slots_after_to_cuda()
            logger.info("[GMS] moe_load_balancer finalizing model...")
            moe_load_balancer.finalize_model()
            logger.info("[GMS] moe_load_balancer finalize model done")

        torch.cuda.current_stream().synchronize()

    return model, moe_load_balancer
