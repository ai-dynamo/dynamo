"""Candidate-only vLLM L3 sleep: release KV mappings without touching weights.

This is deliberately a narrow runtime patch for the pinned vLLM build.  It
uses its tagged CuMem allocations: weights remain mapped, while KV pages are
unmapped/released and explicitly zeroed before they are remapped on wake.
"""

import gc

import torch

from vllm.device_allocator import get_mem_allocator_instance
from vllm.device_allocator.cumem import CuMemAllocator, unmap_and_release
from vllm.device_allocator.sleep_mode_backend import CuMemBackend


def _sleep_kv_only(self: CuMemAllocator) -> None:
    total = 0
    for _ptr, data in self.pointer_to_data.items():
        if data.tag != "kv_cache":
            continue
        total += data.handle[1]
        try:
            unmap_and_release(data.handle)
        finally:
            data.is_asleep = True
    gc.collect()
    torch.cuda.empty_cache()
    if total <= 0:
        raise RuntimeError("Ghost KV L3 found no tagged KV allocation")


CuMemAllocator.sleep_kv_only = _sleep_kv_only
_original_wake_up = CuMemAllocator.wake_up


def _wake_up_asleep_only(self: CuMemAllocator, tags=None) -> None:
    """Do not recreate live weight mappings during KV-only wake."""
    if not any(data.is_asleep for data in self.pointer_to_data.values()):
        return _original_wake_up(self, tags)
    gc.collect()
    torch.accelerator.empty_cache()
    for ptr, data in self.pointer_to_data.items():
        if not data.is_asleep or (tags is not None and data.tag not in tags):
            continue
        from vllm.device_allocator.cumem import create_and_map, libcudart

        create_and_map(data.handle)
        # cuMemCreate does not give this security boundary an explicit zeroing
        # contract.  Do not rely on driver allocation hygiene: a re-created KV
        # mapping must not expose the previous tenant's cache contents.
        libcudart.cudaMemset(ptr, 0, data.handle[1])
        data.is_asleep = False
        if data.cpu_backup_tensor is not None:
            backup = data.cpu_backup_tensor
            libcudart.cudaMemcpy(ptr, backup.data_ptr(), backup.numel())
            data.cpu_backup_tensor = None


CuMemAllocator.wake_up = _wake_up_asleep_only
_original_suspend = CuMemBackend.suspend


def _suspend(self: CuMemBackend, level: int = 1) -> None:
    if level != 3:
        return _original_suspend(self, level)
    self._state = "SUSPENDED"
    allocator = get_mem_allocator_instance()
    if not isinstance(allocator, CuMemAllocator):
        raise RuntimeError("Ghost KV L3 requires CuMemAllocator")
    allocator.sleep_kv_only()


CuMemBackend.suspend = _suspend
