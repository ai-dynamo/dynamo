"""Shared utilities for GPU Memory Service vLLM integration."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional, Type

if TYPE_CHECKING:
    from dynamo.gpu_memory_service import GMSClientMemoryManager

logger = logging.getLogger(__name__)


def get_vllm_worker_class() -> Optional[Type]:
    """Get vLLM Worker class with fallback to GPUWorker.

    vLLM 0.12+ renamed GPUWorker to Worker. This function handles both versions.

    Returns:
        Worker class if found, None otherwise.
    """
    # Try vLLM 0.12+ naming
    try:
        from vllm.v1.worker.gpu_worker import Worker
        return Worker
    except ImportError:
        pass

    # Try older naming
    try:
        from vllm.v1.worker.gpu_worker import GPUWorker
        return GPUWorker
    except ImportError:
        pass

    return None


def get_gms_memory_manager() -> Optional["GMSClientMemoryManager"]:
    """Get the GMS client memory manager singleton.

    Returns:
        GMSClientMemoryManager instance if registered, None otherwise.
    """
    from dynamo.gpu_memory_service import get_gms_client_memory_manager
    return get_gms_client_memory_manager()


def has_vmm_allocations() -> bool:
    """Check if there are active VMM allocations from GPU Memory Service.

    Returns:
        True if there are active VMM allocations that would be destroyed by
        torch.cuda.empty_cache().
    """
    manager = get_gms_memory_manager()
    if manager is None:
        return False
    return len(manager._mappings) > 0
