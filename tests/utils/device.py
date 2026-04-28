# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

try:
    import torch
except ImportError:
    torch = None


def detect_target_device() -> str:
    """Detect the runtime accelerator expected by the current test environment."""
    if torch is None:
        return "cuda"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    return "cuda"


def get_device_visibility_env_var() -> str:
    """Return the runtime-specific device visibility env var."""
    if detect_target_device() == "xpu":
        return "ZE_AFFINITY_MASK"
    return "CUDA_VISIBLE_DEVICES"


def get_default_vllm_block_size() -> int:
    """Return a runtime-compatible default vLLM block size for tests."""
    return 64 if detect_target_device() == "xpu" else 16


def build_nixl_kv_transfer_config() -> dict[str, Any]:
    """Build a runtime-compatible NIXL kv-transfer config for vLLM tests."""
    config: dict[str, Any] = {
        "kv_connector": "NixlConnector",
        "kv_role": "kv_both",
    }
    if detect_target_device() == "xpu":
        config["kv_buffer_device"] = "xpu"
    return config


def build_nixl_kv_transfer_config_json() -> str:
    """JSON-encode the runtime-compatible NIXL kv-transfer config."""
    return json.dumps(build_nixl_kv_transfer_config())


def get_gpu_memory_utilization(num_workers: int = 1, single_gpu: bool = False) -> float:
    """Get device-aware GPU memory utilization ratio for vLLM tests.

    Args:
        num_workers: Number of vLLM worker processes
        single_gpu: If True, all workers share the same GPU (requires lower utilization)

    Returns:
        GPU memory utilization ratio (0.0-1.0)

    Notes:
        - CUDA (e.g., L40S 48GB): 0.45 per worker is safe in current CI coverage
        - XPU (e.g., Intel 23GB): 0.3 is used for both shared-GPU and general cases
    """
    device = detect_target_device()

    if device == "xpu":
        # XPU general case (single worker or multi-GPU)
        return 0.4

    # CUDA (default): more generous utilization
    return 0.45
