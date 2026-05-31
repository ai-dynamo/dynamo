# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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


def get_default_vllm_block_size() -> int:
    """Return a runtime-compatible default vLLM block size for tests."""
    return 64 if detect_target_device() == "xpu" else 16
