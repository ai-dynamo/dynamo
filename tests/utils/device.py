# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
from typing import Any

try:
    import torch
except ImportError:
    torch = None

logger = logging.getLogger(__name__)


def detect_target_device() -> str:
    """Detect the runtime accelerator expected by the current test environment."""
    if torch is None:
        logger.info("torch not available, defaulting to CUDA")
        return "cuda"

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"

    logger.info("No accelerator detected, defaulting to CUDA")
    return "cuda"


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
