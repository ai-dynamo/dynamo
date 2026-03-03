#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import logging

from transformers import AutoConfig

from dynamo.profiler.utils.model_info import get_model_info

logger = logging.getLogger(__name__)

# Mapping from dtype strings to byte sizes for KV cache.
# Used when --kv-cache-dtype is "auto" to infer from model config's dtype,
# or when explicitly set via CLI (matching vLLM's --kv-cache-dtype choices).
TORCH_DTYPE_BYTES = {
    # auto-detected from model config (torch.dtype str representations)
    "float16": 2,
    "bfloat16": 2,
    "float32": 4,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    # vLLM CLI choices
    "fp8": 1,
    "fp8_ds_mla": 1,
    "fp8_e4m3": 1,
    "fp8_inc": 1,
}

# Default KV transfer bandwidth in GB/s.
# 64 GB/s corresponds to inter-node InfiniBand.
# For intra-node NVLink, typical value is ~450 GB/s.
DEFAULT_KV_TRANSFER_BANDWIDTH_GBPS = 64.0


def _normalize_dtype_str(dtype) -> str:
    """Normalize a dtype to a plain string like 'float16'.

    Handles torch.dtype objects (str() gives 'torch.float16') and plain strings.
    """
    s = str(dtype)
    if s.startswith("torch."):
        s = s[len("torch.") :]
    return s


def get_kv_cache_dtype_bytes(config, kv_cache_dtype: str = "auto") -> int:
    """Get the byte size per element for KV cache based on dtype.

    When kv_cache_dtype is "auto", uses the model's dtype from config.
    Follows vLLM's --kv-cache-dtype convention.
    """
    if kv_cache_dtype == "auto":
        dtype = _normalize_dtype_str(getattr(config, "dtype", "float16"))
        return TORCH_DTYPE_BYTES.get(dtype, 2)
    return TORCH_DTYPE_BYTES.get(kv_cache_dtype, 2)


def compute_kv_bytes_per_token(
    model_path: str, kv_cache_dtype: str = "auto"
) -> int | None:
    """Compute KV cache bytes per token from model config.

    Formula: num_layers * 2 (K+V) * num_kv_heads * head_dim * dtype_bytes

    Uses get_model_info from dynamo.profiler for robust detection of num_kv_heads
    across different model architectures.

    Args:
        model_path: Path to model directory or HuggingFace model ID.
        kv_cache_dtype: KV cache dtype. "auto" uses model's torch_dtype.

    Returns:
        KV bytes per token, or None if model config cannot be parsed.
    """
    try:
        info = get_model_info(model_path)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
        num_layers = config.num_hidden_layers
        num_kv_heads = info.num_kv_heads
        head_dim = config.hidden_size // config.num_attention_heads
        dtype_bytes = get_kv_cache_dtype_bytes(config, kv_cache_dtype)
        kv_bytes = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
        logger.debug(
            f"Auto-computed kv_bytes_per_token={kv_bytes} "
            f"({num_layers} layers, {num_kv_heads} kv_heads, {head_dim} head_dim, "
            f"{dtype_bytes} dtype_bytes)"
        )
        return kv_bytes
    except Exception as e:
        logger.warning(f"Could not compute kv_bytes_per_token from model config: {e}")
        return None
