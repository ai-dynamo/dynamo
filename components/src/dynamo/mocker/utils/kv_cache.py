#  SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
from typing import Any

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
    # AIC KVCacheQuantMode also allows int8 (1 byte per element)
    "int8": 1,
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


def get_kv_cache_dtype_bytes(config: Any, kv_cache_dtype: str = "auto") -> int:
    """Get the byte size per element for KV cache based on dtype.

    When kv_cache_dtype is "auto", uses the model's dtype from config.
    Follows vLLM's --kv-cache-dtype convention.
    """
    if kv_cache_dtype == "auto":
        dtype = _normalize_dtype_str(
            _config_get(config, "dtype", "torch_dtype") or "float16"
        )
        return TORCH_DTYPE_BYTES.get(dtype, 2)
    return TORCH_DTYPE_BYTES.get(kv_cache_dtype, 2)


def _config_get(config: Any, *names: str) -> Any:
    """Return the first non-None value among ``names`` from a dict or config object."""
    for name in names:
        value = (
            config.get(name)
            if isinstance(config, dict)
            else getattr(config, name, None)
        )
        if value is not None:
            return value
    return None


# Keys under which multimodal wrappers nest their language-model config. Mirrors
# what transformers' ``get_text_config`` unwraps for the raw config.json path.
_TEXT_CONFIG_KEYS = ("text_config", "llm_config", "language_config", "decoder")


def _text_config(config: dict[str, Any]) -> dict[str, Any]:
    if "num_hidden_layers" in config and "hidden_size" in config:
        return config
    for key in _TEXT_CONFIG_KEYS:
        sub = config.get(key)
        if isinstance(sub, dict) and "num_hidden_layers" in sub:
            return sub
    return config


def _load_config(model_path: str) -> Any:
    """Return the model's text config for ``model_path`` as a dict or config object.

    A local directory is read straight from its config.json. Importing
    ``transformers`` costs several seconds and instantiating its config classes
    imports ``torch``; together that was ~12 s of every mocker start, so the
    common path (the mocker resolves hub IDs to the local cache first) must not
    touch either. A bare hub ID still goes through transformers as before.
    """
    if os.path.isdir(model_path):
        with open(os.path.join(model_path, "config.json")) as f:
            return _text_config(json.load(f))

    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    if hasattr(config, "get_text_config"):
        config = config.get_text_config()
    return config


def compute_kv_bytes_per_token(
    model_path: str, kv_cache_dtype: str = "auto"
) -> int | None:
    """Compute KV cache bytes per token from model config.

    Formula: num_layers * 2 (K+V) * num_kv_heads * head_dim * dtype_bytes

    Reads the model's text config directly so the mocker stays independent of
    the profiler's upper AIC dependencies.

    Args:
        model_path: Path to model directory or HuggingFace model ID.
        kv_cache_dtype: KV cache dtype. "auto" uses model's torch_dtype.

    Returns:
        KV bytes per token, or None if model config cannot be parsed.
    """
    try:
        config = _load_config(model_path)
        num_layers = _config_get(config, "num_hidden_layers")
        num_attention_heads = _config_get(config, "num_attention_heads")
        num_kv_heads = _config_get(config, "num_key_value_heads", "num_kv_heads")
        if num_kv_heads is None:
            num_kv_heads = num_attention_heads
        head_dim = _config_get(config, "hidden_size") // num_attention_heads
        dtype_bytes = get_kv_cache_dtype_bytes(config, kv_cache_dtype)
        kv_bytes = num_layers * 2 * num_kv_heads * head_dim * dtype_bytes
        logger.debug(
            "Auto-computed kv_bytes_per_token=%s "
            "(%s layers, %s kv_heads, %s head_dim, %s dtype_bytes)",
            kv_bytes,
            num_layers,
            num_kv_heads,
            head_dim,
            dtype_bytes,
        )
        return kv_bytes
    except Exception as e:
        logger.warning("Could not compute kv_bytes_per_token from model config: %s", e)
        return None
