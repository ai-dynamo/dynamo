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


# Wrapper keys under which multimodal and multi-module configs nest the language
# model's config, e.g. ``text_config`` (most VLMs) or ``thinker_config.text_config``
# (Qwen2.5-Omni). Transformers' ``get_text_config`` picks the sub-config the same
# way plus per-model overrides; a layout not found here falls back to transformers.
_TEXT_CONFIG_KEYS = (
    "text_config",
    "llm_config",
    "language_config",
    "decoder",
    "thinker_config",
)
# A raw config.json is used only when these appear verbatim. Legacy aliases such
# as GPT-2's ``n_layer``/``n_embd`` are left to transformers' ``attribute_map``.
_REQUIRED_KEYS = ("num_hidden_layers", "num_attention_heads", "hidden_size")


def _find_text_config(config: dict[str, Any], depth: int = 3) -> dict[str, Any] | None:
    """Return the first dict carrying the required sizes, searching known wrappers."""
    if all(isinstance(config.get(key), int) for key in _REQUIRED_KEYS):
        return config
    if depth == 0:
        return None
    for key in _TEXT_CONFIG_KEYS:
        sub = config.get(key)
        if isinstance(sub, dict):
            found = _find_text_config(sub, depth - 1)
            if found is not None:
                return found
    return None


def _load_config(model_path: str) -> Any:
    """Return the model's text config for ``model_path`` as a dict or config object.

    A local directory whose config.json has the canonical layout is read directly,
    so that path imports neither ``transformers`` nor ``torch``. Any other layout,
    and a bare hub ID, go through transformers as before.
    """
    if os.path.isdir(model_path):
        with open(os.path.join(model_path, "config.json")) as f:
            text_config = _find_text_config(json.load(f))
        if text_config is not None:
            return text_config
        logger.info(
            "config.json layout not recognized, resolving %s through transformers",
            model_path,
        )

    # Imported here on purpose: transformers pulls in torch, which dominates
    # mocker startup, and the common path above does not need it.
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
    except (OSError, ValueError, KeyError) as e:
        # No or unreadable config.json, invalid JSON, or a model type that
        # transformers does not recognize. Anything else is a bug: let it raise.
        logger.warning("Could not compute kv_bytes_per_token from model config: %s", e)
        return None

    num_layers = _config_get(config, "num_hidden_layers")
    num_attention_heads = _config_get(config, "num_attention_heads")
    hidden_size = _config_get(config, "hidden_size")
    sizes = (num_layers, num_attention_heads, hidden_size)
    if not all(isinstance(v, int) for v in sizes) or num_attention_heads == 0:
        logger.warning(
            "Could not compute kv_bytes_per_token: model config for %s lacks "
            "layer, head, or hidden sizes (%s)",
            model_path,
            sizes,
        )
        return None

    num_kv_heads = _config_get(config, "num_key_value_heads", "num_kv_heads")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    head_dim = hidden_size // num_attention_heads
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
