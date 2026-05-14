# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from vllm.config import VllmConfig
from vllm.v1.engine.async_llm import AsyncLLM

logger = logging.getLogger(__name__)

DYNAMO_KV_EVENT_BLOCK_SIZE_KEY = "dynamo_kv_event_block_size"
MAIN_ATTENTION_KV_CACHE_KINDS = {
    "full_attention",
    "mla_attention",
}

# Known Mamba architecture identifiers present in vLLM's HF config.
_MAMBA_ARCHITECTURES = {
    "MambaLMHeadModel",
    "MambaDecodeModel",
    "MambaForCausalLM",
    "FalconMambaForCausalLM",
    "Mamba2ForCausalLM",
    "JambaForCausalLM",
}


def detect_mamba_hybrid_model(vllm_config: VllmConfig) -> bool:
    """Return True if the vLLM model is a Mamba or hybrid (speculative decode) model.

    Mamba models (state-space models) use a different KV cache block size
    than attention-based models. When ``get_kv_cache_group_metadata`` is
    unavailable, we must fail with a clear error rather than silently using
    the wrong block size.
    """
    # Speculative config is populated for hybrid decode (e.g. EAGLE, Medusa).
    if vllm_config.speculative_config is not None:
        return True

    # Inspect the model architecture via HuggingFace config.
    hf_config = getattr(vllm_config.model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None) or []
    for arch in architectures:
        if arch in _MAMBA_ARCHITECTURES:
            return True

    return False


def get_configured_kv_event_block_size(vllm_config: VllmConfig) -> int:
    """Return the configured KV event block size, falling back to vLLM's cache block size."""
    additional_config = vllm_config.additional_config or {}
    return additional_config.get(
        DYNAMO_KV_EVENT_BLOCK_SIZE_KEY,
        vllm_config.cache_config.block_size,
    )


def select_main_attention_block_size(
    group_metadata: list[dict[str, Any]],
    fallback_block_size: int,
) -> int:
    """Select the main-attention KV block size from engine cache-group metadata."""
    if not group_metadata:
        return fallback_block_size

    for group in group_metadata:
        if group.get("kind") in MAIN_ATTENTION_KV_CACHE_KINDS:
            return group.get("block_size", fallback_block_size)

    return fallback_block_size


async def configure_kv_event_block_size(
    engine: AsyncLLM,
    vllm_config: VllmConfig,
) -> int:
    """Fetch engine cache-group metadata and cache the KV event block size on vLLM config.

    Raises
    ------
    ValueError
        When the ``get_kv_cache_group_metadata`` utility call fails and the
        model is Mamba or a hybrid (speculative decode) model, because such
        models require the utility to discover the correct block size.
    """
    fallback_block_size = vllm_config.cache_config.block_size
    is_mamba_or_hybrid = detect_mamba_hybrid_model(vllm_config)
    try:
        group_metadata = await engine.engine_core.call_utility_async(
            "get_kv_cache_group_metadata"
        )
    except Exception as e:
        if is_mamba_or_hybrid:
            hf_config = getattr(vllm_config.model_config, "hf_config", None)
            architectures = list(getattr(hf_config, "architectures", []) or [])
            speculative = vllm_config.speculative_config is not None
            reason = "speculative/hybrid model" if speculative else "Mamba model"
            raise ValueError(
                f"Failed to fetch KV cache group metadata for {reason} "
                f"(architectures={architectures}). "
                f"The get_kv_cache_group_metadata engine utility must be available "
                f"to determine the correct KV event block size. Original error: {e}"
            ) from e
        logger.warning(
            "Failed to fetch KV cache group metadata; falling back to "
            "vLLM cache_config.block_size: %s",
            e,
        )
        kv_event_block_size = fallback_block_size
    else:
        kv_event_block_size = select_main_attention_block_size(
            group_metadata,
            fallback_block_size,
        )

    if vllm_config.additional_config is None:
        vllm_config.additional_config = {}
    vllm_config.additional_config[DYNAMO_KV_EVENT_BLOCK_SIZE_KEY] = kv_event_block_size
    return kv_event_block_size
