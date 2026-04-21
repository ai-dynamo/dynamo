# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shadow mode utilities for GMS TensorRT-LLM integration."""

import logging
import os

logger = logging.getLogger(__name__)

# Minimum KV cache footprint for shadow engine init. The shadow can't share an
# active engine's KV budget at init time, so we shrink the shadow's KV pool
# until it is quiesced. After failover the shadow serves at this reduced
# capacity; TRT-LLM has no live-resize API to grow it back.
SHADOW_KV_CACHE_MAX_BYTES = 2 * 1024 * 1024 * 1024  # 2 GiB per GPU
SHADOW_KV_CACHE_FRACTION = 0.01


def is_shadow_mode() -> bool:
    """True when DYN_GMS_SHADOW_MODE=1 (set by llm_worker.py at startup)."""
    return os.environ.get("DYN_GMS_SHADOW_MODE", "0") == "1"


def is_shadow_standby() -> bool:
    """True for engines that should skip full KV init (shadow, ENGINE_ID != 0)."""
    return is_shadow_mode() and os.environ.get("ENGINE_ID", "0") != "0"


def shrink_kv_cache_for_shadow(kv_cache_config) -> None:
    """Clamp a ``KvCacheConfig`` so shadow init fits alongside an active engine.

    The shadow engine initializes while engine-0 is holding the primary KV
    pool (~60-75 GiB/GPU on Kimi-scale models). With the default
    ``free_gpu_memory_fraction``, TRT-LLM's KV estimation run tries to
    claim another ~60 GiB and OOMs. Clamping both fraction *and*
    ``max_gpu_total_bytes`` keeps the shadow's pool small enough to fit
    in the slack, and the values are kept at the same settings after
    ``materialize_with_tag('kv_cache')`` on wake.
    """
    kv_cache_config.free_gpu_memory_fraction = SHADOW_KV_CACHE_FRACTION
    # Cap hard so the fraction-based estimation can never exceed this.
    kv_cache_config.max_gpu_total_bytes = SHADOW_KV_CACHE_MAX_BYTES
    logger.info(
        "[Shadow] Clamped KV cache to %.2f GiB (fraction=%s) for shadow init",
        SHADOW_KV_CACHE_MAX_BYTES / (1 << 30),
        SHADOW_KV_CACHE_FRACTION,
    )


def configure_gms_lock_mode(extra_config: dict) -> dict:
    """Set gms_read_only in model_loader_extra_config based on ENGINE_ID.

    Shadow-engine failover ensures that only ENGINE_ID="0" loads weights
    from disk (RW_OR_RO). All other engines import from the committed GMS
    layout (RO). This avoids deadlock when multiple engines contend for RW
    locks across TP ranks.

    Raises if user-specified gms_read_only conflicts with ENGINE_ID.
    Mutates and returns the same dict so callers can chain.
    """
    engine_id = os.environ.get("ENGINE_ID", "0")
    user_read_only = extra_config.get("gms_read_only", None)

    if engine_id == "0":
        if user_read_only:
            raise ValueError(
                "ENGINE_ID=0 is the primary writer but "
                "gms_read_only=True was explicitly set. "
                "The primary engine must be able to write weights."
            )
    else:
        if user_read_only is not None and not user_read_only:
            raise ValueError(
                f"ENGINE_ID={engine_id} requires gms_read_only=True, "
                f"but gms_read_only=False was explicitly set."
            )
        extra_config["gms_read_only"] = True

    logger.info(
        "[Shadow] ENGINE_ID=%s → gms_read_only=%s",
        engine_id,
        extra_config.get("gms_read_only", False),
    )
    return extra_config
