# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM configuration extraction for Dynamo scheduler connector.

This module provides utilities to extract configuration parameters from vLLM's
internal configuration objects and transform them into a format compatible with
the Dynamo scheduler connector.
"""

import logging
from typing import Any, Dict

from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import get_kv_cache_layout
from vllm.version import __version_tuple__

from . import KvbmVllmConfig

if __version_tuple__ >= (0, 12, 2):
   raise ImportError("vLLM versions after 0.12.2 are not yet supported")

if __version_tuple__ <= (0, 11, 0):
    raise ImportError("vLLM versions before 0.11 are not supported")


logger = logging.getLogger(__name__)


def format_config_table(config: Dict[str, Dict[str, Any]]) -> str:
    """Format configuration dictionary as a nice ASCII table.

    Args:
        config: Configuration dictionary with 'parallel' and 'attention' sections

    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("Dynamo SchedulerWorker Configuration")
    lines.append("=" * 70)

    # Parallel Configuration
    lines.append("")
    lines.append("Parallel Configuration:")
    lines.append("-" * 70)
    for key, value in config["parallel"].items():
        lines.append(f"  {key:30s} : {value}")

    # Attention/Cache Configuration
    lines.append("")
    lines.append("Attention/Cache Configuration:")
    lines.append("-" * 70)
    for key, value in config["attention"].items():
        lines.append(f"  {key:30s} : {value}")

    lines.append("=" * 70)
    return "\n".join(lines)


def extract_vllm_config_for_kvbm(
    vllm_config: VllmConfig,
    log_config: bool = True,
) -> KvbmVllmConfig:
    """Extract vLLM configuration and create KvbmVllmConfig for Dynamo scheduler connector.

    This function extracts relevant configuration parameters from vLLM's configuration
    objects and creates a KvbmVllmConfig object that can be passed to Rust constructors.

    Args:
        vllm_config: vLLM's VllmConfig object containing parallelism and KV cache configuration
        log_config: If True, log the configuration table (only on rank 0)

    Returns:
        KvbmVllmConfig object containing the extracted configuration parameters.

    Example:
        >>> from kvbm.contrib.vllm.config import extract_vllm_config_for_kvbm
        >>> config = extract_vllm_config_for_kvbm(vllm_config)
        >>> # config is a KvbmVllmConfig object ready to use with Rust constructors
    """
    cfg = vllm_config

    # Extract parallel configuration
    parallel_dict = {
        "world_size": cfg.parallel_config.world_size,
        "rank": cfg.parallel_config.rank,
        "tensor_parallel_size": cfg.parallel_config.tensor_parallel_size,
        "pipeline_parallel_size": cfg.parallel_config.pipeline_parallel_size,
        "data_parallel_size": cfg.parallel_config.data_parallel_size,
        "data_parallel_rank": cfg.parallel_config.data_parallel_rank,
        # (type) DistributedExecutorBackend = Literal['ray', 'mp', 'uni', 'external_launcher']
        "backend": cfg.parallel_config.data_parallel_backend,
    }

    # Extract cache/attention configuration
    # num_gpu_blocks and num_cpu_blocks may be None at connector construction time
    # They get set later during KV cache allocation
    # The actual block count can be derived from tensor shape in register_kv_caches
    attention_dict = {
        "block_size": cfg.cache_config.block_size,
        "num_gpu_blocks": cfg.cache_config.num_gpu_blocks or 0,
        "num_cpu_blocks": cfg.cache_config.num_cpu_blocks or 0,
        "cache_dtype_bytes": _get_cache_dtype_bytes(cfg),
        "kv_cache_layout": get_kv_cache_layout(),
        "head_size": cfg.model_config.get_head_size(),
        "num_heads": cfg.model_config.get_total_num_kv_heads(),
    }

    # Log configuration table on rank 0 only
    if log_config and parallel_dict.get("rank", 0) == 0:
        config = {
            "parallel": parallel_dict,
            "attention": attention_dict,
        }
        config_table = format_config_table(config)
        print("\n" + config_table + "\n")

    logger.debug(
        f"Extracted vLLM config for Dynamo - parallel: {parallel_dict}, attention: {attention_dict}"
    )

    # Create and return KvbmVllmConfig object
    return KvbmVllmConfig(parallel_dict, attention_dict)


def _get_cache_dtype_bytes(vllm_config: VllmConfig) -> int:
    """Get the size in bytes of the cache dtype.

    Args:
        cache_config: vLLM's CacheConfig object

    Returns:
        Size in bytes of the cache dtype
    """
    cache_dtype = vllm_config.cache_config.cache_dtype
    model_dtype = vllm_config.model_config.dtype

    # Convert dtype to string for comparison
    dtype_str = str(cache_dtype).lower()
    model_dtype_str = str(model_dtype).lower()

    if "float16" in dtype_str or "fp16" in dtype_str or "half" in dtype_str:
        return 2
    elif "float32" in dtype_str or "fp32" in dtype_str or "float" in dtype_str:
        return 4
    elif "bfloat16" in dtype_str or "bf16" in dtype_str:
        return 2
    elif "int8" in dtype_str or "fp8" in dtype_str:
        return 1
    elif "auto" in dtype_str:
        # Use model_dtype_str for string comparisons (model_dtype may be torch.dtype)
        if "auto" in model_dtype_str:
            return 2
        elif any(
            t in model_dtype_str
            for t in ["half", "float16", "fp16", "bfloat16", "bf16"]
        ):
            return 2
        elif any(t in model_dtype_str for t in ["float32", "fp32"]):
            return 4
        else:
            raise ValueError(f"Unknown model dtype: {model_dtype}")

    else:
        logger.warning(
            f"Unknown cache dtype: {cache_dtype}, defaulting to 2 bytes (FP16)"
        )
        raise ValueError(f"Unknown cache dtype: {cache_dtype}")
