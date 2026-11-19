# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation at the configuration and model boundaries of strict RL serving."""

import os
from typing import TYPE_CHECKING

import torch

from .constants import DisaggregationMode

if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.engine.arg_utils import AsyncEngineArgs

    from .args import Config


def validate_strict_rl_config(config: "Config") -> None:
    """Require explicit RL opt-in and valid control-plane credentials."""
    if not config.enable_rl:
        raise ValueError("--enable-rl-weight-transfer requires --enable-rl")
    token = os.environ.get("DYN_SYSTEM_ENGINE_AUTH_TOKEN", "")
    if len(token) < 32 or any(not ("!" <= character <= "~") for character in token):
        raise ValueError(
            "strict RL weight transfer requires DYN_SYSTEM_ENGINE_AUTH_TOKEN with at least 32 ASCII graphic characters"
        )


def validate_strict_rl_profile(config: "Config", engine: "AsyncEngineArgs") -> None:
    """Reject startup modes that bypass strict routing or BF16 parity."""
    if config.benchmark_mode is not None:
        raise ValueError("strict RL weight transfer does not support benchmark mode")
    if (
        config.disaggregation_mode != DisaggregationMode.AGGREGATED
        or config.embedding_worker
        or config.classify_worker
        or config.headless
        or config.use_vllm_tokenizer
        or config.enable_multimodal
    ):
        raise ValueError(
            "strict RL weight transfer requires aggregated text generation"
        )
    if any(
        size != 1
        for size in (
            engine.tensor_parallel_size,
            engine.data_parallel_size,
            engine.pipeline_parallel_size,
        )
    ):
        raise ValueError("strict RL weight transfer requires TP=DP=PP=1")
    if (
        engine.enable_lora
        or engine.speculative_config is not None
        or engine.quantization is not None
        or engine.kv_cache_dtype not in ("auto", "bfloat16")
    ):
        raise ValueError(
            "strict RL weight transfer requires BF16 KV cache and does not support LoRA, speculative decoding or quantization"
        )
    if (
        config.gms_shadow_mode
        or engine.load_format == "gms"
        or os.environ.get("DYN_GMS_USE_V1") == "true"
    ):
        raise ValueError("strict RL weight transfer does not support GMS")
    if (
        engine.weight_transfer_config is None
        or engine.weight_transfer_config.backend not in ("ipc", "nccl")
    ):
        raise ValueError(
            "strict RL weight transfer requires an explicit ipc or nccl weight-transfer backend"
        )
    if engine.generation_config != "vllm" or engine.override_generation_config:
        raise ValueError(
            "strict RL weight transfer requires --generation-config=vllm and no sampling overrides"
        )


def validate_strict_rl_model(model: "ModelConfig") -> None:
    """Validate the loaded weights before publishing strict worker routes."""
    if (
        model.is_moe
        or model.is_quantized
        or model.quantization is not None
        or model.dtype != torch.bfloat16
    ):
        raise ValueError(
            "strict RL weight transfer requires a dense BF16 text-only token-in/token-out worker"
        )
