# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Neutral, import-light runtime qualification shared by decision executors."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


def is_exact_vllm_release(installed: str, release: str) -> bool:
    """Require the selected release, permitting only local wheel build metadata."""
    if not isinstance(installed, str):
        return False
    try:
        parsed = Version(installed)
    except InvalidVersion:
        return False
    return parsed.public == release


def pinned_default_scheduler(vllm_config: VllmConfig) -> bool:
    """Match the source-pinned resolved scheduler for each supported engine mode."""
    try:
        scheduler = vllm_config.scheduler_config
        async_mode = scheduler.async_scheduling
        if type(async_mode) is not bool:
            return False
        if vllm_config.model_config.is_diffusion:
            name = "DiffusionAsyncScheduler" if async_mode else "DiffusionScheduler"
            return scheduler.scheduler_cls == (
                f"vllm.v1.core.sched.diffusion_scheduler.{name}"
            )
        expected = (
            "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
            if async_mode
            else "vllm.v1.core.sched.scheduler.Scheduler"
        )
        return scheduler.scheduler_cls is None or scheduler.scheduler_cls == expected
    except AttributeError:
        return False


def qualified_runtime(
    vllm_config: VllmConfig, *, runner_type: str, diffusion: bool, min_model_len: int
) -> bool:
    """Share model and engine qualification without importing an optional runtime."""
    try:
        model = vllm_config.model_config
        return (
            model.runner_type == runner_type
            and model.is_encoder_decoder is False
            and model.model_impl in ("auto", "vllm")
            and not model.model_class_overrides
            and model.convert_type == "none"
            and model.is_diffusion is diffusion
            and type(model.max_model_len) is int
            and model.max_model_len >= min_model_len
            and pinned_default_scheduler(vllm_config)
            and type(vllm_config.parallel_config.data_parallel_size) is int
            and vllm_config.parallel_config.data_parallel_size == 1
            and all(
                value is None
                for value in (
                    vllm_config.lora_config,
                    vllm_config.speculative_config,
                    vllm_config.kv_transfer_config,
                    vllm_config.ec_transfer_config,
                )
            )
        )
    except AttributeError:
        logger.warning("vLLM decision runtime metadata unavailable")
        return False
