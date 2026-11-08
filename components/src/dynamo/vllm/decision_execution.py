# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advertise the qualified vLLM 0.33 causal decision execution profile."""

import logging
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelRuntimeConfig, WorkerType

from .decision_cache_provenance import verify_cache_sources

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from .args import Config

logger = logging.getLogger(__name__)
VLLM_CAUSAL_DECISION_CAPABILITY = "vllm_causal_decision_v1"
_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
    }
)


def _released_runtime() -> bool:
    try:
        installed = version("vllm")
        return isinstance(installed, str) and Version(installed).public == "0.33.0"
    except (PackageNotFoundError, InvalidVersion):
        return False


def _scoring_api_available() -> bool:
    # The backend is optional in CPU installations. Inspect only its wire schema.
    try:
        params = signature(
            import_module("vllm.sampling_params").SamplingParams
        ).parameters
        output = signature(import_module("vllm.outputs").RequestOutput).parameters
        return "logprob_token_ids" in params and {
            "num_cached_tokens",
            "num_cache_creation_tokens",
        }.issubset(output)
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
        logger.warning("vLLM decisions disabled: scoring schema unavailable")
        return False


def _qualified_configuration(config: "Config", runtime: "VllmConfig") -> bool:
    model = runtime.model_config
    scheduler = runtime.scheduler_config
    expected_scheduler = (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        if scheduler.async_scheduling
        else "vllm.v1.core.sched.scheduler.Scheduler"
    )
    return (
        config.disaggregation_mode == DisaggregationMode.AGGREGATED
        and config.enable_rl is False
        and config.route_to_encoder is False
        and config.custom_encoder_class is None
        and config.enable_multimodal is False
        and config.realtime is False
        and model.architecture in _ARCHITECTURES
        and model.runner_type == "generate"
        and model.logprobs_mode == "raw_logprobs"
        and model.is_encoder_decoder is False
        and model.is_diffusion is False
        and model.model_impl in ("auto", "vllm")
        and not model.model_class_overrides
        and model.convert_type == "none"
        and type(model.max_model_len) is int
        and model.max_model_len >= 2
        and type(scheduler.max_num_seqs) is int
        and scheduler.max_num_seqs == 1
        and type(scheduler.async_scheduling) is bool
        and scheduler.scheduler_cls in (None, expected_scheduler)
        and type(runtime.parallel_config.data_parallel_size) is int
        and runtime.parallel_config.data_parallel_size == 1
        and all(
            extension is None
            for extension in (
                runtime.lora_config,
                runtime.speculative_config,
                runtime.kv_transfer_config,
                runtime.ec_transfer_config,
            )
        )
    )


def publish_vllm_decision_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    worker_type: WorkerType,
    vllm_config: "VllmConfig",
    config: "Config",
    *,
    native_generate_supported: bool,
    needs: list[list[WorkerType]] | None = None,
) -> bool:
    """Publish only the single-rank, serialized, raw candidate-read contract.

    The frontend owns question rendering and answer projection. Execution uses
    one generated token per question; this flag does not establish model quality.
    Cache fields alone do not establish measurement provenance, so their installed
    source owners must match the qualified release before advertising support.
    """
    if (
        native_generate_supported is not True
        or model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or any(needs or ())
        or not _released_runtime()
    ):
        return False
    try:
        if not _qualified_configuration(config, vllm_config):
            return False
    except AttributeError:
        logger.warning("vLLM decisions disabled: runtime metadata unavailable")
        return False
    if not verify_cache_sources() or not _scoring_api_available():
        return False
    runtime_config.set_engine_specific(VLLM_CAUSAL_DECISION_CAPABILITY, "true")
    return True
