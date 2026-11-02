# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed runtime advertisement for released vLLM causal scoring.

The v1 contract uses one causal decode token and selected raw token logprobs;
the Rust frontend owns rendering, the 26-candidate/64-question limits, and
response projection. This advertisement is not GPU or model-quality evidence.
Native decision adapters are optional and are not imported by this profile.
Qwen supports releases 0.32.0 and 0.33.0; Gemma requires release 0.33.0.
"""

from __future__ import annotations

import json
import logging
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from typing import TYPE_CHECKING

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelRuntimeConfig, WorkerType

from .decision_cache_provenance import qualified_cache_runtime
from .decision_runtime_policy import is_exact_vllm_release, qualified_runtime

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from dynamo.vllm.args import Config

VLLM_CAUSAL_DECISION_CAPABILITY = "vllm_causal_decision_v1"
_CAUSAL_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
_STABLE_GEMMA_ARCHITECTURES = frozenset(
    {"Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"}
)
logger = logging.getLogger(__name__)


def _decision_api_available() -> bool:
    # Probe only the version-qualified optional backend; ordinary generation
    # must not start depending on these newer API members.
    try:
        sampling_params_type = import_module("vllm.sampling_params").SamplingParams
        request_output_type = import_module("vllm.outputs").RequestOutput
    except (ImportError, OSError, RuntimeError, AttributeError):
        logger.warning(
            "vLLM decision scoring disabled: required API schema unavailable"
        )
        return False
    try:
        sampling_fields = signature(sampling_params_type).parameters
        output_fields = signature(request_output_type).parameters
    except (TypeError, ValueError):
        logger.warning(
            "vLLM decision scoring disabled: required API signature unavailable"
        )
        return False
    return "logprob_token_ids" in sampling_fields and {
        "num_cached_tokens",
        "num_cache_creation_tokens",
    }.issubset(output_fields)


def _qualified_model(vllm_config: VllmConfig, installed: str) -> bool:
    return (
        qualified_runtime(
            vllm_config, runner_type="generate", diffusion=False, min_model_len=2
        )
        and (
            vllm_config.model_config.architecture in _CAUSAL_ARCHITECTURES
            or (
                is_exact_vllm_release(installed, "0.33.0")
                and vllm_config.model_config.architecture in _STABLE_GEMMA_ARCHITECTURES
            )
        )
        and vllm_config.model_config.logprobs_mode == "raw_logprobs"
    )


def decision_worker_eligible(config: Config) -> bool:
    """Keep decision execution on aggregated, text-only non-RL workers."""
    try:
        return (
            config.disaggregation_mode == DisaggregationMode.AGGREGATED
            and not config.enable_rl
            and not config.route_to_encoder
            and config.custom_encoder_class is None
            and config.enable_multimodal is False
            and config.realtime is False
        )
    except AttributeError:
        logger.warning(
            "vLLM decision scoring disabled: worker mode metadata unavailable"
        )
        return False


def publish_vllm_decision_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    worker_type: WorkerType,
    vllm_config: VllmConfig,
    config: Config,
    *,
    native_generate_supported: bool,
    needs: list[list[WorkerType]] | None = None,
) -> bool:
    """Advertise only qualified aggregated token-mode causal scoring workers.

    A false result performs no runtime-metadata writes. Optional cache fields
    establish API availability, not measurement provenance. Source-qualified
    schedulers establish provenance; the Rust collector validates all counters.
    Model-type eligibility comes from the native Generate publisher's result.
    """
    try:
        installed = version("vllm")
    except PackageNotFoundError:
        return False
    if (
        native_generate_supported is not True
        or model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or any(needs or ())
    ):
        return False
    try:
        qualified = (
            decision_worker_eligible(config)
            and type(vllm_config.scheduler_config.max_num_seqs) is int
            and vllm_config.scheduler_config.max_num_seqs == 1
            and _qualified_model(vllm_config, installed)
        )
    except AttributeError:
        logger.warning("vLLM decision scoring disabled: runtime metadata unavailable")
        return False
    if not qualified:
        return False
    if not qualified_cache_runtime(installed):
        return False
    if not _decision_api_available():
        return False
    runtime_config.set_engine_specific(
        VLLM_CAUSAL_DECISION_CAPABILITY, json.dumps(True)
    )
    return True
