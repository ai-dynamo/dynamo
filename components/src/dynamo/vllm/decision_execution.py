# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advertise the qualified vLLM 0.33 causal decision execution profile."""

import logging
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

from packaging.version import InvalidVersion, Version

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
        "Gemma4ForConditionalGeneration",
    }
)


def _released_runtime() -> bool:
    try:
        installed = version("vllm")
        return isinstance(installed, str) and Version(installed).public == "0.33.0"
    except (PackageNotFoundError, InvalidVersion):
        return False


def _unqualified_runtime_condition(runtime: "VllmConfig") -> str | None:
    model = runtime.model_config
    scheduler = runtime.scheduler_config
    expected_scheduler = (
        "vllm.v1.core.sched.async_scheduler.AsyncScheduler"
        if scheduler.async_scheduling
        else "vllm.v1.core.sched.scheduler.Scheduler"
    )
    checks = (
        (
            f"unsupported model architecture {model.architecture!r}",
            model.architecture in _ARCHITECTURES,
        ),
        ("requires runner_type=generate", model.runner_type == "generate"),
        ("requires logprobs_mode=raw_logprobs", model.logprobs_mode == "raw_logprobs"),
        ("requires is_encoder_decoder=False", model.is_encoder_decoder is False),
        ("requires is_diffusion=False", model.is_diffusion is False),
        ("requires model_impl=auto or vllm", model.model_impl in ("auto", "vllm")),
        ("requires no model_class_overrides", not model.model_class_overrides),
        ("requires convert_type=none", model.convert_type == "none"),
        (
            "requires max_model_len integer >= 2",
            type(model.max_model_len) is int and model.max_model_len >= 2,
        ),
        (
            "requires max_num_seqs integer = 1",
            type(scheduler.max_num_seqs) is int and scheduler.max_num_seqs == 1,
        ),
        ("requires async_scheduling boolean", type(scheduler.async_scheduling) is bool),
        (
            "requires standard scheduler_cls",
            scheduler.scheduler_cls in (None, expected_scheduler),
        ),
        (
            "requires data_parallel_size integer = 1",
            type(runtime.parallel_config.data_parallel_size) is int
            and runtime.parallel_config.data_parallel_size == 1,
        ),
        ("requires lora_config=None", runtime.lora_config is None),
        ("requires speculative_config=None", runtime.speculative_config is None),
        ("requires kv_transfer_config=None", runtime.kv_transfer_config is None),
        ("requires ec_transfer_config=None", runtime.ec_transfer_config is None),
    )
    return next((name for name, qualified in checks if not qualified), None)


def qualify_vllm_decision_runtime(vllm_config: "VllmConfig", config: "Config") -> None:
    """Reject an unqualified resolved configuration before the engine starts.

    Cache fields alone do not establish measurement provenance, so their installed
    source owners must match the qualified release before loading the model.
    """
    if config.systemone_serial is not True:
        return
    if not _released_runtime():
        raise ValueError("--systemone-serial requires qualified vLLM 0.33.0")
    try:
        condition = _unqualified_runtime_condition(vllm_config)
    except AttributeError as exc:
        raise ValueError(
            "--systemone-serial cannot qualify missing runtime metadata"
        ) from exc
    if condition is not None:
        raise ValueError(
            f"--systemone-serial: runtime qualification failed: {condition}"
        )
    if not verify_cache_sources():
        raise ValueError("--systemone-serial requires qualified vLLM cache sources")


def publish_vllm_decision_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    worker_type: WorkerType,
    config: "Config",
    *,
    native_generate_supported: bool,
    needs: list[list[WorkerType]] | None = None,
) -> bool:
    """Publish a prequalified runtime's serialized raw candidate-read contract.

    The frontend owns question rendering and answer projection. Execution uses
    one generated token per question; this flag does not establish model quality.
    """
    if config.systemone_serial is not True:
        return False
    if (
        native_generate_supported is not True
        or model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or any(needs or ())
    ):
        raise ValueError(
            "--systemone-serial requires an aggregated token-input worker with "
            "native generate support and no downstream workers"
        )
    runtime_config.set_engine_specific(VLLM_CAUSAL_DECISION_CAPABILITY, "true")
    logger.info("Published vLLM causal decision capability (serialized raw scoring)")
    return True
