# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed runtime advertisement for vLLM causal decision scoring.

The v1 contract uses one causal decode token and selected raw token logprobs;
the Rust frontend owns rendering, the 26-candidate/64-question limits, and
response projection. This advertisement is not GPU or model-quality evidence.
"""

from __future__ import annotations

import hashlib
import json
import logging
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from inspect import signature
from pathlib import Path
from typing import TYPE_CHECKING, Any

from packaging.version import InvalidVersion, Version

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelRuntimeConfig, ModelType, WorkerType
from dynamo.vllm.decision_heads import PINNED_VLLM_COMMIT

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from dynamo.vllm.args import Config

VLLM_CAUSAL_DECISION_CAPABILITY = "vllm_causal_decision_v1"
PINNED_VLLM_VERSION = "0.32.1rc1.dev172+g9da5f2d31"
_CAUSAL_ARCHITECTURES = frozenset(
    {
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    }
)
# These source gates intentionally stand alone from the pooling-head gate:
# causal readiness must not import or require optional pooling implementations.
_PINNED_CAUSAL_SOURCES = (
    (
        "sampling_params.py",
        "3e55bb2dbd16f6d1effdfe1cf61ec3a2430769e1daff7bb687d634865c9551e7",
    ),
    (
        "outputs.py",
        "d6fc34a8766a8fa5364713e6c2f995d9f6c393e50de391370c78c21529b98893",
    ),
    (
        "config/model.py",
        "07cf2f7f8f774d5c147fbf470f2ce3bd99fbcfa359f9e8f6fa90a6a5d10094e0",
    ),
    (
        "model_executor/models/registry.py",
        "add79aa91e4c870ee33c740b16f5c8c85325211b38558af3039e125058fc897a",
    ),
    (
        "v1/engine/async_llm.py",
        "1ae2a81f39dc56f7136ec9aa19aedc4dd2d3f39601228f00a43d2d18687009a9",
    ),
    (
        "v1/engine/input_processor.py",
        "97f66659814177d3f7b72b959f09201cbdd1794de21ec1542fe8970f18a1a2b6",
    ),
    (
        "v1/engine/logprobs.py",
        "148cc1575651c1c39061752a16da620f17951e4528ba30ccafb105e5229ea61a",
    ),
    (
        "v1/engine/output_processor.py",
        "f855e23e61ad572dd463b2f97e0cb5c6786f7530b15eb05cdf5bf27c3b0c35a1",
    ),
    (
        "v1/sample/sampler.py",
        "7d3d001b8b6e1f25f02572df29609a4ab5c50abb023b48f5ebdb4752a2b5f7c5",
    ),
    (
        "v1/sample/metadata.py",
        "6582cfe1a240c7ddb975f0ef6dd7dcd28dcf0fc7670394390250013977aa7ffd",
    ),
    (
        "v1/worker/gpu_input_batch.py",
        "79f1dad9d093f404ad84d8e94906227d69620d356175654e4e8dd1377b55f388",
    ),
    (
        "v1/worker/gpu_model_runner.py",
        "9228cbd1d8b2a03527af7edd5e89764bbd3506efa9f6b3db24c4921162823141",
    ),
    (
        "v1/worker/gpu/sample/logprob.py",
        "ac58bfedaa325ae29e7b652063455239c7354a2af4751396930d5670f890ce2d",
    ),
    (
        "v1/worker/gpu/sample/sampler.py",
        "7f9941f9d7d04a7481ec393925ed5d000d4d77ee6db2ca6a8d9547c211d00a09",
    ),
)
_PINNED_QWEN35_SOURCE = (
    "model_executor/models/qwen3_5.py",
    "ea7e71303c902b9678e85ff3e9a75a7ab6e69b5770605d540ad6ab1084e18187",
)
_PINNED_GEMMA4_SOURCES = (
    (
        "model_executor/models/gemma4.py",
        "e3922b9373db89b1aec00b06f4128448056f324e36c9ce7b81927bda04da3a33",
    ),
    (
        "model_executor/models/gemma4_mm.py",
        "a5709ce3abce629728b8d71aa74c5814ec79dde5058ec44fe3fddc557d120b48",
    ),
)
_PINNED_CAUSAL_MODELS = {
    "Qwen3ForCausalLM": (
        (
            "model_executor/models/qwen3.py",
            "06bee457e1278cecc3d0745f45c44b4711243330e8c2c19d5c732625b67f1374",
        ),
    ),
    "Qwen3_5ForConditionalGeneration": (_PINNED_QWEN35_SOURCE,),
    "Qwen3_5MoeForConditionalGeneration": (_PINNED_QWEN35_SOURCE,),
    "Gemma4ForConditionalGeneration": _PINNED_GEMMA4_SOURCES,
    "Gemma4UnifiedForConditionalGeneration": (
        *_PINNED_GEMMA4_SOURCES,
        (
            "model_executor/models/gemma4_unified.py",
            "29743523e76f394a737056d4cbd8da3b42ee4fe6ee291c4f17d60e6530dd22fc",
        ),
    ),
}
logger = logging.getLogger(__name__)


def _qualified_version(installed: str) -> bool:
    if installed == PINNED_VLLM_VERSION:
        return True
    try:
        parsed = Version(installed)
    except InvalidVersion:
        return False
    return (
        parsed.release[:2] == (0, 32)
        and not parsed.is_prerelease
        and not parsed.is_devrelease
    )


def _pinned_main_available(architecture: str) -> bool:
    model_sources = _PINNED_CAUSAL_MODELS.get(architecture)
    if model_sources is None:
        return False
    try:
        if version("vllm") != PINNED_VLLM_VERSION:
            return False
        package = import_module("vllm")
        provenance = import_module("vllm._version")
        if (
            package.__version__ != PINNED_VLLM_VERSION
            or provenance.__version__ != PINNED_VLLM_VERSION
            or provenance.__commit_id__ != f"g{PINNED_VLLM_COMMIT[:9]}"
        ):
            return False
        root = Path(package.__file__).resolve().parent
        for relative, expected in (*_PINNED_CAUSAL_SOURCES, *model_sources):
            if hashlib.sha256((root / relative).read_bytes()).hexdigest() != expected:
                return False
    except (ImportError, OSError, RuntimeError, AttributeError, TypeError, ValueError):
        return False
    return True


def _decision_api_available(
    sampling_params_type: type[Any] | None,
    request_output_type: type[Any] | None,
) -> bool:
    # Probe only the version-qualified optional backend; ordinary generation
    # must not start depending on these newer API members.
    try:
        if sampling_params_type is None:
            sampling_params_type = import_module("vllm.sampling_params").SamplingParams
        if request_output_type is None:
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
        return False
    return "logprob_token_ids" in sampling_fields and {
        "num_cached_tokens",
        "num_cache_creation_tokens",
    }.issubset(output_fields)


def _qualified_main_model(model: Any) -> bool:
    try:
        return (
            model.is_diffusion is False
            and model.model_impl in ("auto", "vllm")
            and not model.model_class_overrides
            and model.convert_type == "none"
        )
    except AttributeError:
        return False


def _qualified_model(vllm_config: VllmConfig, *, pinned_main: bool) -> bool:
    model = vllm_config.model_config
    architectures = _PINNED_CAUSAL_MODELS if pinned_main else _CAUSAL_ARCHITECTURES
    return (
        model.architecture in architectures
        and model.runner_type == "generate"
        and model.is_encoder_decoder is False
        and (not pinned_main or _qualified_main_model(model))
        and model.logprobs_mode == "raw_logprobs"
        and type(model.max_model_len) is int
        and model.max_model_len >= 2
    )


def publish_vllm_decision_capability(
    runtime_config: ModelRuntimeConfig,
    model_input: ModelInput,
    model_type: ModelType,
    worker_type: WorkerType,
    vllm_config: VllmConfig,
    config: Config,
    *,
    native_generate_supported: bool,
    needs: list[list[WorkerType]] | None = None,
    vllm_version: str | None = None,
    sampling_params_type: type[Any] | None = None,
    request_output_type: type[Any] | None = None,
) -> bool:
    """Advertise only qualified aggregated token-mode causal scoring workers.

    A false result performs no runtime-metadata writes. Optional cache fields
    establish API availability, not that any request has measured cache usage;
    response handling must preserve missing measurements as unknown.
    """
    try:
        installed = version("vllm") if vllm_version is None else vllm_version
    except PackageNotFoundError:
        return False
    pinned_main = installed == PINNED_VLLM_VERSION
    if (
        native_generate_supported is not True
        or model_input != ModelInput.Tokens
        or worker_type != WorkerType.Aggregated
        or not (model_type.supports_chat() or model_type == ModelType.Completions)
        or any(needs or ())
        or not _qualified_version(installed)
    ):
        return False
    if (
        config.disaggregation_mode != DisaggregationMode.AGGREGATED
        or config.enable_rl
        or config.route_to_encoder
        or config.custom_encoder_class is not None
        or vllm_config.lora_config is not None
        or vllm_config.speculative_config is not None
        or vllm_config.kv_transfer_config is not None
        or vllm_config.ec_transfer_config is not None
        or not _qualified_model(vllm_config, pinned_main=pinned_main)
    ):
        return False
    if pinned_main and not _pinned_main_available(
        vllm_config.model_config.architecture
    ):
        return False
    if not _decision_api_available(sampling_params_type, request_output_type):
        return False
    runtime_config.set_engine_specific(
        VLLM_CAUSAL_DECISION_CAPABILITY, json.dumps(True)
    )
    return True
