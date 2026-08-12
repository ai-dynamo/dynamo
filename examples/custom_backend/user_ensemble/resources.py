# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared encoder and decoder resource construction for both placements."""

from __future__ import annotations

from typing import Any

from vllm.config import ModelConfig
from vllm.usage.usage_lib import UsageContext

from dynamo.vllm.args import Config
from dynamo.vllm.decoder_runtime import VllmDecoderRuntime
from dynamo.vllm.decoder_stage import VllmDecoderStage
from dynamo.vllm.main import setup_vllm_engine
from dynamo.vllm.multimodal_utils.custom_encoder import (
    AsyncVisionEncoder,
    VisionEncoderBackend,
    create_custom_encoder_adapter,
)
from examples.custom_backend.user_ensemble.stages import EncoderStage


def build_encoder_stage(
    config: Config,
    backend_type: type[VisionEncoderBackend[Any, Any, Any]],
    *,
    name: str,
    model_config: ModelConfig | None = None,
) -> tuple[EncoderStage, AsyncVisionEncoder[Any, Any, Any]]:
    """Construct and load one encoder stage and its owned backend."""

    resolved_model_config = model_config
    if resolved_model_config is None:
        resolved_model_config = config.engine_args.create_engine_config(
            usage_context=UsageContext.OPENAI_API_SERVER
        ).model_config
    backend = backend_type()
    adapter = create_custom_encoder_adapter(
        backend,
        resolved_model_config,
        config.engine_args,
    )
    encoder = AsyncVisionEncoder(backend, name=name)
    try:
        encoder.load(config.model)
    except BaseException:
        encoder.shutdown()
        raise
    return EncoderStage(encoder, adapter), encoder


def build_decoder_stage(
    config: Config,
) -> tuple[VllmDecoderStage, VllmDecoderRuntime, Any | None]:
    """Construct one decoder stage and its owned runtime resources."""

    (
        engine_client,
        vllm_config,
        default_sampling_params,
        prometheus_temp_dir,
        _component_gauges,
    ) = setup_vllm_engine(config)
    decoder_runtime = VllmDecoderRuntime(
        engine=engine_client,
        vllm_config=vllm_config,
        default_sampling_params=default_sampling_params,
    )
    return VllmDecoderStage(decoder_runtime), decoder_runtime, prometheus_temp_dir


def cleanup_resources(
    encoder: AsyncVisionEncoder[Any, Any, Any] | None,
    decoder_runtime: VllmDecoderRuntime | None,
    prometheus_temp_dir: Any | None,
) -> None:
    """Release independently owned resources even if one cleanup fails."""

    try:
        if encoder is not None:
            encoder.shutdown()
    finally:
        try:
            if prometheus_temp_dir is not None:
                prometheus_temp_dir.cleanup()
        finally:
            if decoder_runtime is not None:
                decoder_runtime.shutdown()
