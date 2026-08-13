# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared model and runtime configuration for the user ensemble."""

from __future__ import annotations

import importlib
from typing import Any, cast

from dynamo.vllm.args import Config, configure_rl_logprobs_mode, parse_args
from dynamo.vllm.multimodal_utils.custom_encoder import VisionEncoderBackend

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def load_encoder_backend(
    class_path: str,
) -> type[VisionEncoderBackend[Any, Any, Any]]:
    """Resolve the configured encoder backend class."""

    module_name, separator, class_name = class_path.rpartition(".")
    if not separator:
        raise ValueError(
            "--custom-encoder-class must be a dotted module.ClassName path; "
            f"got {class_path!r}"
        )
    module = importlib.import_module(module_name)
    backend_type = getattr(module, class_name)
    if not isinstance(backend_type, type) or not issubclass(
        backend_type, VisionEncoderBackend
    ):
        raise TypeError(f"{class_path} must name a VisionEncoderBackend subclass")
    return cast(type[VisionEncoderBackend[Any, Any, Any]], backend_type)


def prepare_user_ensemble_config(
    argv: list[str] | None = None,
) -> tuple[Config, type[VisionEncoderBackend[Any, Any, Any]]]:
    """Parse the shared decoder and encoder configuration."""

    config = parse_args(argv)
    if not config.custom_encoder_class:
        raise ValueError(
            "--custom-encoder-class is required by the user ensemble example"
        )
    if not config.served_model_name:
        config.served_model_name = config.engine_args.served_model_name = config.model
    configure_rl_logprobs_mode(config)
    config.engine_args.enable_prompt_embeds = True
    return config, load_encoder_backend(config.custom_encoder_class)
