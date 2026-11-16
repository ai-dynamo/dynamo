# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility helpers for vLLM model configuration."""

from typing import Any


def _hidden_size(model_config: Any) -> int:
    getter = getattr(model_config, "get_hidden_size", None)
    value = getter() if callable(getter) else None
    if value is None:
        hf_config = getattr(model_config, "hf_config", None)
        text_config = getattr(hf_config, "text_config", None)
        value = getattr(text_config, "hidden_size", None)
        if value is None:
            value = getattr(hf_config, "hidden_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("Could not resolve the model hidden size")
    return value


def _model_architectures(model_config: Any) -> tuple[str, ...]:
    architectures = getattr(model_config, "architectures", None)
    if architectures is None:
        hf_config = getattr(model_config, "hf_config", None)
        architectures = getattr(hf_config, "architectures", None)
    return tuple(str(architecture) for architecture in (architectures or ()))


def _is_multimodal_model(model_config: Any) -> bool:
    value = getattr(model_config, "is_multimodal_model", False)
    return bool(value() if callable(value) else value)
