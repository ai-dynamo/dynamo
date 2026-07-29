# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Model configuration helpers shared by custom encoder adapters."""

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
        raise ValueError("CustomEncoder could not resolve the decoder hidden size")
    return value


def _model_architectures(model_config: Any) -> tuple[str, ...]:
    hf_config = getattr(model_config, "hf_config", None)
    architectures = getattr(hf_config, "architectures", None)
    if architectures is None:
        architectures = getattr(model_config, "architectures", None)
    return tuple(str(architecture) for architecture in (architectures or ()))


def _is_multimodal_model(model_config: Any) -> bool:
    value = getattr(model_config, "is_multimodal_model", False)
    return bool(value() if callable(value) else value)


def _spatial_merge_size(model_config: Any) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    vision_config = getattr(hf_config, "vision_config", None)
    value = getattr(vision_config, "spatial_merge_size", None)
    if value is None:
        value = getattr(hf_config, "spatial_merge_size", None)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise ValueError("CustomEncoder could not resolve Qwen spatial_merge_size")
    return value


def _required_token_id(model_config: Any, name: str) -> int:
    hf_config = getattr(model_config, "hf_config", None)
    value = getattr(hf_config, name, None)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"CustomEncoder could not resolve Qwen {name}")
    return value
