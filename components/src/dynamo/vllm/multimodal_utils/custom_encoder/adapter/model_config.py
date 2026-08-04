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


def _is_multimodal_model(model_config: Any) -> bool:
    value = getattr(model_config, "is_multimodal_model", False)
    return bool(value() if callable(value) else value)
