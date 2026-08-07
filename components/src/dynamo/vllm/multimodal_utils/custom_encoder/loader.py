# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve a user-provided custom encoder and its decoder adapter."""

import importlib
from dataclasses import dataclass
from typing import Any

from dynamo.vllm.multimodal_utils.custom_encoder.adapter import (
    CustomEncoderAdapter,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.backend import VisionEncoderBackend


@dataclass(frozen=True)
class PreparedCustomEncoder:
    """A resolved backend and its decoder-selected adapter."""

    backend: VisionEncoderBackend[Any, Any, Any]
    adapter: CustomEncoderAdapter[Any]


def prepare_custom_encoder(
    custom_encoder_class: str | None,
    model_config: Any,
    engine_args: Any,
) -> PreparedCustomEncoder | None:
    """Resolve and validate a CustomEncoder without starting its actor thread."""
    if not custom_encoder_class:
        return None

    module_path, _, class_name = custom_encoder_class.rpartition(".")
    backend_cls = getattr(importlib.import_module(module_path), class_name)
    if not (
        isinstance(backend_cls, type) and issubclass(backend_cls, VisionEncoderBackend)
    ):
        raise TypeError(
            f"--custom-encoder-class {custom_encoder_class!r} must resolve to a "
            f"VisionEncoderBackend subclass, got {backend_cls!r}."
        )

    backend = backend_cls()
    adapter = create_custom_encoder_adapter(backend, model_config, engine_args)
    return PreparedCustomEncoder(backend=backend, adapter=adapter)
