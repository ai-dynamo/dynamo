# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared custom-encoder loading and request extraction."""

from __future__ import annotations

import importlib
from typing import Any

from dynamo.vllm.multimodal_utils.custom_encoder.adapter import (
    CustomEncoderAdapter,
    create_custom_encoder_adapter,
)
from dynamo.vllm.multimodal_utils.custom_encoder.async_encoder import AsyncVisionEncoder
from dynamo.vllm.multimodal_utils.custom_encoder.backend import VisionEncoderBackend

IMAGE_URL_KEY = "image_url"
URL_VARIANT_KEY = "Url"


def load_custom_encoder(
    config: Any,
    model_config: Any,
    *,
    actor_name: str = "vision-encoder",
) -> tuple[AsyncVisionEncoder, CustomEncoderAdapter]:
    """Load the configured backend and bind its decoder-facing adapter."""

    custom_encoder_class = config.custom_encoder_class
    if not custom_encoder_class:
        raise ValueError("custom encoder class is required")
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
    adapter = create_custom_encoder_adapter(
        backend,
        model_config,
        config.engine_args,
    )
    encoder = AsyncVisionEncoder(backend, name=actor_name)
    encoder.load(config.model)
    return encoder, adapter


def extract_custom_encoder_image_urls(request: dict[str, Any]) -> list[str]:
    """Return ordered image URLs, rejecting unsupported or malformed media."""

    multimodal_data = request.get("multi_modal_data") or {}
    unsupported = sorted(
        key for key, value in multimodal_data.items() if key != IMAGE_URL_KEY and value
    )
    if unsupported:
        raise ValueError(
            "CustomEncoder supports image inputs only; got unsupported "
            f"multimodal data: {unsupported}"
        )

    image_items = multimodal_data.get(IMAGE_URL_KEY) or []
    image_urls = [
        item[URL_VARIANT_KEY]
        for item in image_items
        if isinstance(item, dict) and URL_VARIANT_KEY in item
    ]
    if len(image_urls) != len(image_items):
        raise ValueError(
            "CustomEncoder received image multimodal data but only "
            f"{len(image_urls)} of {len(image_items)} item(s) had a usable "
            "'Url'; each item must be a dict with a 'Url' key"
        )
    return image_urls
