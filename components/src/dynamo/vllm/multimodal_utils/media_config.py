# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend media decode and preprocessing discovery configuration."""

import json
import os
from pathlib import Path
from typing import Optional

from dynamo.llm import MediaDecoder, MediaFetcher, MediaPreprocessor


def _load_model_json(model: str, filename: str) -> str:
    local = Path(model) / filename
    if local.is_file():
        return local.read_text(encoding="utf-8")

    from transformers.utils import cached_file

    resolved = cached_file(model, filename)
    if resolved is None:
        raise ValueError(f"{model!r} does not provide {filename}")
    return Path(resolved).read_text(encoding="utf-8")


def create_frontend_media_config(
    enabled: bool,
    *,
    video_preprocessing: bool = False,
    model: Optional[str] = None,
    model_type: Optional[str] = None,
) -> tuple[Optional[MediaDecoder], Optional[MediaPreprocessor], Optional[MediaFetcher]]:
    """Build decoder, preprocessor, and SSRF-aware fetch policies.

    Video preprocessing is deliberately negotiated as a separate discovery
    capability. This keeps decode-only consumers compatible while the model
    processor is selected independently of the inference backend.
    """
    if not enabled:
        if video_preprocessing:
            raise ValueError("video preprocessing requires frontend decoding")
        return None, None, None

    media_decoder = MediaDecoder()
    media_decoder.enable_image({"limits": {"max_alloc": 128 * 1024 * 1024}})
    media_preprocessor = None

    if video_preprocessing:
        if not model:
            raise ValueError("video preprocessing requires a model name or path")

        if model_type is None:
            model_config = json.loads(_load_model_json(model, "config.json"))
            model_type = model_config.get("model_type")
        if not model_type:
            raise ValueError(f"Could not resolve model_type for model {model!r}")

        max_frames = int(os.getenv("DYN_MM_VIDEO_MAX_FRAMES", "32"))
        if max_frames <= 0:
            raise ValueError("DYN_MM_VIDEO_MAX_FRAMES must be greater than zero")
        media_decoder.enable_video(
            {
                "fps": 2.0,
                "max_frames": max_frames,
                "limits": {"max_alloc": 512 * 1024 * 1024},
            }
        )
        media_preprocessor = MediaPreprocessor()
        media_preprocessor.enable_video(
            model_type,
            _load_model_json(model, "preprocessor_config.json"),
        )

    media_fetcher = MediaFetcher()
    media_fetcher.timeout_ms(30000)
    allow_internal = os.getenv("DYN_MM_ALLOW_INTERNAL", "0") == "1"
    media_fetcher.allow_direct_ip(allow_internal)
    media_fetcher.allow_direct_port(allow_internal)
    return media_decoder, media_preprocessor, media_fetcher
