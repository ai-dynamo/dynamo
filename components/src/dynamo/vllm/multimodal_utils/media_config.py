# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frontend media-decoding discovery configuration for vLLM workers."""

import os
from typing import Optional

from dynamo.common.utils.media_decoder import enable_frontend_video_decoding
from dynamo.llm import MediaDecoder, MediaFetcher


def create_frontend_media_config(
    enabled: bool,
) -> tuple[Optional[MediaDecoder], Optional[MediaFetcher]]:
    """Build the legacy-compatible decoder and SSRF-aware fetch policy."""
    if not enabled:
        return None, None

    media_decoder = MediaDecoder()
    media_decoder.enable_image({"limits": {"max_alloc": 128 * 1024 * 1024}})
    enable_frontend_video_decoding(media_decoder)

    media_fetcher = MediaFetcher()
    media_fetcher.timeout_ms(30000)
    allow_internal = os.getenv("DYN_MM_ALLOW_INTERNAL", "0") == "1"
    media_fetcher.allow_direct_ip(allow_internal)
    media_fetcher.allow_direct_port(allow_internal)
    return media_decoder, media_fetcher
