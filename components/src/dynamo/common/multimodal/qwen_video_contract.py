# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen video resize detection shared by the vLLM and SGLang workers.

Each backend passes in the ``smart_resize`` callable from its own Transformers
install, so this module imports neither Transformers nor an engine.
"""

import logging
from collections.abc import Callable
from typing import Any, Optional

QWEN_VIDEO_RESIZE_LEGACY_CEIL = "legacy_ceil"
QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN = "round_ties_even"


def resolve_qwen_video_resize_mode(
    smart_resize: Optional[Callable[..., Any]],
    *,
    logger: logging.Logger,
    feature: str,
) -> Optional[str]:
    """Identify the temporal resize rule of the given Qwen3 ``smart_resize``.

    ``feature`` names what gets disabled in the warning, for example
    ``"video-aware KV routing"``. Returns ``None`` after logging a warning when
    the rule cannot be identified.
    """
    if smart_resize is None:
        logger.warning(
            "Exact %s disabled because the installed "
            "Transformers package has no Qwen3 video processor",
            feature,
        )
        return None
    try:
        # Probe chosen so the two known rounding rules give different results.
        result = smart_resize(
            num_frames=5,
            height=1120,
            width=3760,
            temporal_factor=2,
            factor=32,
            min_pixels=4096,
            max_pixels=25165824,
        )
    except (TypeError, ValueError) as error:
        logger.warning(
            "Exact %s disabled because the installed Qwen "
            "smart_resize API is unsupported: %s",
            feature,
            error,
        )
        return None
    if result == (1216, 4096):
        return QWEN_VIDEO_RESIZE_LEGACY_CEIL
    if result == (1120, 3776):
        return QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN
    logger.warning(
        "Exact %s disabled because the installed Qwen "
        "smart_resize behavior is unsupported: %s",
        feature,
        result,
    )
    return None
