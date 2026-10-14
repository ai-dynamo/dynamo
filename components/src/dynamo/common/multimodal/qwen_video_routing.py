# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Engine-neutral Qwen video resize probe shared by the vLLM and SGLang workers."""

import logging
from collections.abc import Callable
from typing import Any, Optional

QWEN_VIDEO_RESIZE_LEGACY_CEIL = "legacy_ceil"
QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN = "round_ties_even"


def resolve_qwen_video_resize_mode(
    smart_resize: Optional[Callable[..., Any]],
    *,
    logger: logging.Logger,
    disabled_prefix: str,
) -> Optional[str]:
    """Identify the installed Transformers Qwen video resize rule.

    ``smart_resize`` is the Transformers Qwen3-VL ``smart_resize`` callable, or
    ``None`` when the installed package does not provide it. ``disabled_prefix``
    starts each warning (for example ``"Exact video-aware KV routing disabled"``)
    and ``logger`` is the calling backend's logger, so each backend keeps its
    own warning text and logger name.
    """
    if smart_resize is None:
        logger.warning(
            "%s because the installed Transformers package has no Qwen3 video "
            "processor",
            disabled_prefix,
        )
        return None
    try:
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
            "%s because the installed Qwen smart_resize API is unsupported: %s",
            disabled_prefix,
            error,
        )
        return None
    if result == (1216, 4096):
        return QWEN_VIDEO_RESIZE_LEGACY_CEIL
    if result == (1120, 3776):
        return QWEN_VIDEO_RESIZE_ROUND_TIES_EVEN
    logger.warning(
        "%s because the installed Qwen smart_resize behavior is unsupported: %s",
        disabled_prefix,
        result,
    )
    return None
