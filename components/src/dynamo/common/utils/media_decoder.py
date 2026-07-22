# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

from dynamo.common.utils.env import env_bool

DYN_MM_ENABLE_LIBJPEG = "DYN_MM_ENABLE_LIBJPEG"
DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC = 128 * 1024 * 1024


def build_frontend_image_decoder_options(
    *,
    max_alloc: int = DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC,
) -> dict[str, Any]:
    return {
        "enable_libjpeg": env_bool(DYN_MM_ENABLE_LIBJPEG, default=True),
        "limits": {"max_alloc": max_alloc},
    }
