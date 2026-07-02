# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import Any

DYN_MM_IMAGE_DECODER_BACKEND = "DYN_MM_IMAGE_DECODER_BACKEND"
DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC = 128 * 1024 * 1024


def build_frontend_image_decoder_options(
    *,
    max_alloc: int = DEFAULT_FRONTEND_IMAGE_DECODER_MAX_ALLOC,
) -> dict[str, Any]:
    options: dict[str, Any] = {"limits": {"max_alloc": max_alloc}}
    backend = os.getenv(DYN_MM_IMAGE_DECODER_BACKEND, "").strip()
    if backend:
        options["backend"] = backend
    return options
