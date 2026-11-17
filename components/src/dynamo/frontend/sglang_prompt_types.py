# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Result shared by SGLang prompt renderers."""

from dataclasses import dataclass


@dataclass
class RenderedPrompt:
    token_ids: list[int]
    # None keeps the request's image order and string objects.
    image_urls: list[str] | None = None
