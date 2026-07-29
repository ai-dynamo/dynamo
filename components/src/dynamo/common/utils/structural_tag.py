# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any


def serialize_structural_tag(structural_tag: Any) -> str | None:
    """Return structural tag as the JSON string expected by backends."""
    if structural_tag is None:
        return None
    if hasattr(structural_tag, "model_dump"):
        structural_tag = structural_tag.model_dump()
    return (
        structural_tag
        if isinstance(structural_tag, str)
        else json.dumps(structural_tag)
    )
