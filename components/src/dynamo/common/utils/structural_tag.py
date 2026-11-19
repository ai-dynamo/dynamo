# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from typing import Any

from dynamo.llm.exceptions import InvalidArgument


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


def validate_structural_tag_backend(
    structural_tag: str,
    backend: str,
    *,
    backend_label: str,
    supported_backends: tuple[str, ...],
    legacy_backend: str,
) -> None:
    """Reject unsupported tags before handing them to the selected engine."""
    if backend in supported_backends:
        return
    if backend == legacy_backend:
        try:
            tag = json.loads(structural_tag)
        except (json.JSONDecodeError, TypeError):
            # Let the engine report malformed grammar syntax.
            return
        if (
            isinstance(tag, dict)
            and tag.get("triggers") is not None
            and tag.get("structures") is not None
        ):
            return

    supported = " or ".join(repr(name) for name in supported_backends)
    raise InvalidArgument(
        f"{backend_label} {backend!r} does not support this structural tag. "
        f"Configure the worker to use {supported}."
    )
