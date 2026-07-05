#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize the BFCL model context identity from OpenAI-compatible endpoints."""

from __future__ import annotations

from typing import Any


CONTEXT_WINDOW_ALIASES = ("context_window", "max_model_len")


class EndpointModelError(ValueError):
    """The served model entry does not identify the required context window."""


def canonical_endpoint_model(
    entry: dict[str, Any], expected_context_window: int
) -> dict[str, Any]:
    """Return a stable BFCL model identity after validating context aliases."""
    aliases = {
        field: entry.get(field)
        for field in CONTEXT_WINDOW_ALIASES
        if entry.get(field) is not None
    }
    if not aliases:
        raise EndpointModelError(
            "Endpoint model must provide a non-null context_window or max_model_len"
        )
    invalid = {
        field: value for field, value in aliases.items() if type(value) is not int
    }
    if invalid:
        raise EndpointModelError(
            f"Endpoint context aliases must be integers, got {invalid!r}"
        )
    if len(set(aliases.values())) != 1:
        raise EndpointModelError(f"Endpoint context aliases conflict: {aliases!r}")
    context_window = next(iter(aliases.values()))
    if context_window != expected_context_window:
        raise EndpointModelError(
            f"Endpoint context window {context_window!r} != "
            f"campaign {expected_context_window}"
        )
    return {
        "id": entry.get("id"),
        "object": entry.get("object"),
        "owned_by": entry.get("owned_by"),
        "context_window": context_window,
    }
