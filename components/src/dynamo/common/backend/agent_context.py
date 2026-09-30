# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for reading agent context from backend requests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def agent_context_from_request(
    request: Mapping[str, Any],
) -> Mapping[str, Any] | None:
    """Return a request's serialized agent context when it is object-shaped."""
    agent_context = request.get("agent_context")
    return agent_context if isinstance(agent_context, Mapping) else None


def session_id_from_request(request: Mapping[str, Any]) -> str | None:
    """Return the request's non-empty ``agent_context.session_id``."""
    agent_context = agent_context_from_request(request)
    if agent_context is None:
        return None
    session_id = agent_context.get("session_id")
    if not isinstance(session_id, str):
        return None
    session_id = session_id.strip()
    return session_id or None
