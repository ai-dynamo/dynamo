# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward ``agent_context`` session ids to ``Engine.async_generate``.

``session_id`` and ``parent_session_id`` are passed as top-level kwargs of the same
name, filtered against the installed engine's signature so a build receives only
the kwargs it declares.

Never send them as ``session_params.id``: that field is an explicit lifecycle
handle SGLang rejects unless ``open_session`` created it, whereas the top-level
``session_id`` self-registers. ``GenerateReqInput`` also rejects ``session_id``
together with ``session_params``, so anything in this backend that starts sending
``session_params`` must suppress ``session_id`` on those requests. The full
behavior is documented under "Session identity" in ``AGENTS.md``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

from dynamo.sglang._compat import filter_supported_async_generate_kwargs

# agent_context fields forwarded verbatim as async_generate kwargs of the same name.
SESSION_ID_FIELDS = ("session_id", "parent_session_id")


def _clean_session_id(value: Any) -> Optional[str]:
    """Normalize one id; absent, blank and non-string all read as absent."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def session_ids_from_request(request: Mapping[str, Any]) -> dict[str, str]:
    """Return the well-formed ``agent_context`` session ids, keyed by field name.

    Each field is normalized independently, so a malformed ``session_id`` does not
    suppress a well-formed ``parent_session_id``. Returns ``{}`` when the request
    carries no usable agent context.
    """
    agent_context = request.get("agent_context")
    if not isinstance(agent_context, dict):
        return {}

    session_ids = {}
    for field in SESSION_ID_FIELDS:
        session_id = _clean_session_id(agent_context.get(field))
        if session_id is not None:
            session_ids[field] = session_id
    return session_ids


def agent_session_kwargs(engine: Any, request: Mapping[str, Any]) -> dict[str, Any]:
    """Build the optional SGLang per-request agent-session arguments."""
    session_ids = session_ids_from_request(request)
    if not session_ids:
        return {}
    return filter_supported_async_generate_kwargs(engine, session_ids)


__all__ = ["SESSION_ID_FIELDS", "agent_session_kwargs", "session_ids_from_request"]
