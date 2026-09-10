# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Forward agent session identity to the SGLang engine.

The frontend derives agent identity at the HTTP boundary -- from
``x-dynamo-session-id`` / ``x-dynamo-parent-session-id``, or from the Claude Code,
Codex and OpenCode header families -- and serializes it on
``PreprocessedRequest.agent_context``. This module lifts the two session ids back
out and passes them to ``Engine.async_generate``. No routing plumbing is added on
the Rust side; the same field feeds TensorRT-LLM's conversation affinity (see
``dynamo/trtllm/conversation_affinity.py``, which reads ``session_id`` from the same
place).

``session_id`` is SGLang's passive session-aware radix ownership key (sglang >= 0.5.15).
It is deliberately *not* ``session_params.id``: the two fields have opposite
registration rules. ``session_params.id`` is an explicit lifecycle handle and is
rejected unless ``open_session`` created it, whereas the top-level ``session_id``
self-registers -- the scheduler calls ``ensure_session_generation``, which opens the
session the first time it sees the id. So forwarding an arbitrary agent session id
cannot produce an "unknown session" failure. The scheduler acts on it only under
``--enable-session-radix-cache`` (off by default); with the flag off the id is stored on
the request and nothing reads it.

``parent_session_id`` names the session that is blocked waiting on this one. An
engine that consumes it can keep the parent's prefix hot while it is parked on a
subagent it spawned, instead of letting it age out and be re-prefilled when the
subagent returns.

Both are optional ``async_generate`` kwargs, filtered against the installed
engine's signature: a build declaring neither receives neither, and a build
declaring only ``session_id`` receives only that.

``GenerateReqInput`` rejects ``session_id`` and ``session_params`` set together. Nothing
in this backend sends ``session_params``, so the two never collide -- but anything that
starts sending it must suppress ``session_id`` for those requests.
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
