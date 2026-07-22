# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for engine-owned conversation-aware ADP routing (opt-in).

When TensorRT-LLM's ``attention_dp_config.kv_cache_routing_conversation_affinity`` is
enabled, the engine's ``ConversationAwareADPRouter`` pins a conversation to an
attention-DP rank (sticky, load-balanced first turn). Dynamo must then:

  1. forward the stable conversation id via ``ConversationParams``, and
  2. NOT force ``attention_dp_rank`` — an explicit rank is honored *before* affinity
     in the engine and would bypass it.

The conversation id is the frontend-forwarded ``agent_context.session_id`` — no new
routing plumbing is added on the Rust side (see the serialized
``PreprocessedRequest.agent_context`` field). When affinity is off, callers keep their
existing Dynamo-owned DP-rank forcing unchanged.

``ConversationParams`` and ``kv_cache_routing_conversation_affinity`` require a
TensorRT-LLM build newer than 1.3.0rc20; on older wheels the import is absent and
``CONVERSATION_PARAMS_AVAILABLE`` is False.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional

try:  # Requires a TensorRT-LLM release newer than 1.3.0rc20.
    from tensorrt_llm.llmapi import ConversationParams
except ImportError:  # pragma: no cover - depends on installed wheel
    ConversationParams = None  # type: ignore[assignment]

CONVERSATION_PARAMS_AVAILABLE: bool = ConversationParams is not None


def session_id_from_request(request: Mapping[str, Any]) -> Optional[str]:
    """Return the stable conversation/session id the frontend forwards as
    ``agent_context.session_id``, or ``None`` if absent, blank, or malformed.

    Uses ``session_id`` (the active reasoning/tool chain), not ``parent_session_id``.
    """
    agent_context = request.get("agent_context")
    if not isinstance(agent_context, dict):
        return None
    session_id = agent_context.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def engine_conversation_affinity_enabled(llm: Any) -> bool:
    """Whether the resolved engine config enables conversation-affinity ADP routing.

    Reads ``llm.args.attention_dp_config.kv_cache_routing_conversation_affinity``.
    Only a genuine boolean ``True`` counts as enabled; tolerates a config model or a
    plain dict, and defaults to False when the field is absent (e.g. older wheels).
    """
    adp_cfg = getattr(getattr(llm, "args", None), "attention_dp_config", None)
    if adp_cfg is None:
        return False
    if isinstance(adp_cfg, dict):
        return adp_cfg.get("kv_cache_routing_conversation_affinity") is True
    return getattr(adp_cfg, "kv_cache_routing_conversation_affinity", None) is True


def conversation_params_for(session_id: Optional[str]) -> Optional[Any]:
    """Build ``ConversationParams`` for ``session_id``, or ``None`` when there is no id
    (the engine router then applies its no-id balancing fallback). Requires the API."""
    if session_id is None or ConversationParams is None:
        return None
    return ConversationParams(conversation_id=session_id)
