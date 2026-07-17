# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Optional


def request_cache_salt(request: Mapping[str, Any]) -> Optional[str]:
    """Return the first non-empty cache_salt, preferring routing hints."""
    routing = request.get("routing") or {}
    if isinstance(routing, dict):
        cache_salt = routing.get("cache_salt")
        if cache_salt:
            return cache_salt

    extra_args = request.get("extra_args") or {}
    nvext = extra_args.get("nvext") if isinstance(extra_args, dict) else None
    if isinstance(nvext, dict):
        cache_salt = nvext.get("cache_salt")
        if cache_salt:
            return cache_salt

    return None


def stored_event_cache_salt(data: Mapping[str, Any]) -> Optional[str]:
    """Extract one cache salt from a TRT-LLM stored-event payload.

    TRT-LLM 1.3 serializes ``cache_salt`` on each item in ``data["blocks"]``.
    Keep the parent-level lookup as a compatibility fallback, but fail closed
    if an event combines blocks from different cache namespaces.
    """
    cache_salts: set[str] = set()

    parent_cache_salt = data.get("cache_salt")
    if parent_cache_salt is not None:
        cache_salts.add(parent_cache_salt)

    blocks = data.get("blocks") or []
    for block in blocks:
        if not isinstance(block, Mapping):
            continue
        block_cache_salt = block.get("cache_salt")
        if block_cache_salt is not None:
            cache_salts.add(block_cache_salt)

    if len(cache_salts) > 1:
        raise ValueError("stored KV event contains conflicting cache_salt values")

    return next(iter(cache_salts), None)


def apply_stop_conditions_to_sampling_params(
    sampling_params: Any, stop_conditions: Mapping[str, Any]
) -> None:
    """Apply Dynamo stop conditions using TRT-LLM stopping semantics.

    TRT-LLM cannot return a token that it consumes as an engine-side stop. If
    Dynamo marks a stop token as visible, disable TRT-LLM stopping and let the
    Dynamo decoder return the token and terminate the request. TRT-LLM maps
    ``ignore_eos=True`` to both ``end_id=-1`` and an empty stop-word list when
    it builds the executor request.
    """
    visible_stop_token_ids = set(stop_conditions.get("stop_token_ids_visible") or [])
    if stop_conditions.get("ignore_eos") or visible_stop_token_ids:
        sampling_params.ignore_eos = True

    min_tokens = stop_conditions.get("min_tokens")
    if min_tokens:
        sampling_params.min_tokens = min_tokens

    hidden_stop_token_ids = stop_conditions.get("stop_token_ids_hidden")
    if hidden_stop_token_ids:
        engine_stop_token_ids = set(sampling_params.stop_token_ids or [])
        engine_stop_token_ids.update(hidden_stop_token_ids)
        engine_stop_token_ids.difference_update(visible_stop_token_ids)
        sampling_params.stop_token_ids = list(engine_stop_token_ids)
    elif visible_stop_token_ids and sampling_params.stop_token_ids:
        sampling_params.stop_token_ids = list(
            set(sampling_params.stop_token_ids) - visible_stop_token_ids
        )
