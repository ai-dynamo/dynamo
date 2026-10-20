# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Mapping
from typing import Any, Optional


def normalize_top_k_for_trtllm(top_k: int) -> int:
    """Translate Dynamo's disabled top-k sentinel to TRT-LLM's sentinel."""
    return 0 if top_k == -1 else top_k


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
    sampling_params: Any,
    stop_conditions: Mapping[str, Any],
    *,
    no_stop_trim: bool = False,
) -> None:
    """Apply Dynamo stop conditions using TRT-LLM stopping semantics.

    If the request asks for ``ignore_eos`` or ``no_stop_trim``, disable TRT-LLM
    stopping and let the Dynamo decoder return the stop token and terminate the
    request. TRT-LLM maps ``ignore_eos=True`` to both ``end_id=-1`` and an empty
    stop-word list when it builds the executor request.

    Otherwise visible stop tokens stop generation in-engine like hidden ones,
    with ``include_stop_str_in_output`` so TRT-LLM still returns the token to
    the Dynamo decoder and tool parser. Mapping visible tokens to ``ignore_eos``
    made TRT-LLM decode past them until the frontend saw the token in a
    streamed chunk (up to ``stream_interval`` tokens later) and cancelled the
    request; with a tool parser whose end token is the model EOS, every request
    with tools ended in a cancel.

    TRT-LLM only holds back EOS until ``min_tokens``; stop words still fire
    early, so visible stop tokens stay with the Dynamo decoder when a minimum
    is requested.
    """
    visible_stop_token_ids = set(stop_conditions.get("stop_token_ids_visible") or [])
    min_tokens = stop_conditions.get("min_tokens")
    frontend_stops = bool(
        no_stop_trim
        or stop_conditions.get("ignore_eos")
        or (min_tokens and visible_stop_token_ids)
    )
    if frontend_stops:
        sampling_params.ignore_eos = True
    elif visible_stop_token_ids:
        engine_stop_token_ids = set(sampling_params.stop_token_ids or [])
        engine_stop_token_ids.update(visible_stop_token_ids)
        sampling_params.stop_token_ids = list(engine_stop_token_ids)
        sampling_params.include_stop_str_in_output = True

    if min_tokens:
        sampling_params.min_tokens = min_tokens

    hidden_stop_token_ids = stop_conditions.get("stop_token_ids_hidden")
    if hidden_stop_token_ids:
        engine_stop_token_ids = set(sampling_params.stop_token_ids or [])
        engine_stop_token_ids.update(hidden_stop_token_ids)
        if frontend_stops:
            engine_stop_token_ids.difference_update(visible_stop_token_ids)
        sampling_params.stop_token_ids = list(engine_stop_token_ids)
    elif frontend_stops and visible_stop_token_ids and sampling_params.stop_token_ids:
        sampling_params.stop_token_ids = list(
            set(sampling_params.stop_token_ids) - visible_stop_token_ids
        )
