# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Logprob helpers for engines using SGLang's cumulative ``meta_info`` shape.

Currently used by the SGLang and TokenSpeed backends. Pure data
transforms over ``output_options`` and ``meta_info``; no engine
runtime dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from .logprob_wire import TopLogprob, build_chunk

logger = logging.getLogger(__name__)

# Escape hatch: set to "1" (or any truthy value) to allow top_logprobs_num >= 1.
# Default-off because SGLang's tokenizer manager detokenizes top-k tokens
# per-position serially (O(N) per generated token), causing severe latency
# degradation. Flip once upstream lands batched top-logprob detokenization:
# https://github.com/sgl-project/sglang/pull/24447
_ALLOW_TOP_LOGPROBS_ENV = "DYN_SGL_ALLOW_TOP_LOGPROBS"

TOP_LOGPROBS_UNSUPPORTED_MSG = (
    "Dynamo's SGLang backend does not currently support logprobs >= 1 due to "
    "an O(N) per-position detokenization in the upstream sglang tokenizer "
    "manager. Use logprobs=0 for chosen-token logprobs, or set "
    "DYN_SGL_ALLOW_TOP_LOGPROBS=1 to override at your own risk. "
    "Track the upstream fix at https://github.com/sgl-project/sglang/pull/24447."
)


def top_logprobs_allowed() -> bool:
    """Return True if the DYN_SGL_ALLOW_TOP_LOGPROBS escape hatch is enabled."""
    return os.environ.get(_ALLOW_TOP_LOGPROBS_ENV, "").lower() not in ("", "0", "false")


def build_logprob_kwargs(request: dict[str, Any]) -> dict[str, Any]:
    """Build logprob kwargs for SGLang ``async_generate`` from output_options.

    Maps the Dynamo ``output_options`` format (shared with vLLM/TRT-LLM) to
    SGLang's ``async_generate`` keyword arguments:

      - ``return_logprob`` (bool): enables logprob computation
      - ``top_logprobs_num`` (int): number of top-k logprobs per token
      - ``logprob_start_len`` (int): absolute position in the sequence where
        logprob computation begins. SGLang defaults this to -1, which means
        ``len(prompt) - 1`` (output tokens only). Setting it to 0 computes
        from the start of the prompt — this is how we implement
        ``prompt_logprobs``.

    Returns the kwarg dict; empty when logprobs weren't requested.
    """
    kwargs: dict[str, Any] = {}
    output_options = request.get("output_options") or {}
    if not output_options:
        return kwargs

    allow_top = top_logprobs_allowed()

    def _parse(name: str, value: Any) -> Optional[int]:
        try:
            parsed = int(value)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid %s value: %r (must be integer), ignoring", name, value
            )
            return None
        if parsed < 0:
            logger.warning(
                "Invalid %s value: %r (must be non-negative), ignoring", name, value
            )
            return None
        if parsed >= 1 and not allow_top:
            raise ValueError(TOP_LOGPROBS_UNSUPPORTED_MSG)
        return parsed

    logprobs_value = output_options.get("logprobs")
    if logprobs_value is not None:
        parsed = _parse("logprobs", logprobs_value)
        if parsed is not None:
            kwargs["return_logprob"] = True
            kwargs["top_logprobs_num"] = parsed

    prompt_logprobs_value = output_options.get("prompt_logprobs")
    if prompt_logprobs_value is not None:
        parsed = _parse("prompt_logprobs", prompt_logprobs_value)
        if parsed is not None:
            kwargs["return_logprob"] = True
            # SGLang has a single top_logprobs_num for both prompt and
            # output tokens, so take the max of the two.
            kwargs["top_logprobs_num"] = max(kwargs.get("top_logprobs_num", 0), parsed)
            # logprob_start_len=0 computes from prompt start; omitting it
            # (or -1) computes output tokens only.
            kwargs["logprob_start_len"] = 0

    # Belt-and-suspenders: if return_logprob was requested and the gate is
    # not open, pin top_logprobs_num=0 so no future code path can flip it on.
    if kwargs.get("return_logprob") and not allow_top:
        kwargs["top_logprobs_num"] = 0

    return kwargs


def extract_logprobs(
    meta_info: dict[str, Any],
    num_output_logprobs_so_far: int,
    return_tokens_as_token_ids: bool = False,
) -> tuple[Optional[list[float]], Optional[list[list[dict[str, Any]]]], int]:
    """Slice SGLang's cumulative ``meta_info`` arrays into Dynamo's wire shape.

    SGLang keeps ``output_token_logprobs`` / ``output_top_logprobs``
    cumulative across chunks even with ``stream_output=True``, so the
    caller threads ``num_output_logprobs_so_far`` between calls.
    Returns ``(log_probs, top_logprobs, new_total)``.
    """
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if not output_token_logprobs:
        return None, None, num_output_logprobs_so_far

    output_top = meta_info.get("output_top_logprobs")
    # SGLang normally grows the two arrays in lockstep, but if a version
    # regression ever desynchronises them advance only by the shorter
    # array so the next slice doesn't read ahead of either side.
    safe_len = (
        min(len(output_token_logprobs), len(output_top))
        if output_top
        else len(output_token_logprobs)
    )
    # Hold the high-water mark: if the cumulative arrays transiently
    # shrink (e.g. an engine regression rolls back length), keep the
    # persisted cursor where it was — rolling back would cause the next
    # growth tick to re-emit positions that were already streamed.
    high_water = max(num_output_logprobs_so_far, safe_len)
    if num_output_logprobs_so_far >= safe_len:
        return None, None, high_water
    new_logprobs = output_token_logprobs[num_output_logprobs_so_far:safe_len]
    if not new_logprobs:
        return None, None, high_water

    selected = [float(entry[0]) for entry in new_logprobs]

    top_per_position: list[list[TopLogprob]] | None = None
    if output_top:
        new_top = output_top[num_output_logprobs_so_far:safe_len]
        if new_top:
            top_per_position = []
            for position in new_top:
                if position is None:
                    top_per_position.append([])
                    continue
                top_per_position.append(
                    [
                        TopLogprob(
                            rank=rank_idx + 1,
                            token_id=entry[1],
                            token=(
                                f"token_id:{entry[1]}"
                                if return_tokens_as_token_ids
                                else entry[2]
                            ),
                            logprob=float(entry[0]),
                        )
                        for rank_idx, entry in enumerate(position)
                    ]
                )

    log_probs, top_logprobs = build_chunk(selected, top_per_position)
    return log_probs, top_logprobs, high_water
