# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared logprob helpers.

vLLM and TRT-LLM expose logprobs through ``CompletionOutput.logprobs``
(list aligned with ``token_ids``, dicts of ``token_id -> LogprobInfo``).
SGLang exposes them through ``meta_info["output_token_logprobs"]`` as
cumulative tuples ``(logprob, token_id, text_or_None)``.

Both paths emit the same Dynamo wire format on ``GenerateChunk``:
``log_probs`` is a flat ``list[float]``, ``top_logprobs`` is
``list[list[{rank, token_id, token, logprob, bytes?}]]``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


def parse_logprob_options(
    output_options: dict[str, Any],
) -> tuple[Optional[int], Optional[int]]:
    """Return ``(logprobs, prompt_logprobs)`` parsed as non-negative ints,
    or ``None`` for absent / invalid fields. Bad values are logged."""
    if not output_options:
        return None, None
    return (
        _parse_non_negative_int(output_options.get("logprobs"), "logprobs"),
        _parse_non_negative_int(
            output_options.get("prompt_logprobs"), "prompt_logprobs"
        ),
    )


def _parse_non_negative_int(value: Any, name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        logger.warning("Invalid %s value: %r (must be integer), ignoring", name, value)
        return None
    if parsed < 0:
        logger.warning(
            "Invalid %s value: %r (must be non-negative), ignoring", name, value
        )
        return None
    return parsed


def extract_from_completion_output(
    output: Any,
    num_output_tokens_so_far: int,
    tokenizer: Any = None,
    *,
    fallback_to_first_on_missing: bool = False,
    include_bytes: bool = False,
    preserve_missing_rows: bool = False,
) -> tuple[Optional[list[float]], Optional[list[list[dict[str, Any]]]]]:
    """Extract logprobs from a vLLM/TRT-LLM-shaped CompletionOutput.

    ``num_output_tokens_so_far`` slices the per-chunk window out of the
    cumulative array. ``tokenizer`` decodes token ids when the engine
    didn't populate ``decoded_token``. ``fallback_to_first_on_missing``
    falls back to the first dict entry when the selected token is
    absent (TRT-LLM corner case). ``include_bytes`` adds a UTF-8 byte
    array per top entry (matches the vLLM/OpenAI shape).
    ``preserve_missing_rows`` emits the vLLM TITO sentinel ``-9999.0``
    and an empty candidate list when a row is absent or does not contain
    the sampled token. This keeps the arrays positionally aligned without
    treating another token's probability as the sampled token's value.

    Returns ``(None, None)`` when nothing was extracted. Unless
    ``preserve_missing_rows`` is enabled, it also returns ``(None, None)``
    when any selected-token logprob is missing in the slice: the Rust
    response builder zips ``log_probs`` / ``top_logprobs`` against
    ``token_ids`` by position, so emitting a shorter array would misalign
    every later token.
    """
    if getattr(output, "logprobs", None) is None:
        return None, None

    token_ids = list(getattr(output, "token_ids", None) or [])
    if not token_ids or num_output_tokens_so_far >= len(token_ids):
        return None, None

    new_logprobs = output.logprobs[num_output_tokens_so_far:]
    new_token_ids = token_ids[num_output_tokens_so_far:]
    new_logprobs = new_logprobs[: len(new_token_ids)]
    if not new_logprobs:
        return None, None

    # TRT-LLM edge case: the engine sometimes returns a flat list of
    # floats instead of a list of `{token_id: Logprob}` dicts. Surface
    # the chosen-token logprobs and bail on top-k.
    if isinstance(new_logprobs[0], float):
        return [float(lp) for lp in new_logprobs], None

    log_probs: list[float] = []
    top_logprobs: list[list[dict[str, Any]]] = []

    for token_idx, token_logprobs_dict in enumerate(new_logprobs):
        if token_logprobs_dict is None:
            if preserve_missing_rows:
                log_probs.append(-9999.0)
                top_logprobs.append([])
                continue
            return None, None

        actual_token_id = new_token_ids[token_idx]
        selected = token_logprobs_dict.get(actual_token_id)
        if selected is None:
            if preserve_missing_rows:
                log_probs.append(-9999.0)
                top_logprobs.append([])
                continue
            if not fallback_to_first_on_missing:
                return None, None
            selected = next(iter(token_logprobs_dict.values()), None)
            if selected is None:
                return None, None
        log_probs.append(float(selected.logprob))

        position_entries: list[dict[str, Any]] = []
        for tok_id, info in token_logprobs_dict.items():
            token_str = getattr(info, "decoded_token", None)
            if not token_str and tokenizer is not None:
                try:
                    token_str = tokenizer.decode([tok_id])
                except Exception:
                    logger.debug(
                        "tokenizer.decode failed for token_id=%s",
                        tok_id,
                        exc_info=True,
                    )
                    token_str = None
            entry: dict[str, Any] = {
                "rank": info.rank if hasattr(info, "rank") else 0,
                "token_id": tok_id,
                "token": token_str,
                "logprob": float(info.logprob),
            }
            if include_bytes:
                entry["bytes"] = list(token_str.encode("utf-8")) if token_str else None
            position_entries.append(entry)
        top_logprobs.append(position_entries)

    return (
        log_probs if log_probs else None,
        top_logprobs if top_logprobs else None,
    )


def extract_prompt_logprobs_from_completion_output(
    output: Any,
    tokenizer: Any = None,
) -> Optional[list[Optional[dict[str, dict[str, Any]]]]]:
    """Extract prompt logprobs from a vLLM/TRT-LLM-shaped output.

    Reads ``output.prompt_logprobs``. Position 0 stays ``None`` (no
    logprob for BOS). Token-id keys are stringified for the JSON
    round-trip into Rust's ``HashMap<u32, PromptLogprobEntry>``.
    Returns ``None`` if the engine didn't compute prompt logprobs.
    """
    prompt_logprobs = getattr(output, "prompt_logprobs", None)
    if prompt_logprobs is None:
        return None

    payload: list[Optional[dict[str, dict[str, Any]]]] = []
    for pos in prompt_logprobs:
        if pos is None:
            payload.append(None)
            continue
        position_map: dict[str, dict[str, Any]] = {}
        for tok_id, info in pos.items():
            decoded_token = getattr(info, "decoded_token", None)
            if not decoded_token and tokenizer is not None:
                try:
                    decoded_token = tokenizer.decode([tok_id])
                except Exception:
                    logger.debug(
                        "tokenizer.decode failed for prompt token_id=%s",
                        tok_id,
                        exc_info=True,
                    )
                    decoded_token = None
            entry: dict[str, Any] = {"logprob": float(info.logprob)}
            rank = getattr(info, "rank", None)
            if rank is not None:
                entry["rank"] = int(rank)
            if decoded_token is not None:
                entry["decoded_token"] = decoded_token
            position_map[str(tok_id)] = entry
        payload.append(position_map)
    return payload


def extract_prompt_logprobs_from_sglang_meta(
    meta: dict[str, Any],
) -> Optional[list[Optional[dict[str, dict[str, Any]]]]]:
    """Extract prompt logprobs from an SGLang ``meta_info`` dict.

    Reads ``input_token_logprobs`` (tuples ``(logprob, token_id, decoded
    or None)``, starting at prompt position 1) and merges any
    ``input_top_logprobs`` alternatives. Prepends ``None`` at index 0
    so the result aligns with Rust's BOS=None ``PromptLogprobs`` shape.
    """
    input_logprobs = meta.get("input_token_logprobs")
    if not input_logprobs:
        return None

    input_top_logprobs = meta.get("input_top_logprobs") or []

    payload: list[Optional[dict[str, dict[str, Any]]]] = [None]
    for idx, item in enumerate(input_logprobs):
        logprob, tok_id, decoded_token = item
        position_map: dict[str, dict[str, Any]] = {}
        selected_entry: dict[str, Any] = {"logprob": float(logprob)}
        if decoded_token is not None:
            selected_entry["decoded_token"] = decoded_token
        position_map[str(tok_id)] = selected_entry

        if idx < len(input_top_logprobs):
            top_at_pos = input_top_logprobs[idx]
            if top_at_pos:
                for top_lp, top_tok_id, top_decoded in top_at_pos:
                    key = str(top_tok_id)
                    if key in position_map:
                        continue
                    alt_entry: dict[str, Any] = {"logprob": float(top_lp)}
                    if top_decoded is not None:
                        alt_entry["decoded_token"] = top_decoded
                    position_map[key] = alt_entry
        payload.append(position_map)
    return payload


_SGLANG_TOP_LOGPROBS_UNSUPPORTED_MSG = (
    "Dynamo's SGLang backend does not currently support logprobs >= 1 due to "
    "an O(N) per-position detokenization in the upstream sglang tokenizer "
    "manager. Use logprobs=0 for chosen-token logprobs, or set "
    "DYN_SGL_ALLOW_TOP_LOGPROBS=1 to override at your own risk. "
    "Track the upstream fix at https://github.com/sgl-project/sglang/pull/24447."
)

DYN_SGL_ALLOW_TOP_LOGPROBS_ENV = "DYN_SGL_ALLOW_TOP_LOGPROBS"


def sglang_top_logprobs_allowed() -> bool:
    """Read the ``DYN_SGL_ALLOW_TOP_LOGPROBS`` env-var gate."""
    return os.environ.get(DYN_SGL_ALLOW_TOP_LOGPROBS_ENV, "").lower() not in (
        "",
        "0",
        "false",
    )


def build_sglang_logprob_kwargs(
    output_options: dict[str, Any],
    *,
    allow_top_logprobs: bool,
) -> dict[str, Any]:
    """Map ``output_options`` to SGLang's
    ``return_logprob`` / ``top_logprobs_num`` / ``logprob_start_len`` kwargs.

    Raises ``ValueError`` for ``logprobs >= 1`` when
    ``allow_top_logprobs`` is ``False``. SGLang's tokenizer manager
    detokenizes top-k tokens serially (O(N) per generated token), so
    enabling it without a batched detokenize path degrades latency
    badly.
    """
    if not output_options:
        return {}

    kwargs: dict[str, Any] = {}

    def _parse(name: str, value: Any) -> Optional[int]:
        parsed = _parse_non_negative_int(value, name)
        if parsed is None:
            return None
        if parsed >= 1 and not allow_top_logprobs:
            raise ValueError(_SGLANG_TOP_LOGPROBS_UNSUPPORTED_MSG)
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
            kwargs["top_logprobs_num"] = max(kwargs.get("top_logprobs_num", 0), parsed)
            # logprob_start_len=0 starts logprob computation at the
            # prompt; SGLang's default (-1) restricts to output tokens.
            kwargs["logprob_start_len"] = 0

    return kwargs


def extract_from_sglang_meta(
    meta_info: dict[str, Any],
    num_output_logprobs_so_far: int,
    *,
    return_tokens_as_token_ids: bool = False,
) -> tuple[Optional[list[float]], Optional[list[list[dict[str, Any]]]], int]:
    """Extract logprobs from SGLang's ``meta_info`` dict.

    SGLang's ``output_token_logprobs`` / ``output_top_logprobs`` are
    cumulative across stream chunks even though ``output_ids`` is
    disjoint — the caller passes the running count to slice the new
    entries, and the returned third element is the updated count.
    """
    output_token_logprobs = meta_info.get("output_token_logprobs")
    if not output_token_logprobs:
        return None, None, num_output_logprobs_so_far

    new_logprobs = output_token_logprobs[num_output_logprobs_so_far:]
    if not new_logprobs:
        return None, None, num_output_logprobs_so_far

    log_probs = [float(entry[0]) for entry in new_logprobs]

    top_logprobs: Optional[list[list[dict[str, Any]]]] = None
    output_top = meta_info.get("output_top_logprobs")
    if output_top:
        new_top = output_top[num_output_logprobs_so_far:]
        if new_top:
            top_logprobs = []
            for position_entries in new_top:
                if position_entries is None:
                    top_logprobs.append([])
                    continue
                position_list: list[dict[str, Any]] = []
                for rank_idx, entry in enumerate(position_entries):
                    tok_id = entry[1]
                    token_str = (
                        f"token_id:{tok_id}" if return_tokens_as_token_ids else entry[2]
                    )
                    position_list.append(
                        {
                            "rank": rank_idx + 1,
                            "token_id": tok_id,
                            "token": token_str,
                            "logprob": float(entry[0]),
                        }
                    )
                top_logprobs.append(position_list)

    return log_probs, top_logprobs, len(output_token_logprobs)
