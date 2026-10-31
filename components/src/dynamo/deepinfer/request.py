# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo request -> DeepInfer ``SamplingParams`` kwargs, and DeepInfer step
outputs (``RequestOutput`` pointers) -> :class:`Output`.

Imported by rank worker processes: no ``dynamo._core``, torch or DeepInfer.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from typing import Any, Optional, Union

import numpy as np

from dynamo.common.utils.token_ids import token_ids_len
from dynamo.deepinfer.protocol import Output

# Finish reasons whose output carries a newly generated token. Ignored or
# terminated sequences re-emit their last existing token, which must be dropped.
_TOKEN_REASONS = frozenset({"", "stop", "length"})


def map_finish_reason(reason: str) -> str:
    if reason in ("stop", "length"):
        return reason
    if reason == "abort":
        return "cancelled"
    # prompt_length_exceeded, insufficient_system_resource, xgrammar/ve errors
    return "error"


def build_sampling_params(
    request: Mapping[str, Any], max_model_len: int
) -> tuple[dict[str, Any], Optional[int]]:
    """Return ``(SamplingParams kwargs, logprobs)``; ``logprobs`` is None when not
    requested, else the number of top alternatives (0 = chosen token only).

    Raises ``ValueError`` for requests DeepInfer cannot serve. Stop strings and
    stop token ids are enforced upstream by Dynamo, which aborts the request.
    """
    sampling = request.get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    output = request.get("output_options") or {}

    n = sampling.get("n")
    if n is not None and int(n) > 1:
        raise ValueError(f"DeepInfer backend supports n=1 only, got n={n}")
    unsupported = _unsupported_fields(sampling, stop, output)
    if unsupported:
        raise ValueError(f"DeepInfer backend does not support {', '.join(unsupported)}")
    prompt_len = token_ids_len(request.get("token_ids"))
    if prompt_len == 0:
        raise ValueError("prompt must not be empty")
    if prompt_len >= max_model_len:
        raise ValueError(
            f"prompt has {prompt_len} tokens; max_model_len is {max_model_len}"
        )

    params: dict[str, Any] = {}
    for key in ("temperature", "top_p"):
        if sampling.get(key) is not None:
            params[key] = float(sampling[key])
    top_k = sampling.get("top_k")
    if top_k is not None and int(top_k) > 0:
        params["top_k"] = int(top_k)
    if sampling.get("seed") is not None:
        params["seq_seed"] = int(sampling["seed"])

    max_tokens = stop.get("max_tokens")
    # DeepInfer defaults to 16, so always pass an explicit limit.
    params["max_tokens"] = (
        int(max_tokens) if max_tokens is not None else max_model_len - prompt_len
    )
    if stop.get("ignore_eos") is not None:
        params["ignore_eos"] = bool(stop["ignore_eos"])

    logprobs = output.get("logprobs")
    logprobs = None if logprobs is None else max(0, int(logprobs))
    if logprobs:
        params["top_logprobs"] = logprobs
    return params, logprobs


# Constraint fields of Dynamo's GuidedDecodingOptions (backend and
# whitespace_pattern only modify a constraint).
_GUIDED_CONSTRAINTS = ("json", "regex", "choice", "grammar", "structural_tag")


def _unsupported_fields(
    sampling: Mapping[str, Any], stop: Mapping[str, Any], output: Mapping[str, Any]
) -> list[str]:
    """Recognized request fields DeepInfer cannot honor; they must not be
    silently dropped. Default values pass."""
    found = []
    guided = sampling.get("guided_decoding") or {}
    if any(guided.get(key) is not None for key in _GUIDED_CONSTRAINTS):
        found.append("guided_decoding")
    for key in ("presence_penalty", "frequency_penalty"):
        if sampling.get(key) not in (None, 0):
            found.append(key)
    if sampling.get("repetition_penalty") not in (None, 1):
        found.append("repetition_penalty")
    if sampling.get("min_p") not in (None, 0):
        found.append("min_p")
    if sampling.get("use_beam_search"):
        found.append("use_beam_search")
    if sampling.get("best_of") not in (None, 1):
        found.append("best_of")
    # DeepInfer stops on EOS by itself; with ignore_eos there is nothing to defer.
    if (stop.get("min_tokens") or 0) > 0 and not stop.get("ignore_eos"):
        found.append("min_tokens")
    if stop.get("max_thinking_tokens") is not None:
        found.append("max_thinking_tokens")
    if output.get("prompt_logprobs") is not None:
        found.append("prompt_logprobs")
    return found


def prompt_bytes(token_ids: Union[list[int], bytes, bytearray, memoryview]) -> bytes:
    """The prompt as native int32 bytes for the ranks. A packed buffer from the
    request plane (``DYN_TOKEN_IDS_AS_BYTES``, little-endian int32) passes through."""
    if isinstance(token_ids, (bytes, bytearray, memoryview)):
        data = bytes(token_ids)
        if len(data) % 4:
            raise ValueError(
                f"packed token_ids byte length {len(data)} is not a multiple of 4"
            )
        if sys.byteorder != "little":
            data = np.frombuffer(data, dtype="<i4").astype(np.int32).tobytes()
        return data
    return np.asarray(token_ids, dtype=np.int32).tobytes()


def prompt_array(data: bytes) -> np.ndarray:
    # DeepInfer's Sequence takes a writable, contiguous int32 array.
    return np.frombuffer(data, dtype=np.int32).copy()


def collect_outputs(
    ptrs: Any,
    ro: Any,
    live: Mapping[str, int],
    into: dict[str, Output],
) -> None:
    """Copy one step's ``RequestOutput`` pointers into ``into`` (keyed by request).

    ``ro`` is DeepInfer's ``RequestOutput`` class (static getters); ``live`` maps
    the rank's in-flight request ids to their ``top_logprobs``. Pointers of
    unknown requests (placeholders, already-closed streams) are skipped. The
    pointers are freed by the next ``step()``, so nothing here keeps them.
    """
    for ptr in np.asarray(ptrs).tolist():
        rid = ro.get_request_id(ptr)
        top_k = live.get(rid)
        if top_k is None:
            continue
        out = into.get(rid)
        if out is None:
            out = into[rid] = Output(request_id=rid)
        reason = ro.get_finish_reason(ptr)
        if reason in _TOKEN_REASONS:
            out.token_ids.append(int(ro.get_token_id(ptr)))
            out.logprobs.append(float(ro.get_token_logprobs(ptr)))
            if top_k > 0:
                out.top_token_ids.append(list(ro.get_top_token_ids(ptr))[:top_k])
                out.top_logprobs.append(list(ro.get_top_logprobs(ptr))[:top_k])
        if ro.get_finished(ptr):
            out.finish_reason = map_finish_reason(reason)
