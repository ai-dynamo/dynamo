# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate a Dynamo GenerateRequest into DeepInfer SamplingParams kwargs.

The translation is explicit: every field Dynamo can send is either mapped,
neutral (ignored because it equals the DeepInfer behaviour), or rejected with
``InvalidArgument`` so a request is never silently served with different
semantics. The function returns plain kwargs so it does not import DeepInfer
(which pulls in torch and initialises the device).
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any, Optional

from dynamo.common.backend.engine import GenerateRequest
from dynamo.common.backend.logprobs import parse_logprob_options
from dynamo.common.utils.token_ids import token_ids_len
from dynamo.llm.exceptions import InvalidArgument

MAX_TOP_LOGPROBS = 1000

# sampling_options keys that DeepInfer has no equivalent for, mapped to the
# value that means "not requested".
_UNSUPPORTED_SAMPLING: dict[str, Collection[Any]] = {
    "frequency_penalty": (None, 0, 0.0),
    "presence_penalty": (None, 0, 0.0),
    "repetition_penalty": (None, 1, 1.0),
    "min_p": (None, 0, 0.0),
    "logit_bias": (None, {}),
    "n": (None, 1),
    "guided_decoding": (None, {}),
}


def _reject_unsupported(options: dict[str, Any], stop: dict[str, Any]) -> None:
    for key, neutral in _UNSUPPORTED_SAMPLING.items():
        value = options.get(key)
        if value not in neutral:
            raise InvalidArgument(f"DeepInfer does not support {key}={value!r}")
    min_tokens = stop.get("min_tokens")
    if min_tokens not in (None, 0):
        raise InvalidArgument(f"DeepInfer does not support min_tokens={min_tokens!r}")


def build_sampling_kwargs(
    request: GenerateRequest,
    *,
    model_max_len: Optional[int] = None,
) -> dict[str, Any]:
    options = request.get("sampling_options") or {}
    stop = request.get("stop_conditions") or {}
    output = request.get("output_options") or {}

    _reject_unsupported(options, stop)

    kwargs: dict[str, Any] = {}
    for key in ("temperature", "top_p"):
        if options.get(key) is not None:
            kwargs[key] = options[key]
    top_k = options.get("top_k")
    # Dynamo clients use 0 or -1 for "disabled"; DeepInfer only accepts -1.
    kwargs["top_k"] = top_k if top_k is not None and top_k > 0 else -1
    if options.get("seed") is not None:
        kwargs["seq_seed"] = options["seed"]

    max_tokens = stop.get("max_tokens")
    if max_tokens is None and model_max_len is not None:
        max_tokens = model_max_len - token_ids_len(request.get("token_ids"))
    if max_tokens is not None:
        kwargs["max_tokens"] = max(1, max_tokens)
    if stop.get("ignore_eos") is not None:
        kwargs["ignore_eos"] = stop["ignore_eos"]

    logprobs, prompt_logprobs = parse_logprob_options(output)
    if prompt_logprobs:
        raise InvalidArgument("DeepInfer does not support prompt_logprobs")
    if logprobs:
        if logprobs > MAX_TOP_LOGPROBS:
            raise InvalidArgument(f"logprobs must be <= {MAX_TOP_LOGPROBS}")
        kwargs["top_logprobs"] = logprobs
    return kwargs


def validate_prompt(
    token_ids: Sequence[Any],
    *,
    vocab_size: Optional[int],
    max_model_len: Optional[int],
) -> None:
    """Reject a prompt DeepInfer would drop or ignore, with a precise message."""
    if not token_ids:
        raise InvalidArgument("token_ids must not be empty")
    if any(type(token) is not int for token in token_ids):
        raise InvalidArgument("token_ids must be integers")
    low, high = min(token_ids), max(token_ids)
    if low < 0 or (vocab_size is not None and high >= vocab_size):
        raise InvalidArgument(
            f"token ids must be in [0, {vocab_size}); got min={low}, max={high}"
        )
    if max_model_len is not None and len(token_ids) > max_model_len:
        raise InvalidArgument(
            f"prompt of {len(token_ids)} tokens exceeds the model context "
            f"length ({max_model_len})"
        )
