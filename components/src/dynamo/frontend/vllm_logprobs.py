# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt Dynamo's normalized worker probabilities to vLLM's output processor.

Keep incremental detokenization and per-choice probability state in vLLM. The
wire reader validates positional alignment; it never truncates a mismatched zip.
Public projection follows the pinned native chat server, not completion maps.
"""

import math
from typing import Any

import numpy as np
from vllm.v1.outputs import LogprobsLists


def wants_sample_logprobs(request: dict[str, Any]) -> bool:
    return request.get("logprobs") is True and (
        request.get("top_logprobs", 0) is not None
        or bool(request.get("logprob_token_ids"))
    )


def worker_sample_logprobs(response: dict[str, Any]) -> LogprobsLists | None:
    """Reconstruct sampled-first rows without assuming all rows have equal width.

    The worker map has already deduplicated the sampled token from top-k. vLLM
    accepts shorter rows; do not pad them with invented probabilities. Ranks of
    alternatives are recomputed internally but are not part of chat's public
    logprob response. The sampled rank and insertion order are preserved.
    """
    token_ids = response["token_ids"]
    if not token_ids:
        return None
    selected = response.get("log_probs")
    alternatives = response.get("top_logprobs")
    if not isinstance(selected, list) or not isinstance(alternatives, list):
        raise TypeError("Missing generated logprobs in worker response")
    if len(selected) != len(token_ids) or len(alternatives) != len(token_ids):
        raise ValueError("Misaligned generated logprobs in worker response")

    ids_rows = np.empty(len(token_ids), dtype=object)
    probability_rows = np.empty(len(token_ids), dtype=object)
    ranks = np.empty(len(token_ids), dtype=np.int64)
    for position, (sampled_id, entries) in enumerate(zip(token_ids, alternatives)):
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                "Missing generated logprob alternatives in worker response"
            )
        by_id = {}
        for entry in entries:
            if not isinstance(entry, dict):
                raise TypeError("Invalid generated logprob entry in worker response")
            token_id = entry.get("token_id")
            probability = entry.get("logprob")
            if (
                type(token_id) is not int
                or not 0 <= token_id <= 0xFFFFFFFF
                or token_id in by_id
                or type(probability) not in (int, float)
                or math.isnan(probability)
                or probability == math.inf
            ):
                raise ValueError("Invalid generated logprob entry in worker response")
            by_id[token_id] = entry
        sampled = by_id.get(sampled_id)
        if (
            sampled is None
            or type(sampled.get("rank")) is not int
            or not 1 <= sampled["rank"] <= 0x7FFFFFFFFFFFFFFF
        ):
            raise ValueError("Missing sampled token rank in worker response")
        if (
            type(selected[position]) not in (int, float)
            or selected[position] != sampled["logprob"]
        ):
            raise ValueError("Conflicting sampled logprob in worker response")
        # The engine's sampled token is always first; keep the remaining worker
        # order rather than sorting by rank (explicit token selections differ).
        ordered = [sampled] + [
            entry for key, entry in by_id.items() if key != sampled_id
        ]
        ids_rows[position] = np.asarray(
            [entry["token_id"] for entry in ordered], dtype=np.int64
        )
        probability_rows[position] = np.asarray(
            [entry["logprob"] for entry in ordered], dtype=np.float64
        )
        ranks[position] = sampled["rank"]
    return LogprobsLists(ids_rows, probability_rows, ranks)


def chat_logprob_content(
    output: Any, request: dict[str, Any], tokenizer: Any
) -> list[dict[str, Any]]:
    """Project vLLM's detokenized per-token dictionaries into native chat fields."""
    token_ids = output.token_ids
    positions = output.logprobs
    if positions is None or len(positions) != len(token_ids):
        raise ValueError("Misaligned generated logprobs from vLLM output processor")
    return_ids = bool(request.get("return_tokens_as_token_ids"))
    count = request.get("top_logprobs", 0)
    return_all = bool(request.get("logprob_token_ids")) or count == -1

    def text_for(token_id, entry):
        if return_ids:
            return f"token_id:{token_id}"
        return (
            entry.decoded_token
            if entry.decoded_token is not None
            else tokenizer.decode(token_id)
        )

    content = []
    for token_id, entries in zip(token_ids, positions):
        if entries is None or token_id not in entries:
            raise ValueError("Missing sampled logprob from vLLM output processor")
        sampled = entries[token_id]
        alternatives = []
        for index, (candidate_id, entry) in enumerate(entries.items()):
            if not return_all and (count is None or index >= count):
                break
            token = text_for(candidate_id, entry)
            alternatives.append(
                {
                    "token": token,
                    "logprob": max(entry.logprob, -9999.0),
                    "bytes": list(token.encode("utf-8", errors="replace")),
                }
            )
        content.append(
            {
                "token": text_for(token_id, sampled),
                "logprob": max(sampled.logprob, -9999.0),
                "bytes": None
                if sampled.decoded_token is None
                else list(sampled.decoded_token.encode("utf-8", errors="replace")),
                "top_logprobs": alternatives,
            }
        )
    return content
