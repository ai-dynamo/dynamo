# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared logprobs extraction utility for Dynamo backends.

This module provides a unified function for extracting log probabilities from
LLM engine outputs. It handles the common logprobs format used across vLLM and
TensorRT-LLM backends.
"""

from typing import Optional


def extract_logprobs(
    logprobs: Optional[list],
    token_ids: list,
    num_output_tokens_so_far: int,
) -> tuple[Optional[list[float]], Optional[list[list[dict]]]]:
    """Extract logprobs from engine output for new tokens.

    This is the unified extraction logic used by both vLLM and TRT-LLM backends.
    It handles the common TokenLogprobs dict format as well as the TRT-LLM edge
    case where logprobs may be a plain list of floats.

    Args:
        logprobs: List of token logprobs dicts (or list of floats for TRT-LLM edge case).
                  Each dict maps token_id -> logprob_info with .logprob, .rank, .decoded_token.
        token_ids: List of generated token IDs for the full output so far.
        num_output_tokens_so_far: Number of tokens already processed (for delta extraction).

    Returns:
        Tuple of (log_probs, top_logprobs) in Dynamo's expected format:
        - log_probs: List of log probabilities for each new token, or None
        - top_logprobs: List of top logprobs dicts for each new token, or None
    """
    if logprobs is None:
        return None, None

    # Get logprobs for new tokens only
    new_logprobs = logprobs[num_output_tokens_so_far:]
    if not new_logprobs:
        return None, None

    # Handle TRT-LLM edge case where logprobs is List[float] instead of List[dict]
    if isinstance(new_logprobs[0], float):
        return [float(lp) for lp in new_logprobs], None

    log_probs = []
    top_logprobs = []

    for token_idx, token_logprobs_dict in enumerate(new_logprobs):
        if token_logprobs_dict is None:
            continue

        # Get the actual token_id that was generated at this position
        actual_token_id = token_ids[num_output_tokens_so_far + token_idx]

        # Extract log probability for the selected token
        if actual_token_id in token_logprobs_dict:
            selected_logprob = token_logprobs_dict[actual_token_id]
            log_probs.append(float(selected_logprob.logprob))
        else:
            # Fallback: use the first logprob if selected token not found
            first_logprob = next(iter(token_logprobs_dict.values()), None)
            if first_logprob:
                log_probs.append(float(first_logprob.logprob))

        # Build top_logprobs list for this token position
        token_top_logprobs = []
        for tok_id, logprob_info in token_logprobs_dict.items():
            token_top_logprobs.append(
                {
                    "rank": (
                        logprob_info.rank if hasattr(logprob_info, "rank") else 0
                    ),
                    "token_id": tok_id,
                    "token": (
                        logprob_info.decoded_token
                        if hasattr(logprob_info, "decoded_token")
                        else None
                    ),
                    "logprob": float(logprob_info.logprob),
                }
            )
        top_logprobs.append(token_top_logprobs)

    return log_probs if log_probs else None, top_logprobs if top_logprobs else None
