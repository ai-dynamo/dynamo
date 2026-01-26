# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pure utility functions for request handling.

These functions are stateless and have no external dependencies,
making them easy to unit test independently.
"""

from typing import Any


class RequestUtils:
    """
    Utility class for request normalization and processing.

    All methods are static and pure (no side effects beyond input modification).
    """

    @staticmethod
    def normalize_request_format(request: dict) -> None:
        """
        Convert OpenAI request format to TRT-LLM internal format.

        Moves fields from OpenAI locations to where TRT-LLM expects them:
        - max_tokens: top-level -> stop_conditions.max_tokens
        - temperature: top-level -> sampling_options.temperature

        Note: The Rust frontend's PrefillRouter handles the *value* of max_tokens
        (sets to 1 for prefill, restores original for decode). This method only
        moves fields to the correct location.

        Args:
            request: Request dictionary to normalize (modified in place)
        """
        # Ensure stop_conditions exists
        if "stop_conditions" not in request:
            request["stop_conditions"] = {}
        if "max_tokens" in request and "max_tokens" not in request["stop_conditions"]:
            request["stop_conditions"]["max_tokens"] = request.pop("max_tokens")

        # Ensure sampling_options exists
        if "sampling_options" not in request:
            request["sampling_options"] = {}
        if (
            "temperature" in request
            and "temperature" not in request["sampling_options"]
        ):
            request["sampling_options"]["temperature"] = request.pop("temperature")

    @staticmethod
    def extract_logprobs(
        output: Any, num_output_tokens_so_far: int
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from the TRTLLM output for new tokens.

        Args:
            output: TRTLLM CompletionOutput object with logprobs and token_ids attributes
            num_output_tokens_so_far: Number of tokens already processed

        Returns:
            Tuple of (log_probs, top_logprobs) in Dynamo's expected format:
            - log_probs: List of log probabilities for each new token
            - top_logprobs: List of top logprobs dicts for each new token
        """
        if output.logprobs is None:
            return None, None

        # Get logprobs for new tokens only
        new_logprobs = output.logprobs[num_output_tokens_so_far:]
        if not new_logprobs:
            return None, None

        # From TRTLLM CompletionOutput API, logprobs: (TokenLogprobs | List[float], optional)
        # Expect TokenLogprobs output when logprobs is set, check edge case where list[float] is returned
        if isinstance(new_logprobs[0], float):
            return [float(lp) for lp in new_logprobs], None

        log_probs = []
        top_logprobs = []

        for token_idx, token_logprobs_dict in enumerate(new_logprobs):
            if token_logprobs_dict is None:
                continue

            # Get the actual token_id that was generated at this position
            actual_token_id = output.token_ids[num_output_tokens_so_far + token_idx]

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
            # NOTE: TRTLLM LogProb API doesn't have decoded_token, will default to None
            token_top_logprobs = []
            for tok_id, logprob_info in token_logprobs_dict.items():
                token_top_logprobs.append(
                    {
                        "rank": logprob_info.rank
                        if hasattr(logprob_info, "rank")
                        else 0,
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
