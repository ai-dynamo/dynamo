# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utilities for transforming TRT-LLM outputs to Dynamo response format.
"""

from typing import Any


class ResponseUtils:
    """
    Utility functions for transforming TRT-LLM outputs to Dynamo response format.
    """

    @staticmethod
    def extract_logprobs(
        output: Any, num_output_tokens_so_far: int
    ) -> tuple[list[float] | None, list[list[dict]] | None]:
        """
        Extract logprobs from the TRTLLM output for new tokens.

        Args:
            output: TRTLLM CompletionOutput object
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
        # Expect TokenLogprobs output when logprobs is set, check edge case where list[float] is returned instead
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
