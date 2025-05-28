# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
Decoder Component with Universal X-Request-Id Support

This example shows how decoder components can use Dynamo SDK's built-in
request tracing for automatic request ID propagation and decode operations.
"""

import asyncio
import logging
from typing import AsyncIterator, Dict, List, Optional

from dynamo.sdk import RequestTracingMixin, endpoint, get_current_request_id, service

logger = logging.getLogger(__name__)


@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    resources={"cpu": "4", "memory": "8Gi", "gpu": "1"},
)
class Decoder(RequestTracingMixin):
    """
    Decoder component with automatic X-Request-Id support.

    Benefits of using RequestTracingMixin:
    - ensure_request_id(): Automatic request ID management
    - log_with_request_id(): Consistent logging with request ID
    - get_current_request_id(): Access request ID anywhere in call stack
    """

    def __init__(self):
        self.vocab_size = 32000
        self.active_sequences = {}
        self.decode_stats = {"tokens_generated": 0, "sequences_completed": 0}

    @endpoint()
    async def decode(
        self, hidden_states: List[float], request_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, any]]:
        """
        Decode hidden states to tokens with automatic request ID tracking.

        The RequestTracingMixin automatically handles request ID management.
        """
        request_id = self.ensure_request_id(request_id)

        self.log_with_request_id("info", "Starting token decoding")

        try:
            sequence_id = f"seq_{request_id}"
            self.active_sequences[sequence_id] = {
                "request_id": request_id,
                "tokens_generated": 0,
                "start_time": asyncio.get_event_loop().time(),
            }

            async for token_result in self._decode_tokens(hidden_states, sequence_id):
                yield token_result

        except Exception as e:
            self.log_with_request_id("error", f"Decoding failed: {e}")
            raise
        finally:
            if sequence_id in self.active_sequences:
                sequence_info = self.active_sequences.pop(sequence_id)
                self.decode_stats["sequences_completed"] += 1
                self.log_with_request_id(
                    "info",
                    f"Decoding completed. Generated {sequence_info['tokens_generated']} tokens",
                )

    async def _decode_tokens(
        self, hidden_states: List[float], sequence_id: str
    ) -> AsyncIterator[Dict[str, any]]:
        """
        Internal token decoding method that can access request ID from context.
        """
        current_request_id = get_current_request_id()
        if current_request_id:
            logger.debug(f"Decoding tokens for request: {current_request_id}")

        max_tokens = 10

        for step in range(max_tokens):
            self.log_with_request_id("debug", f"Decoding step {step + 1}/{max_tokens}")

            await asyncio.sleep(0.1)

            token_id = (hash(str(hidden_states)) + step) % self.vocab_size
            token_text = self._token_id_to_text(token_id)

            self.active_sequences[sequence_id]["tokens_generated"] += 1
            self.decode_stats["tokens_generated"] += 1

            is_eos = token_text in [".", "!", "?"] or step == max_tokens - 1

            token_result = {
                "token_id": token_id,
                "token_text": token_text,
                "step": step,
                "is_eos": is_eos,
                "logprobs": self._calculate_logprobs(token_id),
                "sequence_id": sequence_id,
            }

            self.log_with_request_id(
                "debug", f"Generated token: '{token_text}' (id: {token_id})"
            )

            yield token_result

            if is_eos:
                self.log_with_request_id("debug", "End of sequence reached")
                break

    def _token_id_to_text(self, token_id: int) -> str:
        """
        Convert token ID to text (simplified simulation).
        """
        special_tokens = {
            0: "<eos>",
            1: "<pad>",
            2: "<unk>",
        }

        if token_id in special_tokens:
            return special_tokens[token_id]

        words = [
            "hello",
            "world",
            "the",
            "quick",
            "brown",
            "fox",
            "jumps",
            "over",
            "lazy",
            "dog",
            ".",
            "!",
            "?",
        ]
        return words[token_id % len(words)]

    def _calculate_logprobs(self, token_id: int) -> Dict[str, float]:
        """
        Calculate log probabilities for the token (simplified simulation).
        """
        import random

        logprobs = {}
        base_prob = -random.uniform(0.1, 2.0)

        for i in range(5):
            prob = base_prob - (i * 0.5) - random.uniform(0, 0.3)
            token_text = self._token_id_to_text((token_id + i) % self.vocab_size)
            logprobs[token_text] = prob

        return logprobs

    @endpoint()
    async def batch_decode(
        self, batch_hidden_states: List[List[float]], request_id: Optional[str] = None
    ) -> AsyncIterator[Dict[str, any]]:
        """
        Batch decode multiple sequences with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id(
            "info", f"Starting batch decode for {len(batch_hidden_states)} sequences"
        )

        for batch_idx, hidden_states in enumerate(batch_hidden_states):
            self.log_with_request_id(
                "debug",
                f"Processing batch item {batch_idx + 1}/{len(batch_hidden_states)}",
            )

            sequence_id = f"batch_{request_id}_{batch_idx}"
            self.active_sequences[sequence_id] = {
                "request_id": request_id,
                "batch_idx": batch_idx,
                "tokens_generated": 0,
                "start_time": asyncio.get_event_loop().time(),
            }

            async for token_result in self._decode_tokens(hidden_states, sequence_id):
                token_result["batch_idx"] = batch_idx
                yield token_result

    @endpoint()
    async def get_stats(self, request_id: Optional[str] = None) -> Dict[str, any]:
        """
        Get decoder statistics with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("debug", "Retrieving decoder statistics")

        return {
            "decode_stats": self.decode_stats,
            "active_sequences": len(self.active_sequences),
            "vocab_size": self.vocab_size,
            "avg_tokens_per_sequence": (
                self.decode_stats["tokens_generated"]
                / max(1, self.decode_stats["sequences_completed"])
            ),
        }

    @endpoint()
    async def stop_sequence(
        self, sequence_id: str, request_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Stop a specific decoding sequence with request tracking.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", f"Stopping sequence: {sequence_id}")

        if sequence_id in self.active_sequences:
            sequence_info = self.active_sequences.pop(sequence_id)
            self.log_with_request_id(
                "debug",
                f"Stopped sequence with {sequence_info['tokens_generated']} tokens",
            )
            return {"status": "stopped", "sequence_info": sequence_info}
        else:
            self.log_with_request_id("warning", f"Sequence {sequence_id} not found")
            return {"status": "not_found"}
