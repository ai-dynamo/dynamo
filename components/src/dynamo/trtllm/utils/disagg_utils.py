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

import base64
import dataclasses

from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi import DisaggregatedParams


class DisaggregatedParamsCodec:
    """
    Codec for encoding and decoding disaggregated params for network transfer.
    """

    @staticmethod
    def serialize_first_gen_log_probs(params_dict: dict) -> None:
        """Convert first_gen_log_probs from TRT-LLM's internal format to a
        JSON-safe transport format.

        TRT-LLM stores logprobs as ``[{token_id(int): Logprob, ...}, ...]``
        where dict keys are integer token IDs. The Rust transport layer
        (pythonize 0.23 → serde_json::Value) requires string map keys, so
        we flatten to a list-of-lists format matching TRT-LLM's own
        ``_serialize_first_gen_log_probs`` in ``openai_protocol.py``::

            Input:  [{4710: Logprob(-2.32, rank=1), 6771: Logprob(-2.51, rank=2)}]
            Output: [[{"token_id": 4710, "logprob": -2.32, "rank": 1},
                       {"token_id": 6771, "logprob": -2.51, "rank": 2}]]
        """
        fglp = params_dict.get("first_gen_log_probs")
        if not fglp:
            return
        params_dict["first_gen_log_probs"] = [
            [
                {"token_id": tid, "logprob": lp["logprob"], "rank": lp.get("rank")}
                for tid, lp in pos.items()
            ]
            if isinstance(pos, dict)
            else pos
            for pos in fglp
        ]

    @staticmethod
    def deserialize_first_gen_log_probs(params_dict: dict) -> None:
        """Reconstruct first_gen_log_probs from the JSON-safe transport format
        back to TRT-LLM's internal ``{token_id(int): Logprob}`` dict format.

        TRT-LLM's ``py_executor.py`` calls ``append_log_probs`` which accesses
        the ``.logprob`` attribute on the dict values, so we must rebuild
        ``Logprob`` dataclass instances.
        """
        fglp = params_dict.get("first_gen_log_probs")
        if not fglp:
            return
        params_dict["first_gen_log_probs"] = [
            {
                item["token_id"]: Logprob(
                    logprob=item["logprob"], rank=item.get("rank")
                )
                for item in pos
            }
            if isinstance(pos, list)
            else pos
            for pos in fglp
        ]

    @staticmethod
    def decode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        opaque_state = disaggregated_params.opaque_state
        if isinstance(opaque_state, str):
            opaque_state = base64.b64decode(opaque_state)
        return dataclasses.replace(disaggregated_params, opaque_state=opaque_state)

    @staticmethod
    def encode(
        disaggregated_params: DisaggregatedParams,
    ) -> DisaggregatedParams:
        if disaggregated_params is None:
            return None

        opaque_state = disaggregated_params.opaque_state
        if isinstance(opaque_state, (bytes, bytearray)):
            opaque_state = base64.b64encode(opaque_state).decode("utf-8")
        return dataclasses.replace(disaggregated_params, opaque_state=opaque_state)
