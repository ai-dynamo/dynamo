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
import inspect

from tensorrt_llm.executor.result import Logprob
from tensorrt_llm.llmapi import DisaggregatedParams
from tensorrt_llm.llmapi.disagg_utils import (
    get_global_disagg_request_id as _trtllm_get_global_disagg_request_id,
)

# This compatibility shim intentionally relies on the public helper retaining
# its name while TRT-LLM distinguishes the two known APIs by the ``process_id``
# parameter added in rc22.
# Dynamo maps its distributed-runtime connection ID into TRT-LLM's historical
# 10-bit machine-ID space with ``connection_id % 1021``. TRT-LLM rc22 split
# that field into an 8-bit node ID and a 6-bit process ID. Preserve Dynamo's
# existing 10-bit worker slot and encode it losslessly into the new pair.
_TRTLLM_DISAGG_ID_HAS_PROCESS_ID = (
    "process_id" in inspect.signature(_trtllm_get_global_disagg_request_id).parameters
)
_TRTLLM_PROCESS_ID_SPACE = 1 << 6
_TRTLLM_NODE_ID_SPACE = 1 << 8
_DYNAMO_DISAGG_MACHINE_ID_SPACE = 1021


def get_compatible_global_disagg_request_id(machine_id: int) -> int:
    """Generate a global TRT-LLM disaggregation request ID across API versions.

    TRT-LLM <= rc21 accepts a 10-bit ``machine_id``. TRT-LLM >= rc22 accepts
    an 8-bit ``node_id`` plus a 6-bit ``process_id``. Dynamo's existing
    machine ID is in ``[0, 1021)``; splitting that value with ``divmod(64)``
    produces a unique, in-range pair without changing Dynamo's collision
    characteristics.
    """

    if not 0 <= machine_id < _DYNAMO_DISAGG_MACHINE_ID_SPACE:
        raise ValueError(
            "Dynamo disagg machine_id must be in range "
            f"[0, {_DYNAMO_DISAGG_MACHINE_ID_SPACE}), got {machine_id}"
        )

    if _TRTLLM_DISAGG_ID_HAS_PROCESS_ID:
        node_id, process_id = divmod(machine_id, _TRTLLM_PROCESS_ID_SPACE)
        if node_id >= _TRTLLM_NODE_ID_SPACE:
            raise ValueError(
                "Dynamo disagg machine_id maps outside TRT-LLM's 8-bit "
                f"node_id space: machine_id={machine_id}, node_id={node_id}"
            )
        return _trtllm_get_global_disagg_request_id(node_id, process_id)

    return _trtllm_get_global_disagg_request_id(machine_id)


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
