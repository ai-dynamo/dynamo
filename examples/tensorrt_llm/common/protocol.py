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
import base64
from typing import List, Optional

from pydantic import BaseModel, Field
from tensorrt_llm.llmapi import DisaggregatedParams as LlmDisaggregatedParams
from tensorrt_llm.serve.openai_protocol import DisaggregatedParams


class TokenRangeRetentionConfig(BaseModel):
    token_start: int = Field(..., description="Start token index for retention")
    token_end: Optional[int] = Field(
        None, description="End token index for retention (exclusive)"
    )
    priority: int = Field(..., description="Priority level for retention")
    duration_ms: Optional[int] = Field(
        None, description="Duration in milliseconds for retention"
    )


class KvCacheRetentionConfig(BaseModel):
    token_range_retention_configs: Optional[List[TokenRangeRetentionConfig]] = Field(
        None, description="List of token range retention configurations"
    )
    decode_retention_priority: Optional[int] = Field(
        None, description="Priority for decode retention"
    )
    decode_duration_ms: Optional[int] = Field(
        None, description="Duration in milliseconds for decode retention"
    )
    transfer_mode: Optional[str] = Field(
        None, description="Transfer mode: 'DRAM', 'GDS', or 'POSIX_DEBUG_FALLBACK'"
    )
    directory: Optional[str] = Field(None, description="Directory for KV cache storage")


class Tokens(BaseModel):
    tokens: list[int]


TokenIdType = int


class DisaggregatedTypeConverter:
    @staticmethod
    def to_llm_disaggregated_params(
        disaggregated_params: DisaggregatedParams,
    ) -> LlmDisaggregatedParams:
        if disaggregated_params is None:
            return None
        else:
            opaque_state = (
                base64.b64decode(disaggregated_params.encoded_opaque_state)
                if disaggregated_params.encoded_opaque_state is not None
                else None
            )

            return LlmDisaggregatedParams(
                request_type=disaggregated_params.request_type,
                first_gen_tokens=disaggregated_params.first_gen_tokens,
                ctx_request_id=disaggregated_params.ctx_request_id,
                opaque_state=opaque_state,
            )

    @staticmethod
    def to_oai_disaggregated_params(
        tllm_disagg_params: LlmDisaggregatedParams,
    ) -> DisaggregatedParams:
        if tllm_disagg_params is None:
            return None
        else:
            encoded_opaque_state = (
                base64.b64encode(tllm_disagg_params.opaque_state).decode("utf-8")
                if tllm_disagg_params.opaque_state is not None
                else None
            )
            return DisaggregatedParams(
                request_type=tllm_disagg_params.request_type,
                first_gen_tokens=tllm_disagg_params.first_gen_tokens,
                ctx_request_id=tllm_disagg_params.ctx_request_id,
                encoded_opaque_state=encoded_opaque_state,
            )


# TODO: move these to common for all LLMs once we adopt dynamo-run
# derived from lib/llm/src/protocols/common/preprocessor.rs
class StopConditions(BaseModel):
    max_tokens: Optional[int] = None
    stop: Optional[List[str]] = None
    stop_token_ids_hidden: Optional[List[TokenIdType]] = None
    min_tokens: Optional[int] = None
    ignore_eos: Optional[bool] = None


class SamplingOptions(BaseModel):
    n: Optional[int] = None
    best_of: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    repetition_penalty: Optional[float] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    use_beam_search: Optional[bool] = None
    length_penalty: Optional[float] = None
    seed: Optional[int] = None


class TRTLLMWorkerRequest(BaseModel):
    token_ids: List[TokenIdType]
    stop_conditions: StopConditions
    sampling_options: SamplingOptions
    eos_token_ids: List[TokenIdType] = Field(default_factory=list)
    mdc_sum: Optional[str] = None
    annotations: List[str] = Field(default_factory=list)
    estimated_prefix_hit_num_blocks: Optional[int] = None
    disaggregated_params: Optional[DisaggregatedParams] = Field(default=None)
    kv_cache_retention_config: Optional[KvCacheRetentionConfig] = Field(default=None)
