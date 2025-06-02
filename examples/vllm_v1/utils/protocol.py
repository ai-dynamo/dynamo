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

import json
from typing import Any, List, Optional

import msgspec
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_core import core_schema
from typing_extensions import NotRequired
from vllm.inputs import TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import PromptLogprobs, RequestMetrics

TokenIdType = int


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


class PreprocessedRequest(BaseModel):
    token_ids: List[TokenIdType]
    stop_conditions: StopConditions
    sampling_options: SamplingOptions
    eos_token_ids: List[TokenIdType] = Field(default_factory=list)
    mdc_sum: Optional[str] = None
    annotations: List[str] = Field(default_factory=list)


# Hack to override the type of multi_modal_data in TokensPrompt
# as pydantic doesn't understand generic types
# TokensPrompt is defined here: https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/inputs/data.py#L38
# multi_modal_data is defined here: https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L103
# ModalityData is defined here: https://github.com/vllm-project/vllm/blob/main/vllm/multimodal/inputs.py#L80
class PatchedTokensPrompt(TokensPrompt):
    multi_modal_data: NotRequired[Optional[Any]]  # type: ignore


# Monkey-patch the SamplingParams and KVTransferParams types to add a dummy core schema so pydantic can validate them
# Sampling params is a mspspec struct
# SamplingParams is defined here: https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/sampling_params.py#L88

SamplingParams.__get_pydantic_core_schema__ = classmethod(
    lambda cls, source, handler: core_schema.any_schema()
)


LoRARequest.__get_pydantic_core_schema__ = classmethod(
    lambda cls, source, handler: core_schema.any_schema()
)


class vLLMGenerateRequest(BaseModel):
    """
    Serializable class of all the fields vLLM engine requires for inference
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    prompt: PatchedTokensPrompt
    sampling_params: SamplingParams
    request_id: str

    @field_validator("sampling_params", mode="before")
    @classmethod
    def parse_sampling_params(cls, v: Any) -> SamplingParams:
        if isinstance(v, str):
            v = json.loads(v)
        if isinstance(v, dict):
            return SamplingParams(**v)
        return v

    model_config = ConfigDict(
        json_encoders={SamplingParams: lambda v: msgspec.json.encode(v)}
    )


class MyRequestOutput(BaseModel):
    """
    RequestOutput from vLLM is not serializable by default
    https://github.com/vllm-project/vllm/blob/a4c402a756fa3213caf9d2cde0e4ceb2d57727f2/vllm/outputs.py#L85

    This class is used to serialize the RequestOutput and any recursively defined types
    We can do this because PromptLogprobs, RequestMetrics, and CompletionOutput are all serializable dataclasses
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str
    prompt: Optional[str] = None
    prompt_token_ids: Optional[List[int]] = None
    prompt_logprobs: Optional[PromptLogprobs] = None
    outputs: List[CompletionOutput]
    finished: bool
    metrics: Optional[RequestMetrics] = None
    kv_transfer_params: Optional[dict[str, Any]] = None
    # lora_request: Optional[LoRARequest] = None
    # encoder_prompt: Optional[str] = None
    # encoder_prompt_token_ids: Optional[List[int]] = None
    # num_cached_tokens: Optional[int] = None
    # multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
