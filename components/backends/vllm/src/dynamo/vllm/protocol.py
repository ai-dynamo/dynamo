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

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict
from vllm.outputs import CompletionOutput
from vllm.sequence import PromptLogprobs, RequestMetrics


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
    # lora_request: Optional[LoRARequest] = None
    # encoder_prompt: Optional[str] = None
    # encoder_prompt_token_ids: Optional[List[int]] = None
    # num_cached_tokens: Optional[int] = None
    # multi_modal_placeholders: Optional[MultiModalPlaceholderDict] = None
    kv_transfer_params: Optional[dict[str, Any]] = None
