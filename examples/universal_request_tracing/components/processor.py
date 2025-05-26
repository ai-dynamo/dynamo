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
Universal Request Tracing Processor Example

This example shows how to use RequestTracingMixin for automatic request ID handling
in processor components without manual implementation.
"""

import logging
from enum import Enum
from typing import Optional, Union

from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

from dynamo.sdk import (
    RequestTracingMixin,
    depends,
    dynamo_endpoint,
    service,
    get_current_request_id,
)
from components.worker import VllmWorker
from components.router import Router

logger = logging.getLogger(__name__)


class RequestType(Enum):
    CHAT = "chat"
    COMPLETION = "completion"


@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    resources={"cpu": "10", "memory": "20Gi"},
    workers=1,
)
class Processor(RequestTracingMixin):
    """
    Processor with automatic X-Request-Id support via RequestTracingMixin.
    
    Benefits:
    - ensure_request_id(): Get request ID from parameter, context, or generate new
    - log_with_request_id(): Automatic logging with request ID
    - get_current_request_id(): Access request ID anywhere in the call stack
    """
    
    worker = depends(VllmWorker)
    router = depends(Router)

    def __init__(self):
        self.model_name = "meta-llama/Llama-3.2-3B-Instruct"
        self.use_router = True

    @dynamo_endpoint(name="chat/completions")
    async def chat_completions(self, raw_request: ChatCompletionRequest, request_id: Optional[str] = None):
        """
        Chat completions with automatic request ID handling.
        
        The RequestTracingMixin provides:
        - ensure_request_id(): Gets request_id from parameter, context, or generates new
        - log_with_request_id(): Logs with automatic request ID prefix
        """
        request_id = self.ensure_request_id(request_id)
        
        self.log_with_request_id("info", f"Processing chat completion for model: {raw_request.model}")
        
        async for response in self._generate(raw_request, RequestType.CHAT, request_id):
            yield response

    @dynamo_endpoint(name="completions")
    async def completions(self, raw_request: CompletionRequest, request_id: Optional[str] = None):
        """
        Completions with automatic request ID handling.
        """
        request_id = self.ensure_request_id(request_id)
        self.log_with_request_id("info", f"Processing completion for model: {raw_request.model}")
        
        async for response in self._generate(raw_request, RequestType.COMPLETION, request_id):
            yield response

    async def _generate(self, raw_request: Union[ChatCompletionRequest, CompletionRequest], 
                       request_type: RequestType, request_id: str):
        """
        Internal generation method with request ID propagation.
        """
        self.log_with_request_id("debug", f"Starting generation with request_type: {request_type}")
        
        if self.use_router:
            self.log_with_request_id("debug", "Using router for request routing")
            engine_generator = await self.router_client.generate(request_data)
        else:
            self.log_with_request_id("debug", "Direct worker processing")
            engine_generator = await self.worker_client.generate(request_data)
        
        async for response in engine_generator:
            current_id = get_current_request_id()
            logger.debug(f"Processing response for request_id: {current_id}")
            yield response

    async def _some_internal_method(self):
        """
        Example of accessing request ID in any internal method.
        """
        request_id = get_current_request_id()
        if request_id:
            logger.info(f"Internal processing for request_id: {request_id}")
        
        request_id = self.ensure_request_id()
        self.log_with_request_id("debug", "Internal method processing")
