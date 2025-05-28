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
Universal Request Tracing Frontend Example

This example shows how to use Dynamo SDK's built-in X-Request-Id support
without manual implementation. Just add @auto_trace_endpoints decorator!
"""

import logging

from components.processor import Processor
from fastapi import FastAPI, Request
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest

from dynamo.sdk import DYNAMO_IMAGE, auto_trace_endpoints, depends, endpoint, service

logger = logging.getLogger(__name__)

app = FastAPI(title="Universal Request Tracing Frontend")


@auto_trace_endpoints
@service(
    dynamo={"enabled": True, "namespace": "dynamo"},
    image=DYNAMO_IMAGE,
    app=app,
)
class Frontend:
    """
    Frontend with automatic X-Request-Id support.

    Benefits:
    - Zero configuration required
    - Automatic header extraction/generation
    - Automatic response header injection
    - Request ID propagation to downstream components
    """

    processor = depends(Processor)

    @endpoint(is_api=True, path="/v1/chat/completions", methods=["POST"])
    async def chat_completions(
        self, request: Request, chat_request: ChatCompletionRequest
    ):
        """
        OpenAI-compatible chat completions with automatic X-Request-Id support.

        The @auto_trace_endpoints decorator automatically:
        1. Extracts X-Request-Id from request.headers
        2. Generates UUID if not provided
        3. Passes request_id to processor.chat_completions()
        4. Adds X-Request-Id header to response

        No manual code needed! 🎉
        """
        logger.info("Processing chat completion request")

        async for response in self.processor.chat_completions(chat_request):
            yield response

    @endpoint(is_api=True, path="/v1/completions", methods=["POST"])
    async def completions(
        self, request: Request, completion_request: CompletionRequest
    ):
        """
        OpenAI-compatible completions with automatic X-Request-Id support.
        """
        logger.info("Processing completion request")

        async for response in self.processor.completions(completion_request):
            yield response

    @endpoint(is_api=True, path="/health", methods=["GET"])
    async def health(self, request: Request):
        """
        Health check endpoint - also gets automatic X-Request-Id support!
        """
        return {"status": "healthy", "service": "universal-tracing-frontend"}
