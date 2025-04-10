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
IMPORTANT:
- This is only supposed to be used by dynamo-run launcher.
- It is part of bring-your-own-engine python feature in dynamo-run.
"""
import json
import os
import sys
from pathlib import Path

from tensorrt_llm.logger import logger

from dynamo.runtime import dynamo_endpoint

# Add the project root to the Python path
project_root = str(Path(__file__).parents[1])  # Go up to llm directory
if project_root not in sys.path:
    sys.path.append(project_root)

from common.base_engine import BaseTensorrtLLMEngine, get_sampling_params  # noqa: E402
from common.parser import parse_dynamo_run_args, LLMAPIConfig  # noqa: E402
from common.protocol import (  # noqa: E402
    DynamoTRTLLMChatCompletionRequest,
    DynamoTRTLLMChatCompletionStreamResponse,
)
from common.utils import ServerType  # noqa: E402

logger.set_level(os.getenv("DYN_TRTLLM_LOG_LEVEL", "info"))


# TODO: support disaggregated as well
async def chat_generator(engine: BaseTensorrtLLMEngine, request):
    if engine._llm_engine is None:
        raise RuntimeError("Engine not initialized")

    logger.debug(f"Received chat request: {request}")
    preprocessed_request = await engine.chat_processor.preprocess(request)
    engine_generator = engine._llm_engine.generate_async(
        inputs=preprocessed_request.prompt,
        sampling_params=get_sampling_params(preprocessed_request.sampling_params),
        disaggregated_params=None,
        streaming=True,
    )
    async for raw_response in engine.chat_processor.postprocess(
        engine_generator, request, preprocessed_request.conversation
    ):
        response = DynamoTRTLLMChatCompletionStreamResponse.model_validate_json(
            raw_response
        )
        yield json.loads(response.model_dump_json(exclude_unset=True))


class DynamoTRTLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_config: LLMAPIConfig):
        super().__init__(engine_config=engine_config, server_type=ServerType.DYN_RUN)
        # Initialize the engine
        self._init_engine()
        self.chat_processor.using_engine_generator = True


engine = None  # Global variable to store the engine instance. This is initialized in the main function.


def init_global_engine(args, engine_config):
    global engine
    logger.debug(f"Received args: {args}")
    logger.info(f"Initializing global engine with engine config: {engine_config}")
    engine = DynamoTRTLLMEngine(engine_config)


@dynamo_endpoint(
    DynamoTRTLLMChatCompletionRequest, DynamoTRTLLMChatCompletionStreamResponse
)
async def generate(request):
    async for response in chat_generator(engine, request):
        yield response


if __name__ == "__main__":
    args, engine_config = parse_dynamo_run_args()
    init_global_engine(args, engine_config)
