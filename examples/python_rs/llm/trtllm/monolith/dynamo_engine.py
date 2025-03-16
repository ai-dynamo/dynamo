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
import sys
from pathlib import Path

from tensorrt_llm.logger import logger
from tensorrt_llm.serve.openai_protocol import ChatCompletionStreamResponse

from dynamo.runtime import dynamo_endpoint

# Add the project root to the Python path
project_root = str(Path(__file__).parents[1])  # Go up to trtllm directory
if project_root not in sys.path:
    sys.path.append(project_root)

from common.base_engine import (  # noqa: E402
    BaseTensorrtLLMEngine,
    TensorrtLLMEngineConfig,
)
from common.generators import chat_generator  # noqa: E402
from common.parser import parse_dynamo_run_args  # noqa: E402
from common.protocol import AdaptedChatCompletionRequest  # noqa: E402

logger.set_level("info")


class DynamoTRTLLMEngine(BaseTensorrtLLMEngine):
    """
    Request handler for the generate endpoint
    """

    def __init__(self, trt_llm_engine_config: TensorrtLLMEngineConfig):
        super().__init__(trt_llm_engine_config)


engine = None  # Global variable to store the engine instance. This is initialized in the main function.


def init_global_engine(args, engine_config):
    global engine
    logger.debug(f"Received args: {args}")
    logger.info(f"Initializing global engine with engine config: {engine_config}")
    trt_llm_engine_config = TensorrtLLMEngineConfig(
        engine_config=engine_config,
    )
    engine = DynamoTRTLLMEngine(trt_llm_engine_config)


@dynamo_endpoint(AdaptedChatCompletionRequest, ChatCompletionStreamResponse)
async def generate(request):
    if request.max_completion_tokens is not None:
        request.max_tokens = request.max_completion_tokens
    async for response in chat_generator(engine, request):
        yield response


if __name__ == "__main__":
    args, engine_config = parse_dynamo_run_args()
    init_global_engine(args, engine_config)
