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


import abc
import logging

from common.chat_processor import ChatProcessor
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client_from_engine_args,
)

logger = logging.getLogger("vllm")


class BaseVllmEngine:
    """
    Request handler for the generate endpoint
    """

    def __init__(self, engine_args: AsyncEngineArgs):
        self.engine_args = engine_args
        self.model_config = self.engine_args.create_model_config()
        self.engine_client = None
        self.chat_processor: ChatProcessor | None = None
        self._engine_context = None

    async def initialize(self):
        """Initialize the engine client and related components."""
        logger.info("Initializing engine client")
        self._engine_context = build_async_engine_client_from_engine_args(
            self.engine_args
        )
        if self._engine_context is not None:
            self.engine_client = await self._engine_context.__aenter__()
            self.tokenizer = await self.engine_client.get_tokenizer()
            self.chat_processor = ChatProcessor(self.tokenizer, self.model_config)
        else:
            raise RuntimeError("Failed to initialize engine client")

    async def cleanup(self):
        """Cleanup resources."""
        print("Cleaning up engine client")
        if self._engine_context is not None:
            await self._engine_context.__aexit__(None, None, None)
            self._engine_context = None
            self.engine_client = None
            self.chat_processor = None

    async def __aenter__(self):
        await self.initialize()
        """Initialize with context manager syntax."""
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.cleanup()

    @abc.abstractmethod
    async def generate(self, raw_request):
        pass
