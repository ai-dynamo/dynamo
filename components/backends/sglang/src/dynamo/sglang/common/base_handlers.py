# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from typing import Any, Optional

import sglang as sgl
from sglang.srt.server_args import ServerArgs


class BaseWorkerHandler(ABC):
    """
    Abstract base class for sglang request handlers. We use this to implement native sglang endpoints for
    workers
    """

    @abstractmethod
    def __init__(
        self,
        engine: sgl.Engine,
        server_args: ServerArgs,
        component,
        decode_client: Optional[Any] = None,
    ):
        self.engine = engine
        self.server_args = server_args
        self.component = component

    @abstractmethod
    async def generate(self, request):
        """Generate tokens from the engine"""
        ...

    async def flush_cache(self, request: dict):
        """Flush KV cache for each worker"""
        _ = request
        await self.engine.tokenizer_manager.flush_cache()
        yield True

    async def start_expert_distribution_record(self, request: dict):
        """
        Start recording expert distribution.
        """
        _ = request
        await self.engine.tokenizer_manager.start_expert_distribution_record()
        yield True

    async def stop_expert_distribution_record(self, request: dict):
        """
        Stop recording expert distribution.
        """
        _ = request
        await self.engine.tokenizer_manager.stop_expert_distribution_record()
        yield True

    async def dump_expert_distribution_record(self, request: dict):
        """
        Dumps the expert distribution record to the directory specified in the environment variable `SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR`.
        """
        _ = request
        await self.engine.tokenizer_manager.dump_expert_distribution_record()
        yield True
