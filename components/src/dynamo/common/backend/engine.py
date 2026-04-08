# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Optional

from dynamo._core import Context


@dataclass
class EngineConfig:
    model: str
    served_model_name: Optional[str] = None
    context_length: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None


class DynamoEngine(ABC):
    @abstractmethod
    async def init(self) -> EngineConfig:
        ...

    @abstractmethod
    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[dict, None]:
        ...
        yield  # type: ignore[misc]

    @abstractmethod
    async def cleanup(self) -> None:
        ...
