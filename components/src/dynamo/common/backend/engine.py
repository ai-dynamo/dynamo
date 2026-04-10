# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from dynamo._core import Context

if TYPE_CHECKING:
    from .worker import WorkerConfig


@dataclass
class EngineConfig:
    model: str
    served_model_name: Optional[str] = None
    context_length: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None


class LLMEngine(ABC):
    """Abstract base for inference engines.

    Lifecycle:
        1. from_args(argv) -- parse CLI args, construct engine (NOT started yet)
        2. init()          -- start the engine, return EngineConfig metadata.
                              After init() returns, generate() MUST be ready
                              to accept calls. Worker begins serving
                              immediately after init().
        3. generate()      -- called for each request (concurrent calls expected)
        4. abort()         -- called when a request is cancelled (optional, default no-op)
        5. cleanup()       -- called once on shutdown, release all resources
    """

    worker_config: WorkerConfig

    @classmethod
    @abstractmethod
    async def from_args(cls, argv: list[str] | None = None) -> LLMEngine:
        """Parse CLI args and construct the engine (not yet started).

        Implementations must set ``worker_config`` on the returned instance.

        Args:
            argv: Command-line arguments.  ``None`` means ``sys.argv[1:]``.
        """
        ...

    @abstractmethod
    async def init(self) -> EngineConfig:
        """Start the engine and return registration metadata.

        After this returns the engine MUST be ready to accept ``generate()``
        calls.  ``Worker`` will register the model and begin serving
        immediately.
        """
        ...

    @abstractmethod
    async def generate(
        self, request: dict, context: Context
    ) -> AsyncGenerator[dict, None]:
        """Yield streaming response chunks for a single request.

        Called concurrently for multiple in-flight requests.

        Each chunk: ``{"token_ids": [...]}``
        Final chunk must include: ``{"token_ids": [...], "finish_reason": "...",
        "completion_usage": {...}}``
        """
        ...
        yield  # type: ignore[misc]

    async def abort(self, context: Context) -> None:
        """Abort an in-flight request (optional, default no-op).

        Called by Worker when the client disconnects or
        the request is cancelled.  Override to release engine resources
        (KV cache, scheduler slots, etc.).
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Release all engine resources.  Called once on shutdown."""
        ...
