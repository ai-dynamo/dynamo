# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Required, Tuple, TypedDict

from dynamo._core import Context
from dynamo.llm import ModelType

if TYPE_CHECKING:
    from dynamo.runtime import Endpoint

    from .worker import WorkerConfig


# ---------------------------------------------------------------------------
# Request / response contracts for generate()
#
# These TypedDicts document the shared fields that all engines read/write.
# Engine-specific keys (output_options, guided_decoding internals, etc.)
# flow through naturally — TypedDict doesn't reject extra keys at runtime.
# ---------------------------------------------------------------------------


class GenerateRequest(TypedDict, total=False):
    """Inbound request dict passed to ``LLMEngine.generate()``.

    ``token_ids`` is always present (set by the Rust preprocessor).
    The remaining groups are optional — engines should access them
    defensively with ``.get(key, {})``.

    ``disaggregated_state`` is the opaque per-request state carried between
    prefill and decode workers in disaggregated serving. The schema is
    backend-defined; the abstraction only guarantees that whatever a prefill
    worker emits in ``GenerateChunk.disaggregated_state`` (or whatever the
    runtime synthesizes for router-resolved bootstrap mode) reaches the
    decode worker through this field.
    """

    token_ids: Required[list[int]]
    sampling_options: dict[str, Any]
    stop_conditions: dict[str, Any]
    output_options: dict[str, Any]
    disaggregated_state: dict[str, Any]


class GenerateChunk(TypedDict, total=False):
    """Single chunk yielded by ``LLMEngine.generate()``.

    Every chunk must include ``token_ids`` and ``index``.
    Use ``index=0`` for single-choice responses. The final chunk must
    additionally include ``finish_reason`` and ``completion_usage``.

    ``disaggregated_state`` is the opaque state a prefill worker emits for
    the synchronous-fallback disagg path; the runtime extracts it and
    threads it into the decode request's ``disaggregated_state``. Backends
    using router-resolved bootstrap mode (advertised via
    ``EngineConfig.disaggregated_endpoint``) do not set this field.
    """

    token_ids: Required[list[int]]
    index: Required[int]
    finish_reason: str
    completion_usage: dict[str, int]
    disaggregated_state: dict[str, Any]


@dataclass
class EngineConfig:
    """Registration metadata returned by ``LLMEngine.start()``.

    Disaggregated serving fields:

    * ``model_type`` — when not ``None``, OR'd into the registered model
      type. A prefill worker sets this to ``ModelType.Prefill`` so Dynamo's
      discovery promotes it to a prefill instance.
    * ``disaggregated_endpoint`` — when set, ``Worker`` calls
      ``runtime_config.set_disaggregated_endpoint(host, port)`` so the
      Rust ``PrefillRouter`` can synthesize a ``disaggregated_state``
      payload (host/port/room) into each decode request. Used by the
      router-resolved bootstrap path (SGLang, TokenSpeed). Backends that
      use the synchronous-fallback path (vLLM, TRT-LLM) leave this ``None``
      and emit state from prefill ``generate()`` chunks instead.
    * ``enable_local_indexer`` — engine override for the worker-level
      flag of the same name. ``None`` (default) respects ``WorkerConfig``;
      a value forces the runtime config. Prefill-only workers typically
      want ``False`` since they ship KV out immediately.
    """

    model: str
    served_model_name: Optional[str] = None
    context_length: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    model_type: Optional[ModelType] = None
    disaggregated_endpoint: Optional[Tuple[str, int]] = None
    enable_local_indexer: Optional[bool] = None


class LLMEngine(ABC):
    """Abstract base for inference engines.

    Lifecycle:
        1. from_args(argv) -- parse CLI args, return (engine, WorkerConfig)
        2. start()         -- start the engine, return EngineConfig metadata.
                              After start() returns, generate() MUST be ready
                              to accept calls. Worker begins serving
                              immediately after start().
        3. generate()      -- called for each request (concurrent calls expected)
        4. abort()         -- called when a request is cancelled (optional, default no-op)
        5. cleanup()       -- called once on shutdown, release all resources
    """

    @classmethod
    @abstractmethod
    async def from_args(
        cls, argv: list[str] | None = None
    ) -> tuple[LLMEngine, WorkerConfig]:
        """Parse CLI args and construct the engine (not yet started).

        Args:
            argv: Command-line arguments.  ``None`` means ``sys.argv[1:]``.

        Returns:
            A ``(engine, worker_config)`` pair.
        """
        ...

    @abstractmethod
    async def start(self) -> EngineConfig:
        """Start the engine and return registration metadata.

        After this returns the engine MUST be ready to accept ``generate()``
        calls.  ``Worker`` will register the model and begin serving
        immediately.
        """
        ...

    @abstractmethod
    async def generate(
        self, request: GenerateRequest, context: Context
    ) -> AsyncGenerator[GenerateChunk, None]:
        """Yield streaming response chunks for a single request.

        Called concurrently for multiple in-flight requests.

        Each chunk: ``{"token_ids": [...], "index": 0}``
        Final chunk must include: ``{"token_ids": [...], "index": 0,
        "finish_reason": "...", "completion_usage": {...}}``
        """
        ...
        yield  # type: ignore[misc]

    async def start_kv_events(
        self, endpoint: "Endpoint", engine_config: EngineConfig
    ) -> None:
        """Start optional backend-specific KV event relays.

        Backends that publish native KV cache events over a side channel can
        override this hook once the Dynamo endpoint exists. The default keeps
        unified backends that do not support KV events unchanged.
        """
        return None

    async def abort(self, context: Context) -> None:
        """Abort an in-flight request (optional, default no-op).

        Called by Worker when the client disconnects or
        the request is cancelled.  Override to release engine resources
        (KV cache, scheduler slots, etc.).
        """

    async def drain(self) -> None:
        """Drain in-flight engine work before cleanup (optional, default no-op).

        Called once during graceful shutdown after the discovery unregister
        + grace-period sleep, but before :meth:`cleanup`.  Use it for
        backend-side draining that must complete while the distributed
        runtime (NATS / etcd) is still alive — e.g. waiting for in-flight
        NIXL KV transfers on prefill workers (issue #7319), so downstream
        decode workers don't observe a use-after-free on freed GPU memory.

        Failures are logged and swallowed; shutdown proceeds regardless.
        """

    @abstractmethod
    async def cleanup(self) -> None:
        """Release all engine resources.

        ``Worker`` invokes ``cleanup()`` at most once, only after ``start()``
        has returned successfully, and never concurrently with ``start()`` or
        another ``cleanup()``. Implementations do not need to defend against
        pre-start, concurrent-with-start, or double-cleanup invocations —
        ``Worker``'s lifecycle state machine serializes these transitions.
        """
        ...
