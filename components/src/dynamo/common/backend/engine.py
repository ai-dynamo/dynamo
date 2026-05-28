# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Required, TypedDict

from dynamo._core import Context

from .publisher import KvEventSource

if TYPE_CHECKING:
    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]

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

    Disaggregated-serving keys (``prefill_result``, ``bootstrap_info``)
    are set by the frontend's PrefillRouter on decode requests; engines
    read them via ``dynamo.common.backend.disagg`` helpers.
    """

    token_ids: Required[list[int]]
    sampling_options: dict[str, Any]
    stop_conditions: dict[str, Any]
    output_options: dict[str, Any]
    prefill_result: dict[str, Any]
    bootstrap_info: dict[str, Any]


class GenerateChunk(TypedDict, total=False):
    """Single chunk yielded by ``LLMEngine.generate()``.

    Every chunk must include ``token_ids`` and ``index``.
    Use ``index=0`` for single-choice responses. The final chunk must
    additionally include ``finish_reason`` and ``completion_usage``.
    Prefill terminals carry ``disaggregated_params``.

    ``log_probs`` and ``top_logprobs`` are populated when the request set
    ``output_options.logprobs`` (or ``prompt_logprobs``) and the engine
    computed them; each ``top_logprobs`` entry follows
    ``{rank, token_id, token, logprob, bytes}`` (bytes is optional).
    ``cum_log_probs`` is a per-completion summary — engines emit it on
    the terminal chunk only, alongside ``completion_usage``.
    """

    token_ids: Required[list[int]]
    index: Required[int]
    finish_reason: str
    completion_usage: dict[str, int]
    disaggregated_params: dict[str, Any]
    log_probs: list[float]
    top_logprobs: list[list[dict[str, Any]]]
    cum_log_probs: float


@dataclass
class EngineConfig:
    model: str
    served_model_name: Optional[str] = None
    context_length: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    # Number of data-parallel ranks this worker hosts (defaults to 1).
    # Engines with attention-DP set this from their engine-side count
    # (e.g. TRT-LLM's `get_attention_dp_size()`).
    data_parallel_size: Optional[int] = None
    # Global index of the first DP rank this worker hosts (defaults to 0).
    # Non-zero only under multi-worker DP layouts where each worker owns a
    # sub-range — vLLM hybrid/external LB, SGLang DP-attention across
    # multiple nodes. The router enumerates ranks
    # `[data_parallel_start_rank, data_parallel_start_rank + data_parallel_size)`.
    data_parallel_start_rank: Optional[int] = None
    # Bootstrap address advertised to decode peers. Only meaningful for
    # backends with a Dynamo-level host/port handshake (today: SGLang).
    # Backends whose KV transport is internal — TRT-LLM, vLLM
    # NixlConnector — leave these None.
    #
    # Engines that do use it populate these from `start()` after the
    # engine has resolved its KV-transport listening address. When both
    # are set, the Rust Worker publishes them via
    # `ModelRuntimeConfig.disaggregated_endpoint` so the frontend's
    # `PrefillRouter` can take its optimised Bootstrap path (route
    # decode concurrent with prefill).
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None
    runtime_data: Optional[dict[str, Any]] = None


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
    async def start(self, worker_id: int) -> EngineConfig:
        """Start the engine and return registration metadata.

        After this returns the engine MUST be ready to accept ``generate()``
        calls.  ``Worker`` will register the model and begin serving
        immediately.

        ``worker_id`` is an opaque, runtime-allocated unique identifier for
        this worker. It is stable from ``start()`` onward for the worker's
        lifetime and unique across replicas in the cluster. Engines that
        need a per-worker key for cluster-wide bookkeeping (e.g. TRT-LLM's
        ``disagg_machine_id`` snowflake field) should derive it from this
        value rather than hashing host/pid or asking operators for a CLI
        override. The internal mechanism (discovery instance ID) is not
        part of the contract — engines should treat it as opaque.
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

    async def abort(self, context: Context) -> None:
        """Abort an in-flight request (optional, default no-op).

        Called by Worker when the client disconnects or
        the request is cancelled.  Override to release engine resources
        (KV cache, scheduler slots, etc.).

        ``context.metadata`` in this callback reflects the original
        propagated request metadata snapshot. Mutations made to
        ``context.metadata`` during :meth:`generate` are not visible here.
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

        ``Worker`` guarantees:

        * ``cleanup()`` runs after a successful ``start()`` on shutdown —
          the common case.
        * ``cleanup()`` also runs after ``start()`` raised, on the partial
          state the engine may have allocated before failing (inner LLM
          handle, sockets, background tasks). Implementations **must**
          be null-safe: guard each resource with an ``is None`` check
          so a partially constructed engine can be released without
          raising.
        * ``cleanup()`` is **not** called when ``start()`` was never
          invoked (e.g. pre-start shutdown). Engines whose constructors
          allocate resources should release them via ``__del__`` /
          context-manager semantics rather than rely on ``cleanup()``.

        ``cleanup()`` is never invoked concurrently with ``start()`` or
        another ``cleanup()`` — ``Worker``'s state machine serializes
        those transitions. The conformance kit asserts that a second
        ``cleanup()`` call after a successful first is a safe no-op.
        """
        ...

    async def kv_event_sources(self) -> list[KvEventSource]:
        """KV event sources, one per data-parallel rank. Default opts out
        of KV-aware routing. ``Worker`` calls once after :meth:`start`."""
        return []

    async def register_prometheus(self, metrics: "EngineMetrics") -> None:
        """Bridge a vendor-prefixed Prometheus registry into the runtime's
        ``/metrics`` output via :func:`metrics.add_expfmt_callback`. Default
        no-op. See :mod:`dynamo.common.backend.metrics` for helpers. Do not
        retain ``metrics`` past return.

        Framework-owned lifecycle + per-rank gauges
        (``dynamo_component_{cleanup_time_seconds,drain_time_seconds,model_load_time_seconds,total_blocks,gpu_cache_usage_percent,kv_cache_hit_rate}``)
        are owned and registered by the framework Rust-side — they do NOT
        require the engine to implement this method."""

    def component_metrics_dp_ranks(self) -> list[int]:
        """Declare the data-parallel ranks this engine publishes
        per-rank snapshots for. Empty (default) opts out.

        Stable for the engine's lifetime. ``Worker`` constructs a
        :class:`SnapshotPublisher` sized to these ranks and hands it
        back via :meth:`attach_snapshot_publisher`. The engine then
        calls ``publisher.publish(rank, snap)`` from its stat-logger
        thread — event-driven, no polling.

        ``ComponentSnapshot.kv_cache_hit_rate`` is tri-state:
        ``None`` means "no data yet" or "no prefix cache" (gauge
        skipped), ``0.0`` is a legitimate measurement (zero hits)."""
        return []

    def attach_snapshot_publisher(self, publisher: Any) -> None:
        """Framework hands the engine the Rust-owned
        :class:`SnapshotPublisher` once, after ``setup_metrics``
        constructed it from :meth:`component_metrics_dp_ranks`. Stash
        the reference; call ``publisher.publish(rank, snap)`` from your
        stat-logger thereafter.

        Only invoked when :meth:`component_metrics_dp_ranks` returns
        non-empty. Default is no-op so engines that opt out don't need
        to override."""

    async def health_check_payload(self) -> Optional[dict[str, Any]]:
        """Canary payload the runtime sends through :meth:`generate` when
        the endpoint is idle. Return ``None`` (default) to disable active
        probing. ``Worker`` calls this once after :meth:`start` and resolves
        ``DYN_HEALTH_CHECK_PAYLOAD`` / ``--health-check-payload`` overrides
        on top."""
        return None
