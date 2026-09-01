# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

from typing_extensions import Required

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode

from .publisher import KvEventSource

if TYPE_CHECKING:
    from dynamo._core.backend import EngineMetrics  # type: ignore[import-not-found]
    from dynamo.logits_processing import BaseLogitsProcessor

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

    Multimodal keys (``multi_modal_data``, ``mm_processor_kwargs``,
    ``mm_routing_info``) are populated by the frontend preprocessor when
    the request carries media. ``encoder_result`` is set by the
    frontend when forwarding a request from an Encode worker
    to a downstream Prefill/Aggregated peer; engines read it via
    :func:`dynamo.common.backend.multimodal.require_encoder_result`. All
    four are object-shaped (``dict``) by contract.

    ``model`` carries the requested model name (set by the Rust
    preprocessor). Engines that support dynamic LoRA read it to route a
    request to a loaded adapter.
    """

    token_ids: Required[list[int]]
    model: str
    sampling_options: dict[str, Any]
    stop_conditions: dict[str, Any]
    output_options: dict[str, Any]
    require_reasoning: bool
    prefill_result: dict[str, Any]
    bootstrap_info: dict[str, Any]
    multi_modal_data: dict[str, Any]
    mm_processor_kwargs: dict[str, Any]
    mm_routing_info: dict[str, Any]
    encoder_result: dict[str, Any]
    extra_args: dict[str, Any]
    routing: dict[str, Any]


class GenerateChunk(TypedDict, total=False):
    """Single chunk yielded by ``LLMEngine.generate()``.

    Every chunk must include ``token_ids`` and ``index``.
    Use ``index=0`` for single-choice responses. The final chunk must
    additionally include ``finish_reason``; ``completion_usage`` is
    optional (the OpenAI frontend aggregates it when present, and
    matches the Rust ``Option<CompletionUsage>`` /
    ``skip_serializing_if = "Option::is_none"`` semantics).

    Prefill terminals carry ``disaggregated_params`` for the
    PrefillRouter to forward to the decode peer. When the caller
    requested logprobs, chunks may also carry ``log_probs`` and
    ``top_logprobs`` aligned to ``token_ids`` — see
    :mod:`dynamo.common.backend.logprobs`.

    Encode terminals carry ``encoder_result`` (an opaque object the
    frontend forwards onto the downstream
    ``PreprocessedRequest.encoder_result``). Construct with
    :func:`dynamo.common.backend.multimodal.encoder_terminal_chunk`.
    """

    token_ids: Required[list[int]]
    index: Required[int]
    finish_reason: str
    completion_usage: dict[str, Any]
    disaggregated_params: dict[str, Any]
    encoder_result: dict[str, Any]
    log_probs: list[float]
    top_logprobs: list[list[dict[str, Any]]]
    # Forwarded verbatim to Rust `LLMEngineOutput.engine_data` as a
    # JSON object. Carries `prompt_logprobs` on the final chunk.
    engine_data: dict[str, Any]


@dataclass
class LlmRegistration:
    """Token-pipeline registration metadata (KV cache, data-parallel layout,
    disaggregation bootstrap). Set by :class:`LLMEngine`s; :class:`RawEngine`s
    leave :attr:`EngineConfig.llm` ``None``. A ``None`` field isn't advertised
    (the router falls back to its defaults)."""

    context_length: Optional[int] = None
    kv_cache_block_size: Optional[int] = None
    # Physical KV capacity per router-visible DP rank, never a process aggregate.
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    # DP ranks this worker hosts (default 1); attention-DP engines set it from
    # the engine count.
    data_parallel_size: Optional[int] = None
    # First DP rank this worker hosts (default 0). Non-zero only when a worker
    # owns a sub-range in a hybrid or externally load-balanced deployment;
    # the router enumerates [start, start + data_parallel_size).
    data_parallel_start_rank: Optional[int] = None
    # Bootstrap address advertised to decode peers. Backends with an internal
    # KV-transport handshake leave it None. When both are set, Worker publishes
    # them so the frontend's PrefillRouter can take its bootstrap path.
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


@dataclass
class EngineConfig:
    """Registration metadata returned by an engine's :meth:`start`.

    The neutral fields (``model``, ``served_model_name``, ``model_aliases``,
    ``runtime_data``) apply to every modality; token-pipeline metadata lives in
    the optional :attr:`llm` sub-record, which raw media engines leave ``None``.
    """

    model: str
    served_model_name: Optional[str] = None
    runtime_data: Optional[dict[str, Any]] = None
    # Token-pipeline registration metadata (KV cache, DP, bootstrap).
    # ``Some`` for LLMEngines; ``None`` for RawEngines.
    llm: Optional[LlmRegistration] = None
    # Kept after existing fields to preserve positional-constructor compatibility.
    model_aliases: list[str] = field(default_factory=list)


class BaseEngine(ABC):
    """Abstract base for all engines — the modality-agnostic lifecycle.

    ``Worker`` drives every engine through the same lifecycle regardless of
    modality; only the request/response shape of :meth:`generate` differs.
    That method is therefore declared on the modality-specific subclasses
    (:class:`LLMEngine` for token-based inference, :class:`RawEngine` for
    raw non-token media generation), not here.

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
    ) -> tuple[BaseEngine, WorkerConfig]:
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
        need a per-worker key for cluster-wide bookkeeping should derive it
        from this value rather than hashing host/pid or asking operators for a
        CLI override. The internal mechanism (discovery instance ID) is not
        part of the contract — engines should treat it as opaque.
        """
        ...

    async def abort(self, context: Context) -> None:
        """Abort an in-flight request (optional, default no-op).

        Called by Worker when the client disconnects or
        the request is cancelled.  Override to release engine resources
        (KV cache, scheduler slots, etc.).

        ``context.metadata`` in this callback reflects the original
        propagated request metadata snapshot. Mutations made to
        ``context.metadata`` during :meth:`generate` are not visible here.
        """

    async def is_quiescent(self) -> Optional[bool]:
        """Whether in-flight KV transfers are done, so :meth:`cleanup` may
        release GPU memory. The Rust ``Worker`` polls this on prefill workers
        between the grace period and :meth:`cleanup`:

        - ``True``  — quiescent; exit the drain loop now.
        - ``False`` — busy; poll again next tick.
        - ``None``  — no introspection (default); poll until the drain budget
          (``DYN_PREFILL_DRAIN_TIMEOUT_S``) expires. Never frees KV early.

        Aggregated/decode workers are never polled. Override only if the engine
        can observe transfer completion.
        """
        return None

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

    def supported_controls(self) -> set[str]:
        """Return the set of engine-control capability keys this engine supports.

        Controls are semantic operations on the engine's serving lifecycle.
        Engines advertise the keys they implement.
        """
        return set()

    async def engine_control(
        self, control: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle one advertised engine-control request."""
        return {
            "status": "error",
            "message": f"unsupported engine control: {control}",
        }

    def supported_updates(self) -> set[str]:
        """Return the set of engine-update capability keys this engine supports.

        Updates are a sibling surface to :meth:`supported_controls` for
        operations that mutate engine-managed assets rather than the engine's
        serving lifecycle. Engines advertise the keys they implement.
        """
        return set()

    async def engine_update(self, update: str, body: dict[str, Any]) -> dict[str, Any]:
        """Handle one advertised engine-update request."""
        return {
            "status": "error",
            "message": f"unsupported engine update: {update}",
        }

    async def on_endpoint_ready(self, endpoint) -> None:
        """Receive the runtime serving ``Endpoint`` once, before serving begins.

        Default no-op. Engines that publish their own discovery records stash
        it for use from :meth:`engine_update`. ``Worker`` calls this exactly
        once; a raised exception is fatal to startup."""
        return None


class LLMEngine(BaseEngine):
    """Abstract base for token-based inference engines.

    The token pipeline: the Rust preprocessor tokenizes the prompt and sets
    ``token_ids`` on the request; :meth:`generate` yields token chunks that
    the Rust postprocessor detokenizes. Registered with
    ``ModelInput.Tokens`` and served through the token request adapter.
    """

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

    async def kv_event_sources(self) -> list[KvEventSource]:
        """KV event sources, one per data-parallel rank. Default opts out
        of KV-aware routing. ``Worker`` calls once after :meth:`start`."""
        return []

    async def logits_processor_spec(self) -> "LogitsProcessorSpec | None":
        """Return backend-neutral logits-processor activation data.

        The default opts out. An engine that overrides this method resolves
        and caches the specification during startup, then passes it to
        :func:`logits_processors_for_request` from :meth:`generate`. The
        engine integration remains responsible for realizing each entry into
        its inference library's processor type.
        """
        return None


# Raw (non-token) request/response for RawEngine.generate. The PyO3 bridge
# passes the request through as a JSON ``dict`` and serializes each yielded
# object back — no Rust request type (the modality-neutral trade-off).
# Canonical field schemas: NvCreateImageRequest/NvImagesResponse in
# dynamo.common.protocols.image_protocol (videos: video_protocol).
RawRequest = dict[str, Any]
RawResponseChunk = dict[str, Any]


class RawEngine(BaseEngine):
    """Engines for raw, non-token generation (image, video, audio).

    Named for the *contract*, not a use case: unlike :class:`LLMEngine` there
    is no token pipeline — the frontend forwards the OpenAI-shaped request as a
    JSON object and :meth:`generate` yields the response object(s) directly.
    Registered with ``ModelInput.Text`` and served through the raw request
    adapter (no tokenization or KV cache). The ``dict`` contract is
    modality-neutral, so a new media modality is a new engine, not a new
    framework path; one engine may serve several modalities. Yield one
    (terminal) object, or intermediate progress objects ending with a terminal
    one. Subclasses like :class:`DiffusionEngine` add no contract.
    """

    @abstractmethod
    async def generate(
        self, request: RawRequest, context: Context
    ) -> AsyncGenerator[RawResponseChunk, None]:
        """Yield response object(s) for a single raw-media request.

        ``request`` is the raw OpenAI-shaped request body (see
        :data:`RawRequest`); yield the response body object(s) (see
        :data:`RawResponseChunk`). For non-streaming modalities yield exactly
        one (terminal) object; for streaming modalities yield intermediate
        progress objects ending with the terminal one.
        """
        ...
        yield  # type: ignore[misc]


class DiffusionEngine(RawEngine):
    """A :class:`RawEngine` for diffusion-family generation (image/video via
    VisualGen, DiffGenerator). Names the family only — non-diffusion raw
    modalities (e.g. TTS audio) subclass :class:`RawEngine` directly. Routing
    keys off :class:`RawEngine`, so any subclass uses the raw adapter.
    """


# ---------------------------------------------------------------------------
# Backend-neutral custom logits-processor contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ForcedTokenSequenceSpec:
    """Force the configured token IDs in order, then force EOS."""

    token_ids: tuple[int, ...]
    eos_token_id: int


@dataclass(frozen=True)
class PythonProcessorSpec:
    """Factory for an in-process Python logits processor.

    No built-in backend currently realizes this entry. It is reserved for
    custom or future in-process integrations that invoke the factory directly.
    Serialization helpers reject it.
    """

    factory: Callable[[], "BaseLogitsProcessor"]


LogitsProcessorEntry = ForcedTokenSequenceSpec | PythonProcessorSpec


@dataclass(frozen=True)
class LogitsProcessorSpec:
    """Engine-declared logits-processor activation.

    ``generation_only`` skips roles that do not emit the visible token stream.
    Entries are immutable activation data; an engine integration must create
    fresh stateful processor instances for each request.
    """

    entries: tuple[LogitsProcessorEntry, ...]
    generation_only: bool = True


_GENERATION_STAGES = frozenset(
    {DisaggregationMode.AGGREGATED, DisaggregationMode.DECODE}
)


def is_generation_stage(disaggregation_mode: DisaggregationMode) -> bool:
    """Return whether a worker role emits the visible token stream."""
    return disaggregation_mode in _GENERATION_STAGES


def logits_processors_for_request(
    spec: LogitsProcessorSpec | None, *, disaggregation_mode: DisaggregationMode
) -> list[LogitsProcessorEntry]:
    """Return the logits-processor entries to realize for one request."""
    if spec is None:
        return []
    if spec.generation_only and not is_generation_stage(disaggregation_mode):
        return []
    return list(spec.entries)


_FORCED_SEQUENCE_KIND = "forced_sequence"


def serialize_logits_processor_entries(
    entries: Sequence[LogitsProcessorEntry],
) -> list[dict[str, Any]]:
    """Encode supported logits-processor entries as JSON-safe dictionaries."""
    payload: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, ForcedTokenSequenceSpec):
            payload.append(
                {
                    "kind": _FORCED_SEQUENCE_KIND,
                    "token_ids": list(entry.token_ids),
                    "eos_token_id": entry.eos_token_id,
                }
            )
        else:
            raise TypeError(
                f"logits-processor entry of type {type(entry).__name__} is not "
                "serializable; only ForcedTokenSequenceSpec can cross a "
                "serialized request boundary"
            )
    return payload


def deserialize_logits_processor_entries(
    payload: Sequence[dict[str, Any]],
) -> list[LogitsProcessorEntry]:
    """Decode logits-processor entries from their JSON-safe representation."""
    entries: list[LogitsProcessorEntry] = []
    for item in payload:
        kind = item.get("kind")
        if kind == _FORCED_SEQUENCE_KIND:
            entries.append(
                ForcedTokenSequenceSpec(
                    token_ids=tuple(item["token_ids"]),
                    eos_token_id=item["eos_token_id"],
                )
            )
        else:
            raise ValueError(f"unknown logits-processor entry kind: {kind!r}")
    return entries
