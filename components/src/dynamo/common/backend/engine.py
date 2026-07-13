# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Optional, TypedDict

from typing_extensions import Required

from dynamo._core import Context
from dynamo.common.constants import DisaggregationMode
from dynamo.common.snapshot.restore_context import RestoredRuntimeConfig

from .publisher import KvEventSource

if TYPE_CHECKING:
    from dynamo._core.backend import (  # type: ignore[import-not-found]
        EngineMetrics,
        PreRuntimeContext,
    )
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
    total_kv_blocks: Optional[int] = None
    max_num_seqs: Optional[int] = None
    max_num_batched_tokens: Optional[int] = None
    # DP ranks this worker hosts (default 1); attention-DP engines set it from
    # the engine count (e.g. TRT-LLM's get_attention_dp_size()).
    data_parallel_size: Optional[int] = None
    # First DP rank this worker hosts (default 0). Non-zero only when a worker
    # owns a sub-range (vLLM hybrid/external LB, multi-node SGLang DP-attention);
    # the router enumerates [start, start + data_parallel_size).
    data_parallel_start_rank: Optional[int] = None
    # Bootstrap address advertised to decode peers. Only for backends with a
    # Dynamo-level handshake (SGLang); internal-KV-transport backends (TRT-LLM,
    # vLLM NixlConnector) leave it None. When both are set, Worker publishes
    # them so the frontend's PrefillRouter can take its Bootstrap path.
    bootstrap_host: Optional[str] = None
    bootstrap_port: Optional[int] = None


@dataclass
class EngineConfig:
    """Registration metadata returned by an engine's :meth:`start`.

    The neutral fields (``model``, ``served_model_name``, ``runtime_data``)
    apply to every modality; token-pipeline metadata lives in the optional
    :attr:`llm` sub-record, which raw media engines leave ``None``.
    """

    model: str
    served_model_name: Optional[str] = None
    runtime_data: Optional[dict[str, Any]] = None
    # Token-pipeline registration metadata (KV cache, DP, bootstrap).
    # ``Some`` for LLMEngines; ``None`` for RawEngines.
    llm: Optional[LlmRegistration] = None


class PreRuntimeAction(str, Enum):
    CONTINUE = "continue"
    PREPARED = "prepared"
    EXIT_SUCCESS = "exit_success"


@dataclass(frozen=True)
class PreRuntimeOutcome:
    """Result returned by :meth:`BaseEngine.prepare_for_runtime`."""

    action: PreRuntimeAction
    restored_runtime: Optional[RestoredRuntimeConfig] = None

    @classmethod
    def continue_startup(cls) -> "PreRuntimeOutcome":
        return cls(PreRuntimeAction.CONTINUE)

    @classmethod
    def prepared(cls, restored_runtime: RestoredRuntimeConfig) -> "PreRuntimeOutcome":
        return cls(PreRuntimeAction.PREPARED, restored_runtime)

    @classmethod
    def exit_success(cls) -> "PreRuntimeOutcome":
        return cls(PreRuntimeAction.EXIT_SUCCESS)


class BaseEngine(ABC):
    """Abstract base for all engines — the modality-agnostic lifecycle.

    ``Worker`` drives every engine through the same lifecycle regardless of
    modality; only the request/response shape of :meth:`generate` differs.
    That method is therefore declared on the modality-specific subclasses
    (:class:`LLMEngine` for token-based inference, :class:`RawEngine` for
    raw non-token media generation), not here.

    Lifecycle:
        1. from_args(argv) -- parse CLI args, return (engine, WorkerConfig)
        2. prepare_for_runtime() -- optional snapshot/restore preparation before
                                   Dynamo opens distributed-runtime connections.
        3. start()         -- start/activate the engine, return EngineConfig metadata.
                              After start() returns, generate() MUST be ready
                              to accept calls. Worker begins serving
                              immediately after start().
        4. generate()      -- called for each request (concurrent calls expected)
        5. abort()         -- called when a request is cancelled (optional, default no-op)
        6. cleanup()       -- called once on shutdown, release all resources
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

    async def prepare_for_runtime(
        self, context: "PreRuntimeContext"
    ) -> PreRuntimeOutcome:
        """Optionally initialize snapshot state before DistributedRuntime.

        The default keeps cold startup unchanged. Snapshot-aware engines return
        ``prepared`` after restore or ``exit_success`` after the producer has
        been captured. ``context`` exposes ``is_cancelled()`` and an awaitable
        ``cancelled()`` for shutdown-aware sentinel waits.
        """

        del context
        return PreRuntimeOutcome.continue_startup()

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
        * ``cleanup()`` runs when ``prepare_for_runtime()`` returned
          ``prepared`` or raised after partial initialization, even if
          ``start()`` was never invoked.
        * ``cleanup()`` is **not** called when preparation returned
          ``continue`` and ``start()`` was never invoked, or when snapshot
          capture returned ``exit_success``. Engines whose constructors
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
    """Abstract base for token-based inference engines (vLLM, SGLang, TRT-LLM).

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
        """Engine-declared logits-processor activation. Default returns
        ``None`` (no engine-level processors).

        Subclasses override to return a :class:`LogitsProcessorSpec` whose
        ``entries`` are backend-neutral activation data. Unlike
        framework-consumed hooks (:meth:`kv_event_sources`,
        :meth:`health_check_payload`), the result is consumed by the
        engine's own :meth:`generate`: resolve it once after engine init
        (typically in ``start()``), cache it, and pass it per request to
        :func:`logits_processors_for_request`, which applies the shared
        generation-stage gating.

        Overrides typically delegate to
        :func:`resolve_test_logits_processor_spec` to honour
        ``DYN_ENABLE_TEST_LOGITS_PROCESSOR=1``; the future public
        CLI/config loader will resolve from that source instead."""
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
# Custom logits processors: backend-neutral spec + per-backend realization
#
# An engine declares a `LogitsProcessorSpec` of entries; each backend realizes
# them through its own attach function (entry shapes and tokenizer-init forcing
# are backend-shaped, so this layer mandates neither). `logits_processors_for_request`
# owns the two backend-agnostic policies: build fresh per-request state (stateful
# processors can't be shared across concurrent requests), and skip non-generation
# workers (PREFILL/ENCODE don't emit the visible stream; a processor there would
# corrupt or waste leading state).
# ---------------------------------------------------------------------------


#: Env var that activates the built-in smoke hook on any unified backend.
DYN_ENABLE_TEST_LOGITS_PROCESSOR = "DYN_ENABLE_TEST_LOGITS_PROCESSOR"


@dataclass(frozen=True)
class ForcedTokenSequenceSpec:
    """Force the next ``len(token_ids)`` outputs, then EOS thereafter.

    Token IDs are pre-resolved at startup (no per-request tokenizer access)
    and JSON-serializable, so the entry survives any worker boundary. A
    tuple, so the cached spec stays immutable.
    """

    token_ids: tuple[int, ...]
    eos_token_id: int


@dataclass(frozen=True)
class PythonProcessorSpec:
    """Escape hatch for live, in-process `BaseLogitsProcessor` instances.

    Only TRT-LLM consumes it; vLLM/SGLang adapters reject it (not
    serializable). Not used by the env-hook smoke.
    """

    factory: Callable[[], "BaseLogitsProcessor"]


LogitsProcessorEntry = ForcedTokenSequenceSpec | PythonProcessorSpec


@dataclass(frozen=True)
class LogitsProcessorSpec:
    """Engine-declared, backend-neutral logits-processor activation.

    ``generation_only`` defaults True because every entry shipping today is
    stateful and only meaningful on the worker that emits the visible
    stream; a future stateless entry can set it False to run everywhere.
    ``entries`` is a tuple so the cached spec stays immutable.
    """

    entries: tuple[LogitsProcessorEntry, ...]
    generation_only: bool = True


# Disaggregation modes that produce the visible output token stream.
_GENERATION_STAGES = frozenset(
    {DisaggregationMode.AGGREGATED, DisaggregationMode.DECODE}
)


def is_generation_stage(disaggregation_mode: DisaggregationMode) -> bool:
    """True for worker roles that emit the visible output stream
    (AGGREGATED, DECODE). PREFILL/ENCODE return False — a ``generation_only``
    spec never runs there, so a backend can also skip the engine-level setup
    (tokenizer init, spec resolution) for those roles."""
    return disaggregation_mode in _GENERATION_STAGES


def logits_processors_for_request(
    spec: LogitsProcessorSpec | None, *, disaggregation_mode: DisaggregationMode
) -> list[LogitsProcessorEntry]:
    """Return the entries a backend should activate for one request.

    Applies the shared generation-stage gating: empty when ``spec`` is None
    or a ``generation_only`` spec runs on a non-generation worker
    (PREFILL/ENCODE); otherwise ``spec.entries``. Realization is each
    backend adapter's job, and since the same entries flow through every
    request, realizers MUST build fresh per-request state.
    """
    if spec is None:
        return []
    if spec.generation_only and not is_generation_stage(disaggregation_mode):
        return []
    return list(spec.entries)


def resolve_test_logits_processor_spec(
    get_tokenizer: Callable[[], Any],
) -> LogitsProcessorSpec | None:
    """Resolve the `DYN_ENABLE_TEST_LOGITS_PROCESSOR=1` smoke hook into a
    `LogitsProcessorSpec`, or ``None`` when the env var is unset.

    ``get_tokenizer`` is called lazily, only after the env check passes, so
    an engine started with ``skip_tokenizer_init=True`` and the hook off
    does not crash. The fixed ``"Hello world!"`` token IDs are resolved here
    once (not per request) into a single :class:`ForcedTokenSequenceSpec`.
    """
    if os.getenv(DYN_ENABLE_TEST_LOGITS_PROCESSOR) != "1":
        return None
    tokenizer = get_tokenizer()
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError(
            "DYN_ENABLE_TEST_LOGITS_PROCESSOR requires a tokenizer with eos_token_id"
        )
    token_ids = tuple(tokenizer.encode("Hello world!", add_special_tokens=False))
    return LogitsProcessorSpec(
        entries=(ForcedTokenSequenceSpec(token_ids=token_ids, eos_token_id=eos),),
        generation_only=True,
    )


# ---------------------------------------------------------------------------
# Wire format for backends that activate processors across a serialization
# boundary.
#
# TRT-LLM attaches live Python callables in-process, so it never serializes.
# vLLM and SGLang both load a batch-level adapter class at engine init and
# carry per-request activation through a JSON-ish side channel
# (`SamplingParams.extra_args` for vLLM, `sampling_params["custom_params"]`
# for SGLang). Both cross that boundary with the SAME entry shape, so the
# format lives here once rather than in each backend adapter.
#
# Only JSON-safe entries are serializable: `ForcedTokenSequenceSpec` carries
# plain ints. `PythonProcessorSpec` wraps a live callable and is rejected —
# it is a TRT-LLM-only in-process escape hatch (the docstring on the class
# says as much), so a backend that needs serialization cannot realize it.
# ---------------------------------------------------------------------------


#: Discriminator stored on each serialized entry so the reader can pick the
#: right `LogitsProcessorEntry` subtype back out.
_FORCED_SEQUENCE_KIND = "forced_sequence"


def serialize_logits_processor_entries(
    entries: Sequence[LogitsProcessorEntry],
) -> list[dict[str, Any]]:
    """Encode spec entries into JSON-safe dicts for a per-request side channel.

    Used by backends (vLLM, SGLang) whose engine-loaded adapter runs in a
    process that only sees the serialized request, not the engine's cached
    spec. Raises ``TypeError`` on `PythonProcessorSpec` (a live callable is
    not serializable), which is how vLLM/SGLang reject the TRT-LLM-only
    escape hatch.
    """
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
                "serializable; only ForcedTokenSequenceSpec crosses the "
                "vLLM/SGLang request boundary (PythonProcessorSpec is "
                "TRT-LLM-only, in-process)"
            )
    return payload


def deserialize_logits_processor_entries(
    payload: Sequence[dict[str, Any]],
) -> list[LogitsProcessorEntry]:
    """Inverse of :func:`serialize_logits_processor_entries`.

    Runs inside the backend's engine-loaded adapter to rebuild spec entries
    from the per-request payload. Unknown ``kind`` values raise ``ValueError``
    so a forward-incompatible request fails loudly instead of silently
    skipping a processor.
    """
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
