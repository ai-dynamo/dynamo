# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Framework-agnostic base classes for the KVBM v2 connector.

These bases hold the Rust objects and expose thin delegations for all
operations that are identical across frameworks (vLLM, TRT-LLM, …).
Framework-specific code — constructor arguments, scheduler-output
translation, KV-cache tensor layout, and layer-name resolution — lives
entirely in the subclass.

Hierarchy
---------
KvbmConnectorLeaderBase
    vLLM:   SchedulerConnectorLeader   (kvbm.v2.vllm.schedulers.leader)
    TRT-LLM: TrtllmConnectorLeader     (tensorrt_llm._torch.pyexecutor.kvbm_connector)

KvbmConnectorWorkerBase
    vLLM:   SchedulerConnectorWorker   (kvbm.v2.vllm.schedulers.worker)
    TRT-LLM: TrtllmConnectorWorker     (tensorrt_llm._torch.pyexecutor.kvbm_connector)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import kvbm

if kvbm.v2.is_available():
    _KvbmRuntime = kvbm.v2.KvbmRuntime
    _ConnectorLeader = kvbm.v2.ConnectorLeader
    _ConnectorWorker = kvbm.v2.ConnectorWorker
    _KvbmRequest = kvbm.v2.KvbmRequest
else:
    _KvbmRuntime = None
    _ConnectorLeader = None
    _ConnectorWorker = None
    _KvbmRequest = None


@dataclass
class VeloPeerMetadata:
    """
    Velo peer info exported by a worker for the leader handshake.

    Maps directly to ``velo::PeerInfo { instance_id, worker_address }`` on
    the Rust side.  The leader consumes these in ``register_worker`` and
    registers the worker with the Velo messenger so subsequent RPCs
    (layout config, connector metadata, offload acks) can be routed.

    Attributes:
        instance_id:    16-byte UUID identifying the worker's Velo instance.
        worker_address: JSON-serialised ``velo::WorkerAddress`` of the peer.
    """

    instance_id: bytes  # 16-byte UUID
    worker_address: bytes  # JSON-serialised velo::WorkerAddress


# ---------------------------------------------------------------------------
# Leader base
# ---------------------------------------------------------------------------


class KvbmConnectorLeaderBase(ABC):
    """
    Framework-agnostic leader base.

    Subclasses must implement:
    - ``__init__``: build ``KvbmRuntime`` and call ``_init_connector(runtime, block_size)``
    - ``build_connector_meta``: translate the framework's scheduler output and
      call ``self._connector.build_connector_metadata(output)``
    - ``_create_slot``: extract request fields and call
      ``self._connector.create_slot(KvbmRequest(...))``

    Everything else is a thin delegation to the Rust ``ConnectorLeader``.
    """

    def __init__(self) -> None:
        self._runtime: Optional[object] = None
        self._connector: Optional[object] = None
        # Tracks in-flight requests; value is framework-specific request object.
        self._inflight_requests: dict[str, object] = {}
        self._iteration: int = 0
        self.enable_decode_offload: bool = (
            os.getenv("KVBM_DECODE_OFFLOAD", "false") == "true"
        )

    # ------------------------------------------------------------------
    # Initialisation helper (called by subclass __init__)
    # ------------------------------------------------------------------

    def _init_connector(self, runtime: object, block_size: int) -> None:
        """Store the runtime and create the ``ConnectorLeader`` over it."""
        if not kvbm.v2.is_available():
            raise ImportError(
                "KvbmConnectorLeaderBase requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )
        self._runtime = runtime
        self._connector = _ConnectorLeader(runtime, block_size)

    # ------------------------------------------------------------------
    # Handshake — framework-independent
    # ------------------------------------------------------------------

    def set_xfer_handshake_metadata(self, metadata: dict) -> None:
        """
        Register all workers as Velo peers (rank-ordered) and drive init.

        ``metadata`` must be a mapping of ``{rank: VeloPeerMetadata}``.
        Raises ``ValueError`` if ranks are not a consecutive ``0..N-1`` range
        or if any value is not a ``VeloPeerMetadata``.
        """
        sorted_workers = sorted(metadata.items(), key=lambda x: x[0])
        num_workers = len(sorted_workers)
        actual_ranks = [rank for rank, _ in sorted_workers]

        if actual_ranks != list(range(num_workers)):
            raise ValueError(
                f"Expected consecutive ranks 0..{num_workers - 1}, got {actual_ranks}"
            )

        for rank, worker_meta in sorted_workers:
            if not isinstance(worker_meta, VeloPeerMetadata):
                raise ValueError(
                    f"Expected VeloPeerMetadata for rank {rank}, "
                    f"got {type(worker_meta).__name__}"
                )
            self._connector.register_worker(
                rank, worker_meta.instance_id, worker_meta.worker_address
            )

        self._connector.initialize_workers()

    # ------------------------------------------------------------------
    # Per-request API — thin delegations
    # ------------------------------------------------------------------

    def has_slot(self, request_id: str) -> bool:
        return self._connector.has_slot(request_id)

    def get_num_new_matched_tokens(
        self, request_id: str, num_computed_tokens: int
    ) -> tuple[Optional[int], bool]:
        return self._connector.get_num_new_matched_tokens(
            request_id, num_computed_tokens
        )

    def update_state_after_alloc(
        self,
        request_id: str,
        block_ids: list[int],
        num_external_tokens: int,
    ) -> None:
        self._connector.update_state_after_alloc(
            request_id, block_ids, num_external_tokens
        )

    def request_finished(self, request_id: str) -> bool:
        """
        Returns ``True`` if block freeing must be delayed (offload in flight).
        """
        self._inflight_requests.pop(request_id, None)
        return self._connector.request_finished(request_id)

    def update_connector_output(
        self,
        finished_sending: set[str],
        finished_recving: set[str],
    ) -> None:
        self._connector.update_connector_output(
            finished_sending or set(),
            finished_recving or set(),
        )

    # ------------------------------------------------------------------
    # Token-sync helpers (used by subclasses for decode offload)
    # ------------------------------------------------------------------

    def get_slot_total_tokens(self, request_id: str) -> int:
        return self._connector.get_slot_total_tokens(request_id)

    def extend_slot_tokens(self, request_id: str, new_tokens: list[int]) -> None:
        self._connector.extend_slot_tokens(request_id, new_tokens)

    # ------------------------------------------------------------------
    # Abstract interface (framework-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def build_connector_meta(self, scheduler_output: object) -> bytes:
        """
        Build per-iteration connector metadata bytes for the workers.

        Implementations must:
        1. Increment ``self._iteration``.
        2. Optionally call ``_sync_slot_tokens`` for in-flight requests when
           ``self.enable_decode_offload`` is ``True``.
        3. Translate ``scheduler_output`` to a Rust ``SchedulerOutput``.
        4. Call ``bytes(self._connector.build_connector_metadata(rust_output))``
           and return the result.
        """
        ...

    @abstractmethod
    def _create_slot(self, request: object) -> None:
        """
        Create a ``RequestSlot`` in the Rust leader for *request*.

        Implementations must:
        1. Guard with ``self.has_slot(request_id)`` — call
           ``_sync_slot_tokens`` instead of creating a duplicate if it exists.
        2. Build a ``KvbmRequest`` from the framework request object.
        3. Call ``self._connector.create_slot(kv_request)``.
        4. Store ``request`` in ``self._inflight_requests[request_id]``.
        """
        ...

    # ------------------------------------------------------------------
    # Token sync — common helper for subclasses
    # ------------------------------------------------------------------

    def _sync_slot_tokens(self, request_id: str, current_tokens: list[int]) -> None:
        """
        Extend the Rust slot's token sequence if new tokens have been decoded.

        Call this from ``build_connector_meta`` (when ``enable_decode_offload``)
        and from ``_create_slot`` when the slot already exists (resume after
        preemption).

        Args:
            request_id:     The request ID.
            current_tokens: Current full token sequence from the framework.
        """
        if not self._connector.has_slot(request_id):
            return
        slot_count = self._connector.get_slot_total_tokens(request_id)
        request_count = len(current_tokens)
        if slot_count < request_count:
            new_tokens = current_tokens[slot_count:]
            self._connector.extend_slot_tokens(request_id, new_tokens)


# ---------------------------------------------------------------------------
# Worker base
# ---------------------------------------------------------------------------


class KvbmConnectorWorkerBase(ABC):
    """
    Framework-agnostic worker base.

    Subclasses must implement:
    - ``__init__``: build ``KvbmRuntime`` and call ``_init_worker(runtime)``
    - ``register_kv_caches``: accept the framework's KV-cache tensors, extract
      common parameters, and call
      ``self._worker.register_kv_caches(tensors, num_blocks, page_size, dtype_width)``
    - ``save_kv_layer``: resolve the layer index from the framework's layer
      identifier and call
      ``self._worker.save_kv_layer(layer_index, stream_handle)``

    Everything else is a thin delegation to the Rust ``ConnectorWorker``.
    """

    def __init__(self) -> None:
        self._runtime: Optional[object] = None
        self._worker: Optional[object] = None
        self._handshake_metadata: Optional[VeloPeerMetadata] = None

    # ------------------------------------------------------------------
    # Initialisation helper (called by subclass __init__)
    # ------------------------------------------------------------------

    def _init_worker(self, runtime: object) -> None:
        """Store the runtime, create ``ConnectorWorker``, and cache peer info."""
        if not kvbm.v2.is_available():
            raise ImportError(
                "KvbmConnectorWorkerBase requires the 'v2' feature. "
                "Rebuild kvbm with: maturin develop --features v2"
            )
        self._runtime = runtime
        self._worker = _ConnectorWorker(runtime)
        instance_id, worker_addr = runtime.peer_info()
        self._handshake_metadata = VeloPeerMetadata(
            instance_id=instance_id,
            worker_address=worker_addr,
        )

    # ------------------------------------------------------------------
    # Handshake
    # ------------------------------------------------------------------

    def get_handshake_metadata(self) -> VeloPeerMetadata:
        """Return this worker's Velo peer info for the leader handshake."""
        return self._handshake_metadata

    # ------------------------------------------------------------------
    # Per-iteration API — thin delegations
    # ------------------------------------------------------------------

    def bind_connector_metadata(self, data: bytes) -> bool:
        """Bind per-iteration connector metadata from the leader."""
        return self._worker.bind_connector_metadata(data)

    def clear_connector_metadata(self) -> None:
        """Release per-iteration metadata after the forward pass."""
        self._worker.clear_connector_metadata()

    def start_load_kv(self, **kwargs) -> None:
        """Launch intra-pass H2D DMA (if the bound metadata requests it)."""
        self._worker.start_load_kv()

    def wait_for_layer_load(self, layer_index: int, stream_handle: int) -> None:
        """
        Insert ``cudaStreamWaitEvent`` so attention for *layer_index* cannot
        start before its KV slots are populated.
        """
        self._worker.wait_for_layer_load(layer_index, stream_handle)

    def wait_for_save(self) -> None:
        """
        Python-side no-op.

        Forward-pass completion is signalled to the leader asynchronously
        via Velo on the Rust side; there is nothing to block on here.
        """

    def get_finished(self) -> tuple[Optional[set[str]], Optional[set[str]]]:
        """Return ``(finished_sending, finished_recving)`` request ID sets."""
        return self._worker.get_finished()

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs whose intra-pass onboard failed."""
        return self._worker.get_failed_onboarding()

    # ------------------------------------------------------------------
    # Abstract interface (framework-specific)
    # ------------------------------------------------------------------

    @abstractmethod
    def register_kv_caches(self, kv_caches: object) -> None:
        """
        Register the framework's KV cache tensors with NIXL.

        Implementations must extract ``(tensors, num_device_blocks, page_size,
        dtype_width_bytes)`` from the framework-specific ``kv_caches`` argument
        and call:
            ``self._worker.register_kv_caches(tensors, num_blocks, page_size, dtype_width)``
        """
        ...

    @abstractmethod
    def save_kv_layer(self, layer_identifier: object, **kwargs) -> None:
        """
        Notify the Rust worker that the forward pass just finished *layer_identifier*.

        Implementations must resolve a zero-based ``layer_index`` from the
        framework-specific identifier, obtain ``stream_handle`` from the current
        CUDA stream, and call:
            ``self._worker.save_kv_layer(layer_index, stream_handle)``
        """
        ...
