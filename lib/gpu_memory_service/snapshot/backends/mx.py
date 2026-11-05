# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""ModelExpress transfer backend: restore GMS weights over RDMA from a peer.

A peer that already serves the same weights publishes its committed GMS
allocations through ModelExpress (see ``gpu_memory_service.v1.snapshot.mx_source``),
one byte tensor per allocation, named by creation order. This backend reads
them into the restoring server's allocations with ModelExpress's RDMA P2P
path. Allocation IDs are per process, so the two sides pair allocations by
creation order, which an identical model load reproduces, and every pair must
agree on size. The checkpoint's shard files are not read.

``backend_config`` keys: ``mx_rank`` (the restoring rank), ``mx_source_key``
(which peers serve these weights), ``mx_listen_port`` and optionally
``mx_server_url`` (default: ModelExpress's own environment).
"""

from __future__ import annotations

import logging
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from typing import Any, Dict, Mapping, Sequence

from gpu_memory_service.snapshot.transfer import (
    FileIntegrity,
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)


def allocation_tensor_name(index: int) -> str:
    """ModelExpress name of a rank's ``index``-th weight allocation."""
    return f"gms.{index:05d}"


def source_identity(source_key: str):
    """The ModelExpress identity of GMS weights published under ``source_key``.

    ModelExpress finds peers by a hash of this identity. GMS needs no model
    metadata from it: the key names the weights, typically the checkpoint they
    were captured in, and the restore checks every allocation's size. So the
    key stands in for the model name and every other field keeps its default.
    """
    from modelexpress import p2p_pb2

    try:
        mx_version = pkg_version("modelexpress")
    except PackageNotFoundError:
        mx_version = "0.0.0"
    identity = p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=source_key,
    )
    identity.extra_parameters["memory"] = "gms-v1"
    return identity


class _CudaBytes:
    def __init__(self, ptr: int, size: int):
        self.__cuda_array_interface__ = {
            "shape": (size,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
        }


def byte_tensors(extents: Sequence[tuple[int, int]], device: int) -> Dict[str, Any]:
    """Name each (va, size) extent as a ModelExpress byte tensor."""
    import torch

    return {
        allocation_tensor_name(index): torch.as_tensor(
            _CudaBytes(va, size), device=device
        )
        for index, (va, size) in enumerate(extents)
    }


def mx_load_context(
    config: Mapping[str, Any],
    rank: int,
    device: int,
    tensors: Mapping[str, Any],
    **overrides,
):
    """A ModelExpress load context whose engine is a set of GMS allocations."""
    import torch
    from modelexpress.adapter import EngineAdapter
    from modelexpress.load_strategy.context import LoadContext
    from modelexpress.metadata.client_factory import create_metadata_client

    identity = source_identity(config["mx_source_key"])

    class _AllocationsAdapter(EngineAdapter):
        def build_identity(self):
            return identity

        def get_worker_rank(self) -> int:
            return rank

        def get_global_rank(self) -> int:
            return rank

        def get_device_id(self) -> int:
            return device

        def discover_tensors(self, result):
            return dict(tensors)

        def reinit_for_retry(self, result):
            # The next source overwrites every byte of these preallocated buffers.
            return result

    server_url = config.get("mx_server_url")
    return LoadContext(
        model_config=None,
        load_config=None,
        target_device=torch.device("cuda", device),
        global_rank=rank,
        worker_rank=rank,
        local_rank=device,
        device_id=device,
        identity=identity,
        mx_client=create_metadata_client(worker_rank=rank, server_url=server_url),
        worker_id=f"gms-{uuid.uuid4().hex[:8]}",
        mx_server_url=server_url,
        adapter=_AllocationsAdapter(),
        tensors=dict(tensors),
        **overrides,
    )


def _start_nixl_manager(rank: int, device: int, listen_port: int):
    import torch
    from modelexpress.load_strategy import rdma_strategy  # noqa: F401
    from modelexpress.load_strategy.base import _init_nixl_manager

    torch.cuda.set_device(device)
    # NIXL fetches the source's metadata through the listen thread.
    return _init_nixl_manager(rank, device, "gms-mx", listen_port)


class MXTransferSession:
    """One restore: the artifact's allocations, filled from a peer."""

    def __init__(
        self,
        config: GMSSnapshotConfig,
        sources: Sequence[FileTransferSource],
        manager: Future,
    ):
        self._config = config
        self._sources = list(sources)
        self._manager: Future | None = manager

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        validate_transfer_targets(self._sources, targets, device=self._config.device)
        # The agent's thread imports ModelExpress; importing it here before that
        # finishes can deadlock on Python's per-module import locks.
        manager, self._manager = self._manager.result(), None
        try:
            self._fill(targets, manager)
        finally:
            # UCX keeps pages pinned until its agent goes away.
            manager.shutdown()

    def _fill(self, targets: Mapping[str, GMSTransferTarget], manager) -> None:
        import torch
        from modelexpress.load_strategy.context import LoadResult
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        from torch import nn

        backend = self._config.backend_config
        device = self._config.device
        rank = int(backend["mx_rank"])
        listen_port = int(backend["mx_listen_port"])
        extents = [(targets[s.allocation_id].va, s.byte_count) for s in self._sources]
        tensors = byte_tensors(extents, device)

        class _SameAllocationsRdmaStrategy(RdmaStrategy):
            """Accepts only a source with exactly these allocation sizes."""

            def _scope_source_tensors(self, ctx, source_tensors):
                scoped = super()._scope_source_tensors(ctx, source_tensors)
                theirs = {t.name: t.size for t in scoped}
                ours = {name: t.numel() for name, t in ctx.tensors.items()}
                if theirs != ours:
                    raise RuntimeError(
                        f"source allocations differ: {len(theirs)} vs {len(ours)} allocations, "
                        f"{sum(theirs.values())} vs {sum(ours.values())} bytes"
                    )
                return scoped

            def rollback(self, ctx) -> None:
                # Without an agent, ModelExpress would start one on
                # MX_METADATA_PORT + CUDA ordinal, which other GMS devices or
                # the engine may hold. Restart it on this device's port.
                super().rollback(ctx)
                ctx.nixl_manager = _start_nixl_manager(rank, device, listen_port)

        torch.cuda.set_device(device)
        ctx = mx_load_context(backend, rank, device, tensors, nixl_manager=manager)
        try:
            strategy = _SameAllocationsRdmaStrategy()
            if not strategy.is_available(ctx):
                raise RuntimeError("ModelExpress RDMA P2P is unavailable")
            strategy.load(LoadResult(value=None, model=nn.Module()), ctx)
            torch.cuda.synchronize(device)
        finally:
            # restore() shuts down the first agent. A retry replaced it.
            if ctx.nixl_manager is not None and ctx.nixl_manager is not manager:
                ctx.nixl_manager.shutdown()

    def close(self) -> None:
        if self._manager is not None and self._manager.exception() is None:
            self._manager.result().shutdown()
        self._manager = None


class MXTransferBackend:
    """Restore GMS weights from a peer serving the same weights.

    Importing ModelExpress and starting its NIXL agent take seconds and need
    no weights memory, so they start with the backend, in the background,
    while the loader connects and allocates.
    """

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        missing = {"mx_rank", "mx_source_key", "mx_listen_port"} - set(
            config.backend_config
        )
        if missing:
            raise ValueError(f"ModelExpress transfer backend needs {sorted(missing)}")
        self._config = config
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="gms-mx-nixl")
        self._manager = self._pool.submit(
            _start_nixl_manager,
            int(config.backend_config["mx_rank"]),
            config.device,
            int(config.backend_config["mx_listen_port"]),
        )
        self._started = False

    def start_restore(
        self,
        sources: Sequence[FileTransferSource],
        integrity_by_path: Mapping[str, FileIntegrity] | None = None,
    ) -> MXTransferSession:
        # The shard digests describe files this backend never reads.
        del integrity_by_path
        # One restore per backend: the session consumes the NIXL agent.
        if self._started:
            raise RuntimeError("ModelExpress transfer backend restores once")
        self._started = True
        return MXTransferSession(self._config, sources, self._manager)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        if not self._started and self._manager.exception() is None:
            self._manager.result().shutdown()
