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

``backend_config`` keys: ``mx_rank`` (the restoring rank), ``mx_model``,
``mx_revision`` and ``mx_tensor_parallel_size`` (what the weights are),
``mx_server_url`` and ``mx_listen_port``, and optionally ``mx_nixl_manager``,
a warm ModelExpress NIXL agent with nothing registered, which is consumed.
"""

from __future__ import annotations

import logging
from concurrent.futures import Future
from typing import Any, Dict, Mapping, Sequence

from gpu_memory_service.snapshot.transfer import (
    FileTransferSource,
    GMSSnapshotConfig,
    GMSTransferTarget,
    validate_transfer_targets,
)

logger = logging.getLogger(__name__)


def allocation_tensor_name(index: int) -> str:
    """ModelExpress name of a rank's ``index``-th weight allocation."""
    return f"gms.{index:05d}"


def source_identity(model: str, revision: str, tensor_parallel_size: int):
    """The ModelExpress identity of GMS weights; the same on source and target."""
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as pkg_version

    from modelexpress import p2p_pb2

    try:
        mx_version = pkg_version("modelexpress")
    except PackageNotFoundError:
        mx_version = "0.0.0"
    identity = p2p_pb2.SourceIdentity(
        mx_version=mx_version,
        mx_source_type=p2p_pb2.MX_SOURCE_TYPE_WEIGHTS,
        model_name=model,
        backend_framework=p2p_pb2.BACKEND_FRAMEWORK_VLLM,
        tensor_parallel_size=tensor_parallel_size,
    )
    identity.extra_parameters["memory"] = "gms-v1"
    identity.extra_parameters["revision"] = revision
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
    import uuid

    import torch
    from modelexpress.adapter import EngineAdapter
    from modelexpress.load_strategy.context import LoadContext
    from modelexpress.metadata.client_factory import create_metadata_client

    identity = source_identity(
        config["mx_model"],
        config.get("mx_revision", ""),
        int(config["mx_tensor_parallel_size"]),
    )

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


def _resolve(manager):
    """A NIXL manager, or a future of one from prepare_nixl_manager."""
    return manager.result() if isinstance(manager, Future) else manager


def prepare_nixl_manager(config: Mapping[str, Any], device: int):
    """A NIXL agent for the restoring rank, ready before its allocations exist.

    Importing ModelExpress and bringing up UCX take seconds and need no
    weights memory, so a loader can do both while its server starts and its
    weights are allocated. Pass the result, or a future of it, as
    ``mx_nixl_manager``.
    """
    import torch
    from modelexpress.load_strategy import rdma_strategy  # noqa: F401
    from modelexpress.load_strategy.base import _init_nixl_manager

    torch.cuda.set_device(device)
    return _init_nixl_manager(
        int(config["mx_rank"]), device, "gms-mx", int(config["mx_listen_port"])
    )


class MXTransferSession:
    """One restore: the artifact's allocations, filled from a peer."""

    def __init__(
        self, config: GMSSnapshotConfig, sources: Sequence[FileTransferSource]
    ):
        self._config = config
        self._sources = list(sources)
        self._manager = config.backend_config.pop("mx_nixl_manager", None)

    def restore(self, targets: Mapping[str, GMSTransferTarget]) -> None:
        import torch
        from modelexpress.load_strategy.base import _init_nixl_manager
        from modelexpress.load_strategy.context import LoadResult
        from modelexpress.load_strategy.rdma_strategy import RdmaStrategy
        from torch import nn

        validate_transfer_targets(self._sources, targets, device=self._config.device)
        backend = self._config.backend_config
        device = self._config.device
        rank = int(backend["mx_rank"])
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

        torch.cuda.set_device(device)
        manager, self._manager = _resolve(self._manager), None
        if manager is None:
            # NIXL fetches the source's metadata through the listen thread.
            manager = _init_nixl_manager(
                rank, device, "gms-mx", int(backend["mx_listen_port"])
            )
        try:
            ctx = mx_load_context(backend, rank, device, tensors, nixl_manager=manager)
            strategy = _SameAllocationsRdmaStrategy()
            if not strategy.is_available(ctx):
                raise RuntimeError("ModelExpress RDMA P2P is unavailable")
            strategy.load(LoadResult(value=None, model=nn.Module()), ctx)
            torch.cuda.synchronize(device)
        finally:
            # UCX keeps pages pinned until its agent goes away.
            manager.shutdown()

    def close(self) -> None:
        if self._manager is not None:
            _resolve(self._manager).shutdown()
            self._manager = None


class MXTransferBackend:
    """Restore GMS weights from a peer serving the same weights."""

    def __init__(self, *, config: GMSSnapshotConfig) -> None:
        missing = {"mx_rank", "mx_model", "mx_tensor_parallel_size"} - set(
            config.backend_config
        )
        if missing:
            raise ValueError(f"ModelExpress transfer backend needs {sorted(missing)}")
        self._config = config

    def start_restore(self, sources: Sequence[FileTransferSource]) -> MXTransferSession:
        return MXTransferSession(self._config, sources)

    def close(self) -> None:
        pass
