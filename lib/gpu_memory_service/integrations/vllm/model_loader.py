# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM model loader for GPU Memory Service integration.

Provides a model loader that loads weights via GMS for cross-process sharing.
The loader uses RW_OR_RO mode: first process loads from disk (RW), subsequent
processes import from GMS metadata (RO).
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from typing import TYPE_CHECKING

import torch
from gpu_memory_service.client.torch.allocator import (
    get_or_create_gms_client_memory_manager,
    gms_use_mem_pool,
)
from gpu_memory_service.client.torch.module import materialize_module_from_gms
from gpu_memory_service.common.locks import GrantedLockType
from gpu_memory_service.common.utils import get_socket_path
from gpu_memory_service.integrations.common.utils import (
    finalize_gms_write,
    get_gms_lock_mode,
    setup_meta_tensor_workaround,
    strip_gms_model_loader_config,
)

if TYPE_CHECKING:
    from gpu_memory_service.client.memory_manager import GMSClientMemoryManager

logger = logging.getLogger(__name__)

# Track imported weights for memory accounting
_last_imported_weights_bytes: int = 0


def get_imported_weights_bytes() -> int:
    """Return bytes of weights imported in the last load_model call."""
    return _last_imported_weights_bytes


# =============================================================================
# MX (ModelExpress) Integration — Optional P2P weight transfer
#
# Uses the refactored modelexpress primitives directly:
#   LoadContext      — state bag (MxClient, NixlTransferManager, identity, ...)
#   register_tensors — collect model tensors + NIXL register
#   publish_metadata — publish to MX server + start heartbeat
# =============================================================================

_mx_ctx = None  # type: LoadContext | None


def get_mx_ctx(
    vllm_config=None,
    model_config=None,
    device_id: int | None = None,
):
    """Get or create the process-global MX LoadContext singleton.

    With no arguments, returns the existing instance (or None).
    When all arguments are provided, creates the singleton on first call.
    Checks MX_ENABLED env var, modelexpress installation, and NIXL
    availability.
    """
    global _mx_ctx
    if _mx_ctx is not None:
        return _mx_ctx

    if vllm_config is None or model_config is None or device_id is None:
        return None

    if os.environ.get("MX_ENABLED", "0") != "1":
        return None

    try:
        from modelexpress.client import MxClient
        from modelexpress.load_strategy import LoadContext
        from modelexpress.metadata import build_source_identity
        from modelexpress.nixl_transfer import is_nixl_available
    except ImportError:
        logger.warning(
            "[GMS-MX] MX_ENABLED=1 but modelexpress is not installed. "
            "Skipping MX integration."
        )
        return None

    if not is_nixl_available():
        logger.warning(
            "[GMS-MX] MX_ENABLED=1 but NIXL is not available. "
            "Skipping MX integration."
        )
        return None

    import uuid

    try:
        import torch.distributed as dist

        global_rank = dist.get_rank() if dist.is_initialized() else device_id
    except Exception:
        global_rank = device_id

    _mx_ctx = LoadContext(
        vllm_config=vllm_config,
        model_config=model_config,
        load_config=vllm_config.load_config,
        target_device=torch.device("cuda", device_id),
        global_rank=global_rank,
        device_id=device_id,
        identity=build_source_identity(vllm_config, model_config),
        mx_client=MxClient(),
        worker_id=uuid.uuid4().hex[:8],
    )
    logger.info(
        "[GMS-MX] Created MX context (rank=%d, device=%d)",
        global_rank,
        device_id,
    )
    return _mx_ctx


def _mx_find_source(ctx):
    """Find a READY RDMA source for this rank.

    Returns (source_worker, mx_source_id) or None.
    """
    import random

    from modelexpress import p2p_pb2

    try:
        resp = ctx.mx_client.list_sources(
            identity=ctx.identity,
            status_filter=p2p_pb2.SOURCE_STATUS_READY,
        )
    except Exception as e:
        logger.warning("[GMS-MX] Error listing sources: %s", e)
        return None

    if not resp.instances:
        return None

    candidates = [i for i in resp.instances if i.worker_rank == ctx.global_rank]
    random.shuffle(candidates)

    for inst in candidates[:3]:
        try:
            meta = ctx.mx_client.get_metadata(inst.mx_source_id, inst.worker_id)
        except Exception as e:
            logger.warning("[GMS-MX] Failed to fetch metadata: %s", e)
            continue
        if meta.found and (meta.worker.tensors or meta.worker.worker_grpc_endpoint):
            logger.info(
                "[GMS-MX] Found source (mx_source_id=%s, worker=%s)",
                inst.mx_source_id,
                inst.worker_id,
            )
            return meta.worker, inst.mx_source_id

    return None


def _mx_receive(ctx, source_worker):
    """RDMA receive from source. Called after register_tensors."""
    from modelexpress.load_strategy import SourceTransferError
    from modelexpress.types import TensorDescriptor

    is_p2p = bool(source_worker.worker_grpc_endpoint)
    remote_agent_name = None

    if is_p2p:
        from modelexpress.worker_server import fetch_tensor_manifest

        tensor_protos = fetch_tensor_manifest(
            endpoint=source_worker.worker_grpc_endpoint,
            mx_source_id=source_worker.mx_source_id
            if hasattr(source_worker, "mx_source_id")
            else "",
        )
        source_tensors = [
            TensorDescriptor(
                name=t.name,
                addr=t.addr,
                size=t.size,
                device_id=t.device_id,
                dtype=t.dtype,
            )
            for t in tensor_protos
        ]
        host, port_str = source_worker.metadata_endpoint.rsplit(":", 1)
        ctx.nixl_manager.fetch_remote_and_wait(
            remote_agent_name=source_worker.agent_name,
            ip=host,
            port=int(port_str),
        )
        remote_agent_name = source_worker.agent_name
    else:
        source_tensors = [
            TensorDescriptor(
                name=t.name,
                addr=t.addr,
                size=t.size,
                device_id=t.device_id,
                dtype=t.dtype,
            )
            for t in source_worker.tensors
        ]

    coalesce = os.environ.get("MX_CONTIGUOUS_REG", "0") == "1"
    try:
        ctx.nixl_manager.receive_from_source(
            source_metadata=source_worker.nixl_metadata,
            source_tensors=source_tensors,
            timeout_seconds=300.0,
            coalesce_transfers=coalesce,
            remote_agent_name=remote_agent_name,
        )
    except Exception as e:
        raise SourceTransferError(f"RDMA receive failed: {e}") from e

    torch.cuda.synchronize()


def register_gms_loader(load_format: str = "gms") -> None:
    """Register the GMS model loader with vLLM's loader registry."""
    from vllm.model_executor.model_loader import register_model_loader
    from vllm.model_executor.model_loader.base_loader import BaseModelLoader
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    @register_model_loader(load_format)
    class GMSModelLoader(BaseModelLoader):
        """vLLM model loader that loads weights via GPU Memory Service."""

        def __init__(self, load_config):
            super().__init__(load_config)
            # Strip GMS-specific keys before creating the fallback loader,
            # otherwise DefaultModelLoader rejects unknown extra config.
            self.default_loader = DefaultModelLoader(
                strip_gms_model_loader_config(
                    load_config,
                    load_format="auto",
                )
            )

        def download_model(self, model_config) -> None:
            self.default_loader.download_model(model_config)

        def load_weights(self, model: torch.nn.Module, model_config) -> None:
            self.default_loader.load_weights(model, model_config)

        def load_model(self, vllm_config, model_config, prefix="") -> torch.nn.Module:
            device = torch.cuda.current_device()
            extra = getattr(self.load_config, "model_loader_extra_config", {}) or {}
            mode = get_gms_lock_mode(extra)
            gms_client = get_or_create_gms_client_memory_manager(
                get_socket_path(device, "weights"),
                device,
                mode=mode,
                tag="weights",
            )

            if gms_client.granted_lock_type == GrantedLockType.RO:
                return _load_read_mode(gms_client, vllm_config, model_config, device)
            else:
                return _load_write_mode(
                    gms_client,
                    vllm_config,
                    model_config,
                    self.default_loader,
                    torch.device("cuda", device),
                )


# =============================================================================
# Helper functions
# =============================================================================


def _load_read_mode(
    gms_client: "GMSClientMemoryManager",
    vllm_config,
    model_config,
    device_index: int,
) -> torch.nn.Module:
    """Load model by importing weights from GMS (RO mode).

    When MX is active, registers materialized tensors with NIXL so this
    node is discoverable as a P2P source (e.g. for shadow engine failover).
    """
    global _last_imported_weights_bytes

    try:
        model = _create_meta_model(vllm_config, model_config)
        materialize_module_from_gms(gms_client, model, device_index=device_index)

        # MX: register materialized tensors (available for P2P transfer)
        mx_ctx = get_mx_ctx(vllm_config, model_config, device_index)
        if mx_ctx is not None:
            from modelexpress.load_strategy import publish_metadata, register_tensors

            try:
                register_tensors(model, mx_ctx)
                publish_metadata(mx_ctx)
            except Exception as e:
                logger.warning("[GMS-MX] Registration failed: %s", e)

        _last_imported_weights_bytes = gms_client.total_bytes
        logger.info(
            "[GMS] Read mode: imported %.2f GiB",
            _last_imported_weights_bytes / (1 << 30),
        )
        return model.eval()
    except Exception:
        gms_client.close()
        raise


def _load_write_mode(
    gms_client: "GMSClientMemoryManager",
    vllm_config,
    model_config,
    default_loader,
    target_device: torch.device,
) -> torch.nn.Module:
    """Load model from disk and publish weights to GMS (RW mode).

    Initializes model using GMS memory pool, loads weights from disk,
    registers tensors with GMS, and commits for cross-process sharing.

    When MX is active, detects whether a P2P source exists:
    - Source (no existing source): loads from disk, registers with NIXL
    - Target (source found): dummy-loads for tensor layout, receives via RDMA
    """
    global _last_imported_weights_bytes

    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    device_id = target_device.index
    mx_ctx = get_mx_ctx(vllm_config, model_config, device_id)
    source_info = _mx_find_source(mx_ctx) if mx_ctx is not None else None

    loader = default_loader
    if source_info is not None:
        from vllm.model_executor.model_loader.dummy_loader import DummyModelLoader

        loader = DummyModelLoader(
            replace(default_loader.load_config, load_format="dummy")
        )

    # Allocate model tensors using GMS memory pool
    with set_default_torch_dtype(model_config.dtype):
        with gms_use_mem_pool("weights", target_device):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config, model_config=model_config
                )

            loader.load_weights(model, model_config)
            process_weights_after_loading(model, model_config, target_device)

            # MX: register with NIXL, optionally receive via RDMA, publish
            if mx_ctx is not None:
                from modelexpress.load_strategy import (
                    publish_metadata,
                    register_tensors,
                )

                register_tensors(model, mx_ctx)
                if source_info is not None:
                    # Target: overwrite dummy weights via RDMA (failure is fatal)
                    source_worker, _ = source_info
                    _mx_receive(mx_ctx, source_worker)
                try:
                    publish_metadata(mx_ctx)
                except Exception as e:
                    logger.warning("[GMS-MX] Publish failed: %s", e)

            torch.cuda.empty_cache()

    _last_imported_weights_bytes = finalize_gms_write(gms_client, model)

    logger.info(
        "[GMS] Write mode: published %.2f GiB",
        _last_imported_weights_bytes / (1 << 30),
    )
    return model.eval()


def _create_meta_model(vllm_config, model_config) -> torch.nn.Module:
    """Create model on meta device for RO mode materialization."""
    from vllm.model_executor.model_loader.utils import (
        initialize_model,
        process_weights_after_loading,
    )
    from vllm.utils.torch_utils import set_default_torch_dtype

    setup_meta_tensor_workaround()
    meta_device = torch.device("meta")

    with set_default_torch_dtype(model_config.dtype):
        with meta_device:
            model = initialize_model(vllm_config=vllm_config, model_config=model_config)

    try:
        process_weights_after_loading(model, model_config, meta_device)
    except Exception as e:
        logger.debug("[GMS] Post-processing on meta tensors: %s", e)

    return model
