# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight-transport plug-ins for ``dynamo.vllm`` (Phase 1+4 of the
WeightTransferConfig design).

In scope this iteration:

* :class:`FilesystemTransport` — current default, safetensors via shared FS.
* :class:`NcclTransport` — collective broadcast on a pre-formed group
  (vLLM ``collective_rpc("update_weights_from_distributed", …)``).

Future (deferred): ``NixlTransport``, ``ModelExpressTransport``,
``IpcTransport``, plus an ``SglangEngineAdapter`` for the second engine
flavor.
"""

from .base import (
    EngineAdapter,
    InitCtx,
    InitResult,
    TransportState,
    UpdateResult,
    UpdateWeightsRequest,
    WeightTarget,
    WeightTransport,
)
from .engine_adapter import VllmEngineAdapter
from .filesystem import FilesystemTransport
from .nccl import NcclTransport

__all__ = [
    "EngineAdapter",
    "FilesystemTransport",
    "InitCtx",
    "InitResult",
    "NcclTransport",
    "TransportState",
    "UpdateResult",
    "UpdateWeightsRequest",
    "VllmEngineAdapter",
    "WeightTarget",
    "WeightTransport",
    "build_transport",
]


def build_transport(backend: str, engine_adapter, cfg: dict):
    """Factory: instantiate the right transport for the given backend id."""
    if backend == "filesystem":
        return FilesystemTransport(engine_adapter, cfg)
    if backend == "nccl":
        return NcclTransport(engine_adapter, cfg)
    raise ValueError(
        f"Unsupported weight-transport backend '{backend}'. "
        "In-scope this iteration: filesystem, nccl. "
        "Future (deferred): nixl, model_express, ipc."
    )
