# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt Dynamo's existing NIXL WRITE embedding transport to workflow edges."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Mapping
from typing import Any

from dynamo.common.multimodal.embedding_transfer import (
    NixlWriteEmbeddingReceiver,
    NixlWriteEmbeddingSender,
    TransferRequest,
)
from dynamo.workflow.nixl import EmbeddingTransferRef
from dynamo.workflow.perf import WORKFLOW_PERF_TRACE
from dynamo.workflow.runtime import WorkflowExecutionError

logger = logging.getLogger(__name__)


class NixlWriteTensorCarrier:
    """Publish workflow tensors through the stock NIXL WRITE handshake.

    The wire value is the existing multimodal ``TransferRequest``. The decoder
    owns the pre-registered receive ring; this sender retains each source tensor
    until the decoder's transfer-complete notification arrives.
    """

    def __init__(self, *, sender: Any = None, torch_module: Any = None) -> None:
        if torch_module is None:
            try:
                import torch
            except ImportError as error:
                raise RuntimeError(
                    "NIXL WRITE tensor carrier requires PyTorch"
                ) from error
            torch_module = torch
        self._torch = torch_module
        self._sender = sender or NixlWriteEmbeddingSender()
        self._observers: set[asyncio.Task[None]] = set()

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Mapping[str, Any]:
        references = await self.export_tensor_fanout(tensor, (transfer_id,))
        return references[transfer_id]

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        if not isinstance(tensor, self._torch.Tensor):
            raise WorkflowExecutionError(
                "NIXL WRITE carrier can export torch.Tensor only"
            )
        if not isinstance(transfer_ids, tuple) or not transfer_ids:
            raise WorkflowExecutionError("NIXL tensor export requires transfer ids")
        if any(
            not isinstance(transfer_id, str) or not transfer_id
            for transfer_id in transfer_ids
        ):
            raise WorkflowExecutionError(
                "NIXL tensor transfer ids must be non-empty strings"
            )
        if len(set(transfer_ids)) != len(transfer_ids):
            raise WorkflowExecutionError("NIXL tensor transfer ids must be unique")
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        started_ns = time.perf_counter_ns()
        references: dict[str, Mapping[str, Any]] = {}
        for transfer_id in transfer_ids:
            request, completion = await self._sender.send_embeddings(
                tensor, stage_embeddings=True
            )
            references[transfer_id] = EmbeddingTransferRef(
                shape=tuple(request.embeddings_shape),
                dtype=request.embedding_dtype_str,
                serialized_request=request.serialized_request,
                transfer_id=transfer_id,
            ).to_dict()
            observer = asyncio.create_task(
                self._observe_completion(
                    completion,
                    transfer_id=transfer_id,
                    started_ns=started_ns,
                    tensor_bytes=tensor.numel() * tensor.element_size(),
                )
            )
            self._observers.add(observer)
            observer.add_done_callback(self._observers.discard)

        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        for transfer_id in transfer_ids:
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.write_export",
                transfer_id,
                bytes=tensor.numel() * tensor.element_size(),
                elapsed_ms=elapsed_ms,
                fanout=len(transfer_ids),
            )
        return references

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        del reference
        raise WorkflowExecutionError(
            "NIXL WRITE workflow carrier is sender-only; use the decoder receiver"
        )

    async def _observe_completion(
        self,
        completion: Awaitable[None],
        *,
        transfer_id: str,
        started_ns: int,
        tensor_bytes: int,
    ) -> None:
        try:
            await completion
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("NIXL WRITE transfer %s failed", transfer_id)
        else:
            WORKFLOW_PERF_TRACE.emit(
                logger,
                "nixl.write_complete",
                transfer_id,
                bytes=tensor_bytes,
                elapsed_ms=(time.perf_counter_ns() - started_ns) / 1_000_000,
            )

    async def close(self) -> None:
        if self._observers:
            for observer in tuple(self._observers):
                observer.cancel()
            await asyncio.gather(*tuple(self._observers), return_exceptions=True)
        close = getattr(self._sender, "close", None)
        if callable(close):
            await close()


class NixlWriteTensorReceiverCarrier:
    """Import borrowed tensors from a pre-registered NIXL WRITE receive ring."""

    def __init__(self, *, receiver: Any = None, buffer_size: int | None = None) -> None:
        if receiver is not None and buffer_size is not None:
            raise ValueError("receiver and buffer_size cannot both be provided")
        self._receiver = (
            receiver
            if receiver is not None
            else (
                NixlWriteEmbeddingReceiver()
                if buffer_size is None
                else NixlWriteEmbeddingReceiver(buffer_size=buffer_size)
            )
        )
        self._tensor_ids: dict[int, int] = {}

    async def export_tensor(self, tensor: Any, transfer_id: str) -> Mapping[str, Any]:
        del tensor, transfer_id
        raise WorkflowExecutionError(
            "NIXL WRITE receiver carrier cannot export tensors"
        )

    async def export_tensor_fanout(
        self, tensor: Any, transfer_ids: tuple[str, ...]
    ) -> Mapping[str, Mapping[str, Any]]:
        del tensor, transfer_ids
        raise WorkflowExecutionError(
            "NIXL WRITE receiver carrier cannot export tensors"
        )

    async def import_tensor(self, reference: Mapping[str, Any]) -> Any:
        parsed = EmbeddingTransferRef.from_dict(reference)
        tensor_id, tensor = await self._receiver.receive_embeddings(
            TransferRequest(
                embeddings_shape=list(parsed.shape),
                embedding_dtype_str=parsed.dtype,
                serialized_request=parsed.serialized_request,
            )
        )
        identity = id(tensor)
        if identity in self._tensor_ids:
            self._receiver.release_tensor(tensor_id)
            raise WorkflowExecutionError(
                "NIXL WRITE receiver reused a live tensor view"
            )
        self._tensor_ids[identity] = tensor_id
        return tensor

    def release_imported_tensor(self, tensor: Any) -> None:
        try:
            tensor_id = self._tensor_ids.pop(id(tensor))
        except KeyError as error:
            raise WorkflowExecutionError(
                "NIXL WRITE imported tensor is not live"
            ) from error
        self._receiver.release_tensor(tensor_id)

    async def close(self) -> None:
        for tensor_id in tuple(self._tensor_ids.values()):
            self._receiver.release_tensor(tensor_id)
        self._tensor_ids.clear()
        close = getattr(self._receiver, "close", None)
        if callable(close):
            await close()
