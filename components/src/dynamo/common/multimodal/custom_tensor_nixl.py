# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Raw Custom tensor adapter over the maintained embedding NIXL READ path.

No vLLM object codec, domain interpretation, or reusable reference registry.
NixlReadEmbeddingSender/Receiver own registration, transfer handles, completion
and descriptor lifetime. Expiry is an admission check; completion ownership
follows the existing embedding path.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import math
import time
import zlib
from typing import Any

import torch

from .custom_tensor import TENSOR_KEY, NixlTensor, _tensor_bytes

MAX_AGENT_METADATA = 64 * 1024
MAX_REFERENCE_TTL = 3600.0
_CLEANUPS: set[asyncio.Task] = set()
logger = logging.getLogger(__name__)


def _cleanup_done(task: asyncio.Task) -> None:
    _CLEANUPS.discard(task)
    # A cancelled HTTP caller no longer awaits this task. Retrieve and report
    # its error rather than dropping an unobserved cleanup failure.
    if not task.cancelled() and task.exception() is not None:
        logger.warning(
            "Custom NIXL registration cleanup failed", exc_info=task.exception()
        )


def nixl_available() -> bool:
    try:
        _api()
    except ValueError:
        return False
    return True


def _api():
    try:
        import nixl._api  # noqa: F401 -- nixl_connect itself defers missing bindings

        from dynamo import nixl_connect
    except ImportError as exc:
        raise ValueError("custom NIXL tensor transport is unavailable") from exc
    return nixl_connect


def _embedding_transfer():
    # Optional NIXL adapter: importing this module must not require backend
    # bindings on clients that use only JSON, inline, or SHM inputs.
    from . import embedding_transfer

    return embedding_transfer


def validate_nixl_reference(spec: NixlTensor, size: int, worker_id: int | None):
    """Validate public metadata without registering agents or allocating buffers."""
    api = _api()
    if spec.worker_id != worker_id:
        raise ValueError("NIXL tensor requires its pinned worker")
    if time.time() >= spec.expires_at:
        raise ValueError("custom NIXL tensor reference expired")
    if spec.size != size:
        raise ValueError("tensor reference size does not match shape and dtype")
    if len(json.dumps(spec.serialized_request)) > MAX_AGENT_METADATA:
        raise ValueError("NIXL transfer descriptor is too large")
    try:
        metadata = api.RdmaMetadata.model_validate(spec.serialized_request)
        if metadata.operation_kind != api.OperationKind.READ.value:
            raise ValueError("NIXL tensor requires a READ operation")
        remote = metadata.to_descriptors()
        if isinstance(remote, list) or remote.ptr == 0 or remote.size != size:
            raise ValueError("NIXL tensor requires one matching nonempty descriptor")
        if remote.device.kind != api.DeviceKind.HOST:
            raise ValueError("custom NIXL tensors require CPU staging")
        if not metadata.notification_key or len(metadata.notification_key) > 128:
            raise ValueError("NIXL tensor requires a completion key")
        # Bound expansion before using Remote.decode_metadata's internal codec
        # on untrusted descriptors received through HTTP.
        encoded = metadata.nixl_metadata
        compressed = (
            base64.b64decode(encoded[4:], validate=True)
            if encoded.startswith("b64:")
            else bytes.fromhex(encoded)
        )
        decoder = zlib.decompressobj()
        decoded = decoder.decompress(compressed, MAX_AGENT_METADATA + 1)
        if (
            not decoded
            or len(decoded) > MAX_AGENT_METADATA
            or not decoder.eof
            or decoder.unused_data
        ):
            raise ValueError("invalid NIXL agent metadata")
    except (ValueError, TypeError, zlib.error) as exc:
        raise ValueError("invalid custom NIXL transfer descriptor") from exc
    return metadata


def read_nixl_tensor(spec: NixlTensor, size: int, worker_id: int | None) -> bytearray:
    """Run only in the bounded resolver thread, never on the worker event loop."""
    api = _api()
    metadata = validate_nixl_reference(spec, size, worker_id)

    async def read():
        transfer = _embedding_transfer()
        # Reuse the actual maintained receiver. A raw request has a known byte
        # budget and does not need the embedding service's large warm pool.
        receiver = transfer.NixlReadEmbeddingReceiver(max_items=0)
        request = transfer.TransferRequest(
            embeddings_shape=[size],
            embedding_dtype_str="uint8",
            serialized_request=metadata.model_dump(),
        )
        tensor_id, tensor = await receiver.receive_embeddings(request)
        try:
            # Copy before returning the receiver's buffer, just as processed
            # mm_kwargs copies out of its native transfer storage.
            view = tensor.reshape(-1).view(torch.uint8).numpy()
            data = bytearray()
            for offset in range(0, size, 1024 * 1024):
                data.extend(view[offset : offset + 1024 * 1024])
                time.sleep(0)
            return data
        finally:
            receiver.release_tensor(tensor_id)

    try:
        data = asyncio.run(read())
    except (
        RuntimeError,
        TimeoutError,
        ImportError,
        api.nixl_bindings.nixlBackendError,
        api.nixl_bindings.nixlRemoteDisconnectError,
        api.nixl_bindings.nixlNotFoundError,
        api.nixl_bindings.nixlInvalidParamError,
        api.nixl_bindings.nixlMismatchError,
        api.nixl_bindings.nixlCancelledError,
        api.nixl_bindings.nixlNotSupportedError,
    ) as exc:
        raise ValueError("custom NIXL tensor transfer failed or expired") from exc
    if hashlib.sha256(data).hexdigest() != spec.sha256:
        raise ValueError("tensor reference checksum mismatch")
    return data


class NixlTensorPublisher:
    """Keep registered CPU bytes alive for one HTTP request and one worker READ.

    Enter once, send envelope once, await HTTP response inside the async context.
    Each retry needs a fresh publisher. Pass the service's existing
    NixlReadEmbeddingSender to reuse its connector across requests. Completion
    ownership follows the encoder's send-completion queue: cancellation of the
    caller does not cancel the completion waiter or free the registered source.

    Missing-reader and native failure cleanup follow the maintained embedding
    transport; this adapter does not introduce a separate abort protocol.
    """

    def __init__(
        self, tensor: torch.Tensor, *, worker_id: int, ttl: float = 60, sender=None
    ):
        if type(worker_id) is not int or not 0 <= worker_id < 2**64:
            raise ValueError("NIXL tensor requires a valid worker ID")
        if not math.isfinite(ttl) or not 0 < ttl <= MAX_REFERENCE_TTL:
            raise ValueError("NIXL reference TTL must be in (0, 3600] seconds")
        self._source = tensor
        self._worker_id = worker_id
        self._ttl = ttl
        self._sender = sender
        self._completion = None
        self._completion_task = None
        self._entered = False
        self._sent = False
        self._spec = None

    async def __aenter__(self):
        if self._entered:
            raise ValueError("NIXL publisher is request-scoped and cannot be reused")
        self._entered = True
        _api()
        spec, data = await asyncio.to_thread(_tensor_bytes, self._source)
        self._source = None
        if not data:
            raise ValueError(
                "NIXL tensor must be nonempty; use inline for empty tensors"
            )
        # Explicit CPU staging, registration and network READ; not zero-copy GPU.
        buffer = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        sender = self._sender
        if sender is None:
            sender = _embedding_transfer().NixlReadEmbeddingSender()
        request, self._completion = await sender.send_embeddings(
            buffer, stage_embeddings=True
        )
        try:
            self._spec = NixlTensor(
                **spec,
                transport="nixl",
                size=len(data),
                worker_id=self._worker_id,
                expires_at=time.time() + self._ttl,
                sha256=hashlib.sha256(data).hexdigest(),
                serialized_request=request.serialized_request,
            )
        except BaseException:
            # Metadata was never handed to a reader. Closing the unused
            # upstream coroutine drops its ownership safely in this case.
            self._completion.close()
            self._completion = None
            raise
        return self

    def envelope(self) -> dict[str, Any]:
        if self._completion is None or self._spec is None or self._sent:
            raise ValueError(
                "NIXL publisher must be active and used for only one request"
            )
        if time.time() >= self._spec.expires_at:
            raise ValueError("custom NIXL tensor reference expired")
        self._sent = True
        self._completion_task = asyncio.create_task(self._completion)
        _CLEANUPS.add(self._completion_task)
        self._completion_task.add_done_callback(_cleanup_done)
        return {TENSOR_KEY: self._spec.model_dump()}

    async def __aexit__(self, exc_type=None, exc_value=None, traceback=None):
        if self._completion is None:
            return
        if not self._sent:
            self._completion.close()
            self._completion = None
            return
        if exc_type is None:
            await asyncio.shield(self._completion_task)
