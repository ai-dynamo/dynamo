# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import base64
import hashlib
import json
import subprocess
import sys
import time
import zlib
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
import torch

from dynamo import nixl_connect as api
from dynamo.common.multimodal import custom_tensor_nixl as transport
from dynamo.common.multimodal import embedding_transfer
from dynamo.common.multimodal.custom_tensor import (
    TENSOR_KEY,
    NixlTensor,
    resolve_custom_tensors,
    resolve_custom_tensors_async,
)

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture
def reference(monkeypatch):
    monkeypatch.setattr(transport, "_api", lambda: api)
    # Real descriptor schema and validation, without registration or native I/O.
    data = torch.ones(4)
    descriptor = api.Descriptor(data)
    return NixlTensor(
        version=1,
        transport="nixl",
        dtype="float32",
        shape=[4],
        order="C",
        size=16,
        worker_id=7,
        expires_at=time.time() + 30,
        sha256="0" * 64,
        serialized_request=api.RdmaMetadata(
            descriptors=[descriptor.metadata],
            operation_kind=int(api.OperationKind.READ),
            notification_key="request-key",
            nixl_metadata="b64:" + base64.b64encode(zlib.compress(b"agent")).decode(),
        ).model_dump(),
    )


@pytest.mark.parametrize(
    "change", ["expired", "worker", "size", "write", "empty", "compressed-bomb"]
)
def test_descriptor_admission_precedes_registration(reference, change):
    if change == "expired":
        reference.expires_at = 0.0
    elif change == "worker":
        reference.worker_id = 8
    elif change == "size":
        reference.size = 4
    elif change == "write":
        reference.serialized_request["operation_kind"] = int(api.OperationKind.WRITE)
    elif change == "empty":
        reference.serialized_request["descriptors"] = []
    else:
        reference.serialized_request["nixl_metadata"] = (
            "b64:"
            + base64.b64encode(
                zlib.compress(b"x" * (transport.MAX_AGENT_METADATA + 1))
            ).decode()
        )
    with pytest.raises(ValueError):
        transport.validate_nixl_reference(reference, 16, 7)


def test_request_scoped_descriptor_cannot_be_repeated(reference, monkeypatch):
    read = Mock(side_effect=AssertionError("must reject before transfer"))
    monkeypatch.setattr(transport, "read_nixl_tensor", read)
    envelope = {TENSOR_KEY: reference.model_dump()}
    with pytest.raises(ValueError, match="cannot be repeated"):
        resolve_custom_tensors({"a": envelope, "b": [envelope]}, worker_id=7)
    read.assert_not_called()


def test_nixl_unavailable_is_not_advertised(monkeypatch):
    def unavailable():
        raise ValueError("custom NIXL tensor transport is unavailable")

    monkeypatch.setattr(transport, "_api", unavailable)
    assert not transport.nixl_available()


def test_native_backend_errors_have_bounded_client_diagnostics(reference, monkeypatch):
    bindings = pytest.importorskip("nixl._bindings")

    async def fail(*args):
        raise bindings.nixlBackendError("backend-specific peer details")

    receiver = SimpleNamespace(receive_embeddings=fail)
    monkeypatch.setattr(
        embedding_transfer, "NixlReadEmbeddingReceiver", lambda **kw: receiver
    )
    with pytest.raises(
        ValueError, match="custom NIXL tensor transfer failed or expired"
    ):
        transport.read_nixl_tensor(reference, 16, 7)


def test_receiver_copies_before_returning_upstream_buffer(reference, monkeypatch):
    source = torch.ones(4)
    expected = source.numpy().tobytes()
    reference.sha256 = hashlib.sha256(expected).hexdigest()
    receive = AsyncMock(return_value=(42, source))
    release = Mock(side_effect=lambda tensor_id: source.zero_())
    receiver = SimpleNamespace(receive_embeddings=receive, release_tensor=release)
    factory = Mock(return_value=receiver)
    monkeypatch.setattr(embedding_transfer, "NixlReadEmbeddingReceiver", factory)
    data = transport.read_nixl_tensor(reference, 16, 7)
    factory.assert_called_once_with(max_items=0)
    request = receive.call_args.args[0]
    assert request.embeddings_shape == [16]
    assert request.embedding_dtype_str == "uint8"
    assert request.serialized_request == reference.serialized_request
    release.assert_called_once_with(42)
    assert data == expected
    assert torch.count_nonzero(source) == 0


@pytest.mark.asyncio
async def test_publisher_cancellation_retains_upstream_completion(monkeypatch):
    complete = asyncio.Event()
    released = Mock()

    async def wait():
        await complete.wait()
        released()

    sender = SimpleNamespace(
        send_embeddings=AsyncMock(
            return_value=(SimpleNamespace(serialized_request={}), wait())
        )
    )
    monkeypatch.setattr(transport, "_api", lambda: api)
    publisher = transport.NixlTensorPublisher(torch.ones(4), worker_id=7, sender=sender)
    await publisher.__aenter__()
    publisher.envelope()
    cleanup = asyncio.create_task(publisher.__aexit__())
    await asyncio.sleep(0)
    cleanup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cleanup
    released.assert_not_called()
    assert len(transport._CLEANUPS) == 1
    # Expiry prevents admission; it never authorizes active transfer cleanup.
    publisher._spec.expires_at = 0
    await asyncio.sleep(0)
    released.assert_not_called()
    complete.set()
    await asyncio.gather(*transport._CLEANUPS)
    released.assert_called_once()
    assert sender.send_embeddings.call_args.kwargs == {"stage_embeddings": True}


@pytest.mark.asyncio
async def test_unpublished_reference_closes_unused_upstream_waiter(monkeypatch):
    async def wait():
        raise AssertionError("unpublished descriptor must never wait for READ")

    completion = wait()
    sender = SimpleNamespace(
        send_embeddings=AsyncMock(
            return_value=(SimpleNamespace(serialized_request={}), completion)
        )
    )
    monkeypatch.setattr(transport, "_api", lambda: api)
    async with transport.NixlTensorPublisher(torch.ones(4), worker_id=7, sender=sender):
        pass
    assert completion.cr_frame is None


@pytest.mark.asyncio
async def test_failed_http_request_does_not_wait_for_unstarted_read(monkeypatch):
    complete = asyncio.Event()

    async def wait():
        await complete.wait()

    sender = SimpleNamespace(
        send_embeddings=AsyncMock(
            return_value=(SimpleNamespace(serialized_request={}), wait())
        )
    )
    monkeypatch.setattr(transport, "_api", lambda: api)
    try:
        with pytest.raises(ValueError, match="request rejected"):
            async with transport.NixlTensorPublisher(
                torch.ones(4), worker_id=7, sender=sender
            ) as publisher:
                publisher.envelope()
                raise ValueError("request rejected")
        assert len(transport._CLEANUPS) == 1
    finally:
        complete.set()
        await asyncio.gather(*transport._CLEANUPS)


@pytest.mark.asyncio
async def test_publisher_envelope_cannot_be_reused(monkeypatch):
    async def wait():
        return

    sender = SimpleNamespace(
        send_embeddings=AsyncMock(
            return_value=(SimpleNamespace(serialized_request={}), wait())
        )
    )
    monkeypatch.setattr(transport, "_api", lambda: api)
    async with transport.NixlTensorPublisher(
        torch.ones(4), worker_id=7, sender=sender
    ) as publisher:
        publisher.envelope()
        with pytest.raises(ValueError, match="only one request"):
            publisher.envelope()


@pytest.mark.integration
@pytest.mark.timeout(60)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("scalar", [False, True])
async def test_native_nixl_resolution_and_owned_lifetime(dtype, scalar):
    if not transport.nixl_available():
        pytest.skip("requires NIXL bindings")
    source = (
        torch.ones((), dtype=dtype)
        if scalar
        else torch.ones(4096, dtype=dtype).reshape(64, 64).t()
    )
    async with transport.NixlTensorPublisher(source, worker_id=7, ttl=10) as publisher:
        envelope = publisher.envelope()
        assert "data_base64" not in json.dumps(envelope)
        result = await resolve_custom_tensors_async(envelope, worker_id=7)
        assert torch.equal(result, source)
        assert result.dtype == source.dtype and result.shape == source.shape
    assert torch.equal(result, source)  # descriptor/source registration released


@pytest.mark.integration
@pytest.mark.timeout(60)
@pytest.mark.asyncio
async def test_native_dead_producer_rejected_without_unraisable_cleanup_errors():
    if not transport.nixl_available():
        pytest.skip("requires NIXL bindings")
    # A crash cannot send a completion/abort notification or run Python cleanup.
    code = """
import asyncio, json, os, torch
from dynamo.common.multimodal.custom_tensor_nixl import NixlTensorPublisher
async def main():
    publisher = NixlTensorPublisher(torch.ones(4096), worker_id=7, ttl=30)
    await publisher.__aenter__()
    print('REFERENCE=' + json.dumps(publisher.envelope()), flush=True)
    os._exit(0)
asyncio.run(main())
"""
    producer = await asyncio.to_thread(
        subprocess.run,
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    envelope = json.loads(
        next(
            line.removeprefix("REFERENCE=")
            for line in producer.stdout.splitlines()
            if line.startswith("REFERENCE=")
        )
    )
    with pytest.raises(
        ValueError, match="custom NIXL tensor transfer failed or expired"
    ):
        await resolve_custom_tensors_async(envelope, worker_id=7)
