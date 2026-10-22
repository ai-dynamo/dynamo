# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import subprocess
import sys
import time

import pytest
import torch

from dynamo.common.multimodal.custom_tensor import (
    TENSOR_KEY,
    ShmTensorPublisher,
    inline_tensor,
    resolve_custom_tensors,
)
from dynamo.common.multimodal.shm_bytes import read_shm_bytes

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize(
    "dtype", [torch.float32, torch.bfloat16, torch.float16, torch.int64, torch.bool]
)
@pytest.mark.parametrize("shape", [(2, 3), (0, 3), ()])
def test_inline_reference_equivalence_and_owned_lifetime(dtype, shape):
    source = torch.ones(shape, dtype=dtype)
    inline = resolve_custom_tensors(inline_tensor(source))
    with ShmTensorPublisher(source, worker_id=7) as producer:
        envelope = producer.envelope()
        assert "data_base64" not in json.dumps(envelope)
        reference = resolve_custom_tensors(envelope, worker_id=7)
        assert torch.equal(reference, inline)
        assert reference.dtype == source.dtype
        assert reference.shape == source.shape
        # Multiple readers/retries retain independent storage and item ordering.
        second = resolve_custom_tensors(envelope, worker_id=7)
        assert torch.equal(second, reference)
    assert torch.equal(reference, source)
    with pytest.raises(ValueError, match="released"):
        resolve_custom_tensors(envelope, worker_id=7)


def test_noncontiguous_and_arbitrary_json_preserve_metadata_and_item_boundaries():
    source = torch.arange(12).reshape(3, 4).t()
    fields = {
        "items": [inline_tensor(source), inline_tensor(source + 1)],
        "metadata": [None, True, 1, {"x": "y"}],
    }
    result = resolve_custom_tensors(fields)
    assert torch.equal(result["items"][0], source)
    assert torch.equal(result["items"][1], source + 1)
    assert result["metadata"] == fields["metadata"]
    assert result["items"][0].is_contiguous()


@pytest.mark.parametrize(
    "change,match",
    [
        ({"transport": "nixl"}, "invalid custom tensor descriptor"),
        ({"dtype": "unknown"}, "invalid custom tensor descriptor"),
        ({"shape": [-1]}, "invalid custom tensor descriptor"),
        ({"shape": [True]}, "invalid custom tensor descriptor"),
        ({"order": "F"}, "invalid custom tensor descriptor"),
        ({"shape": [99]}, "byte count"),
        ({"data_base64": "!!!!!!!!"}, "base64"),
    ],
)
def test_invalid_inline_descriptors_are_rejected(change, match):
    envelope = inline_tensor(torch.ones(1))
    envelope[TENSOR_KEY].update(change)
    with pytest.raises(ValueError, match=match):
        resolve_custom_tensors(envelope)


@pytest.mark.parametrize(
    "change,worker,match",
    [
        ({}, None, "pinned worker"),
        ({}, 8, "pinned worker"),
        ({"placement": "other-host"}, 7, "placement mismatch"),
        ({"expires_at": 0.0}, 7, "expired"),
        ({"size": 3}, 7, "size does not match"),
        ({"sha256": "0" * 64}, 7, "checksum"),
        ({"data_base64": "bulk"}, 7, "invalid custom tensor descriptor"),
        ({"name": "../../etc/passwd"}, 7, "invalid custom tensor descriptor"),
    ],
)
def test_invalid_reference_has_no_fallback(change, worker, match):
    with ShmTensorPublisher(torch.ones(1), worker_id=7) as producer:
        envelope = producer.envelope()
        envelope[TENSOR_KEY].update(change)
        with pytest.raises(ValueError, match=match):
            resolve_custom_tensors(envelope, worker_id=worker)


def test_context_exception_releases_allocation():
    with pytest.raises(RuntimeError):
        with ShmTensorPublisher(torch.ones(1), worker_id=7) as producer:
            envelope = producer.envelope()
            raise RuntimeError("request failed")
    with pytest.raises(ValueError, match="released"):
        resolve_custom_tensors(envelope, worker_id=7)
    producer.close()  # idempotent


@pytest.mark.asyncio
async def test_cancellation_releases_allocation():
    producer = ShmTensorPublisher(torch.ones(1), worker_id=7)
    envelope = producer.envelope()

    async def request():
        with producer:
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await request()
    with pytest.raises(ValueError, match="released"):
        resolve_custom_tensors(envelope, worker_id=7)


@pytest.mark.integration
@pytest.mark.timeout(90)
def test_real_producer_to_worker_process_and_retry():
    # Start independent interpreters without depending on pytest's test-module
    # import mode or its resource tracker. These are readers, not model engines.
    script = """
import json, sys
from dynamo.common.multimodal.custom_tensor import resolve_custom_tensors
result = resolve_custom_tensors(json.loads(sys.stdin.read()), worker_id=7)
print(json.dumps([result.tolist(), str(result.dtype), list(result.shape)]))
"""
    with ShmTensorPublisher(torch.arange(6).reshape(2, 3), worker_id=7) as producer:
        envelope = producer.envelope()
        for _ in range(2):
            process = subprocess.run(
                [sys.executable, "-c", script],
                input=json.dumps(envelope),
                capture_output=True,
                text=True,
                timeout=25,
                check=True,
            )
            assert json.loads(process.stdout.strip().splitlines()[-1]) == [
                [[0, 1, 2], [3, 4, 5]],
                "torch.int64",
                [2, 3],
            ]
        # Reader exit must not let resource_tracker unlink producer-owned data.
        assert len(read_shm_bytes(producer.spec.name, producer.spec.size)) == 48


@pytest.mark.integration
@pytest.mark.timeout(10)
def test_abandoned_lease_expires_and_unlinks():
    producer = ShmTensorPublisher(torch.ones(1), worker_id=7, ttl=0.05)
    envelope = producer.envelope()
    deadline = time.monotonic() + 5
    while not producer._closed and time.monotonic() < deadline:
        time.sleep(0.01)
    assert producer._closed
    with pytest.raises(FileNotFoundError):
        read_shm_bytes(producer.spec.name, producer.spec.size)
    with pytest.raises(ValueError, match="expired"):
        resolve_custom_tensors(envelope, worker_id=7)
