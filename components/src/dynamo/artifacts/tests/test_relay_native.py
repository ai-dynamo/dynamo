# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TCP integration; requires an explicitly configured isolated test master.

Set DYN_TEST_MOONCAKE_MASTER and DYN_TEST_MOONCAKE_METADATA (including /metadata).
No services are started by this test. The optional SDK and its CUDA runtime must
be installed separately. Cleanup retires leases after the test consumer returns.
"""

import os
import uuid
from types import SimpleNamespace

import numpy as np
import pytest

from dynamo.artifacts.relay import MooncakeByteRelay, RelayError
from dynamo.artifacts.vllm.format import (
    GenerationArtifactChoice,
    GenerationArtifactView,
    Tensor,
    decode_generation_artifact,
    encode_generation_artifact,
)

mooncake = pytest.importorskip("mooncake.store", exc_type=ImportError)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.post_merge,
    pytest.mark.timeout(60),
]


@pytest.fixture
def relays():
    master = os.environ.get("DYN_TEST_MOONCAKE_MASTER")
    metadata = os.environ.get("DYN_TEST_MOONCAKE_METADATA")
    if not master or not metadata:
        pytest.skip("an isolated native Mooncake test master was not configured")
    namespace = "relay-tests/" + uuid.uuid4().hex
    clients = []
    try:
        for segment_bytes in (16 << 20, 0):
            store = mooncake.MooncakeDistributedStore()
            result = store.setup(
                "127.0.0.1", metadata, segment_bytes, 16 << 20, "tcp", "", master
            )
            if result != 0:
                store.close()
                pytest.fail("native Mooncake client setup failed")
            clients.append(
                MooncakeByteRelay(store, namespace=namespace, max_bytes=1 << 20)
            )
        yield tuple(clients)
    finally:
        for client in reversed(clients):
            client.close()


@pytest.mark.parametrize("payload_kind", ["opaque", "generation_artifact_v1"])
def test_native_tcp_roundtrip(relays, payload_kind):
    producer, consumer = relays
    if payload_kind == "opaque":
        payload = b"\x00arbitrary\xffbinary\x00" * 4096
    else:
        tokens = Tensor.from_array(np.asarray([1, 2, 3, 4], dtype=np.int64))
        payload = encode_generation_artifact(
            GenerationArtifactView((GenerationArtifactChoice(0, 2, tokens),))
        ).data
    descriptor = producer.publish(payload)
    try:
        received = consumer.read(descriptor)
        assert received == payload
        if payload_kind == "generation_artifact_v1":
            assert decode_generation_artifact(received).choices[
                0
            ].sequence_token_ids.as_array().tolist() == [1, 2, 3, 4]
    finally:
        producer.remove(descriptor, force=True)
    with pytest.raises(RelayError):
        consumer.read(descriptor)


def test_native_put_collision_cannot_claim_wrong_bytes(relays, monkeypatch):
    producer, consumer = relays
    monkeypatch.setattr(
        "dynamo.artifacts.relay.uuid.uuid4", lambda: SimpleNamespace(hex="a" * 32)
    )
    descriptor = producer.publish(b"first")
    try:
        with pytest.raises(RelayError, match="digest"):
            producer.publish(b"other")
        assert consumer.read(descriptor) == b"first"
    finally:
        producer.remove(descriptor, force=True)
