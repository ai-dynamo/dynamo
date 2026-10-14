# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ctypes
import hashlib
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from dynamo.artifacts.relay import MooncakeByteRelay, RelayError

pytestmark = [pytest.mark.unit, pytest.mark.gpu_0, pytest.mark.pre_merge]


@pytest.fixture
def native_store():
    objects = {}
    store = Mock()

    def put(key, data):
        objects.setdefault(key, data)
        return 0

    def get_into(key, address, size):
        data = objects[key]
        if len(data) > size:
            return -600
        ctypes.memmove(address, data, len(data))
        return len(data)

    store.put.side_effect = put
    store.get_size.side_effect = lambda key: (
        len(objects[key]) if key in objects else -704
    )
    store.get_into.side_effect = get_into
    store.register_buffer.return_value = 0
    store.unregister_buffer.return_value = 0
    store.remove.side_effect = lambda key: (
        0 if objects.pop(key, None) is not None else -704
    )
    store.batch_remove.side_effect = lambda keys, force=False: [
        0 if objects.pop(key, None) is not None else -704 for key in keys
    ]
    store.close.return_value = 0
    return store


def test_publish_read_and_remove_opaque_bytes(native_store):
    relay = MooncakeByteRelay(
        native_store, namespace="tenant/operation", max_bytes=1024
    )
    data = b"\x00opaque\xff\x00"
    receipt = relay.publish(data)
    assert receipt.byte_count == len(data)
    assert receipt.sha256 == hashlib.sha256(data).hexdigest()
    assert receipt.key.startswith("tenant/operation/")
    assert relay.read(receipt) == data
    relay.remove(receipt)
    with pytest.raises(RelayError, match="unavailable"):
        relay.read(receipt)
    relay.close()
    native_store.close.assert_called_once()


@pytest.mark.parametrize("data", [b"", b"x" * 33, bytearray(b"mutable")])
def test_publication_rejects_invalid_size_and_mutability(native_store, data):
    relay = MooncakeByteRelay(native_store, namespace="operation", max_bytes=32)
    with pytest.raises(RelayError):
        relay.publish(data)
    native_store.put.assert_not_called()


@pytest.mark.parametrize("namespace", ["", "../a", "/a", "a//b", "a\\b", "a\x00b"])
def test_namespace_is_confined(native_store, namespace):
    with pytest.raises(RelayError):
        MooncakeByteRelay(native_store, namespace=namespace)


def test_reader_uses_descriptor_bound_not_untrusted_store_size(native_store):
    relay = MooncakeByteRelay(native_store, namespace="operation", max_bytes=1024)
    receipt = relay.publish(b"small")
    native_store.get_into.reset_mock()
    native_store.get_size.side_effect = None
    native_store.get_size.return_value = 1 << 40
    with pytest.raises(RelayError, match="length"):
        relay.read(receipt)
    native_store.get_into.assert_not_called()


def test_same_length_corruption_is_rejected(native_store):
    relay = MooncakeByteRelay(native_store, namespace="operation")
    receipt = relay.publish(b"expected")

    def corrupt(key, address, size):
        ctypes.memmove(address, b"tampered", size)
        return size

    native_store.get_into.side_effect = corrupt
    with pytest.raises(RelayError, match="digest"):
        relay.read(receipt)


def test_native_put_success_does_not_bypass_readback(native_store):
    native_store.get_size.side_effect = None
    native_store.get_size.return_value = -704
    relay = MooncakeByteRelay(native_store, namespace="operation")
    with pytest.raises(RelayError):
        relay.publish(b"lost")
    native_store.batch_remove.assert_called_once()


def test_failed_readback_removes_only_newly_published_object(native_store, monkeypatch):
    key = "operation/" + "a" * 32
    monkeypatch.setattr(
        "dynamo.artifacts.relay.uuid.uuid4", lambda: SimpleNamespace(hex="a" * 32)
    )
    relay = MooncakeByteRelay(native_store, namespace="operation")

    def corrupt(_key, address, size):
        ctypes.memmove(address, b"wrong", size)
        return size

    native_store.get_into.side_effect = corrupt
    with pytest.raises(RelayError, match="digest"):
        relay.publish(b"value")
    native_store.batch_remove.assert_called_once_with([key], force=True)
    assert native_store.get_size(key) == -704


def test_existing_key_is_not_overwritten_or_removed(native_store, monkeypatch):
    key = "operation/" + "a" * 32
    native_store.put(key, b"first")
    monkeypatch.setattr(
        "dynamo.artifacts.relay.uuid.uuid4", lambda: SimpleNamespace(hex="a" * 32)
    )
    relay = MooncakeByteRelay(native_store, namespace="operation")
    native_store.put.reset_mock()
    with pytest.raises(RelayError, match="exists"):
        relay.publish(b"other")
    native_store.put.assert_not_called()
    native_store.batch_remove.assert_not_called()
    assert native_store.get_size(key) == 5


def test_force_remove_uses_native_batch_api(native_store):
    relay = MooncakeByteRelay(native_store, namespace="operation")
    receipt = relay.publish(b"value")
    relay.remove(receipt, force=True)
    native_store.batch_remove.assert_called_once_with([receipt.key], force=True)
    native_store.remove.assert_not_called()


def test_receipt_scope_and_digest_validated_before_io(native_store):
    relay = MooncakeByteRelay(native_store, namespace="operation")
    receipt = relay.publish(b"value")
    native_store.get_size.reset_mock()
    for invalid in (
        replace(receipt, key="other/key"),
        replace(receipt, sha256="bad"),
        replace(receipt, byte_count=True),
    ):
        with pytest.raises(RelayError):
            relay.read(invalid)
    native_store.get_size.assert_not_called()


def test_sdk_exceptions_are_sanitized(native_store):
    native_store.put.side_effect = RuntimeError("secret-token")
    relay = MooncakeByteRelay(native_store, namespace="operation")
    with pytest.raises(RelayError) as error:
        relay.publish(b"value")
    assert "secret-token" not in str(error.value)
    assert error.value.__cause__ is None


def test_failed_transfer_unregisters_buffer(native_store):
    relay = MooncakeByteRelay(native_store, namespace="operation")
    receipt = relay.publish(b"value")
    native_store.get_into.return_value = -600
    native_store.get_into.side_effect = None
    native_store.unregister_buffer.reset_mock()
    with pytest.raises(RelayError):
        relay.read(receipt)
    native_store.unregister_buffer.assert_called_once()


def test_failed_unregister_quarantines_buffer_until_store_close(native_store):
    relay = MooncakeByteRelay(native_store, namespace="operation")
    receipt = relay.publish(b"value")
    native_store.unregister_buffer.return_value = -1
    with pytest.raises(RelayError):
        relay.read(receipt)
    native_store.get_size.reset_mock()
    with pytest.raises(RelayError, match="unavailable"):
        relay.read(receipt)
    native_store.get_size.assert_not_called()
    relay.close()
    with pytest.raises(RelayError):
        relay.publish(b"more")
