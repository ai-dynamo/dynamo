# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit coverage for GMS layout metadata protocol and snapshots."""

from __future__ import annotations

import base64
import hashlib
import json

import pytest
from _deps import HAS_GMS

if not HAS_GMS:
    pytest.skip(
        "gpu_memory_service package is not available in this test image",
        allow_module_level=True,
    )

from gpu_memory_service.common.protocol.messages import (  # noqa: E402
    LayoutMetadataDeleteRequest,
    LayoutMetadataDeleteResponse,
    LayoutMetadataGetRequest,
    LayoutMetadataGetResponse,
    LayoutMetadataKey,
    LayoutMetadataListRequest,
    LayoutMetadataListResponse,
    LayoutMetadataPutRequest,
    LayoutMetadataPutResponse,
    decode_message,
    encode_message,
)
from gpu_memory_service.server.allocations import AllocationInfo  # noqa: E402
from gpu_memory_service.server.gms import GMS, MetadataEntry  # noqa: E402
from gpu_memory_service.snapshot.disk import (  # noqa: E402
    LAYOUT_METADATA_FILENAME,
    load_layout_metadata,
)
from gpu_memory_service.snapshot.storage_client import GMSStorageClient  # noqa: E402

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.none,
    pytest.mark.gpu_0,
]


@pytest.mark.parametrize(
    "message",
    [
        LayoutMetadataPutRequest("trtllm", "identity", b"\x00\xff"),
        LayoutMetadataPutResponse(success=True),
        LayoutMetadataGetRequest("trtllm", "identity"),
        LayoutMetadataGetResponse(found=True, value=b"payload"),
        LayoutMetadataGetResponse(found=False),
        LayoutMetadataDeleteRequest("trtllm", "identity"),
        LayoutMetadataDeleteResponse(deleted=True),
        LayoutMetadataListRequest(namespace="trtllm", prefix="source"),
        LayoutMetadataListResponse(
            keys=[LayoutMetadataKey(namespace="trtllm", key="source_identity")]
        ),
    ],
)
def test_layout_metadata_protocol_round_trip(message):
    assert decode_message(encode_message(message)) == message


def test_layout_hash_without_layout_metadata_preserves_legacy_algorithm():
    service = object.__new__(GMS)
    service._metadata = {
        "tensor.0": MetadataEntry(
            allocation_id="allocation",
            offset_bytes=16,
            value=b"tensor-spec",
        )
    }
    service._layout_metadata = {}
    allocation = AllocationInfo(
        allocation_id="allocation",
        size=4096,
        aligned_size=4096,
        handle=0,
        export_fd=-1,
        tag="weights",
        layout_slot=0,
        created_at=0.0,
    )
    expected = hashlib.sha256()
    expected.update(b"0:4096:4096:weights")
    expected.update(b"tensor.0:0:16:")
    expected.update(b"tensor-spec")

    assert service._compute_memory_layout_hash([allocation]) == expected.hexdigest()


def test_load_layout_metadata_accepts_legacy_snapshot(tmp_path):
    assert load_layout_metadata(str(tmp_path)) == {}


def test_load_layout_metadata_requires_manifest_declared_sidecar(tmp_path):
    with pytest.raises(FileNotFoundError, match="manifest requires missing"):
        load_layout_metadata(str(tmp_path), required=True)


def test_load_layout_metadata_decodes_namespaced_opaque_bytes(tmp_path):
    payload = {
        "format_version": 1,
        "entries": [
            {
                "namespace": "trtllm",
                "key": "committed_weight_envelope",
                "value": "AP8=",
            },
            {
                "namespace": "vllm",
                "key": "identity",
                "value": "dmFsdWU=",
            },
        ],
    }
    (tmp_path / LAYOUT_METADATA_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    assert load_layout_metadata(str(tmp_path)) == {
        ("trtllm", "committed_weight_envelope"): b"\x00\xff",
        ("vllm", "identity"): b"value",
    }


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"format_version": 2, "entries": []}, "Unsupported.*format_version"),
        (
            {
                "format_version": 1,
                "entries": [
                    {"namespace": "trtllm", "key": "identity", "value": "YQ=="},
                    {"namespace": "trtllm", "key": "identity", "value": "Yg=="},
                ],
            },
            "Duplicate.*namespace='trtllm'.*key='identity'",
        ),
        (
            {
                "format_version": 1,
                "entries": [{"namespace": "trtllm", "key": "identity", "value": "!!!"}],
            },
            "not valid base64",
        ),
    ],
)
def test_load_layout_metadata_rejects_invalid_snapshot(tmp_path, payload, match):
    (tmp_path / LAYOUT_METADATA_FILENAME).write_text(
        json.dumps(payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match=match):
        load_layout_metadata(str(tmp_path))


class _FakeLayoutMetadataManager:
    def __init__(self, values=None):
        self.values = dict(values or {})

    def layout_metadata_list(self):
        return [
            LayoutMetadataKey(namespace=namespace, key=key)
            for namespace, key in sorted(self.values)
        ]

    def layout_metadata_get(self, namespace, key):
        return self.values.get((namespace, key))

    def layout_metadata_put(self, namespace, key, value):
        self.values[(namespace, key)] = value
        return True


def test_storage_client_serializes_and_restores_layout_metadata():
    client = object.__new__(GMSStorageClient)
    source = _FakeLayoutMetadataManager(
        {
            ("trtllm", "identity"): b"\x00\xff",
            ("vllm", "identity"): b"value",
        }
    )

    saved = client._save_layout_metadata(source)
    assert saved == {
        "format_version": 1,
        "entries": [
            {"namespace": "trtllm", "key": "identity", "value": "AP8="},
            {"namespace": "vllm", "key": "identity", "value": "dmFsdWU="},
        ],
    }

    destination = _FakeLayoutMetadataManager()
    decoded = {
        (entry["namespace"], entry["key"]): base64.b64decode(entry["value"])
        for entry in saved["entries"]
    }
    client._restore_layout_metadata(destination, decoded)
    assert destination.values == source.values
