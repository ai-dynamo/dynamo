#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests to verify vLLM KV events API compatibility.

These tests check that the vLLM KV events classes have the expected fields
that our Rust deserializers depend on. If vLLM changes their API, these tests
will fail early, before hitting runtime deserialization errors.

This test is the early warning for vLLM KV-event wire-format changes.

In the normal case, if this fails, first check whether
`lib/kv-router/src/zmq_wire.rs` already accepts the new upstream vLLM event
shape. If not, update that compatibility layer before updating this test.

That file is Dynamo's compatibility layer for vLLM KV events:
- it decodes vLLM's msgpack tagged-map wire format and legacy
  `array_like=True` payloads
- it handles field changes in `BlockStored` / `BlockRemoved` / `EventBatch`
- it translates upstream `extra_keys` into Dynamo's internal `block_mm_infos`

Only touch consolidator files if we explicitly need the consolidator publisher
to preserve and republish a new upstream field.
"""

import importlib

import pytest

# Import vllm first to ensure it's properly loaded before accessing submodules.
# This works around potential issues with pytest's import machinery.
_vllm = importlib.import_module("vllm")
_kv_events = importlib.import_module("vllm.distributed.kv_events")

# Re-export the classes we need for tests
BlockStored = _kv_events.BlockStored
BlockRemoved = _kv_events.BlockRemoved
EventBatch = _kv_events.EventBatch
KVCacheEvent = _kv_events.KVCacheEvent

pytestmark = [
    pytest.mark.vllm,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _has_group_idx(event_cls):
    return "group_idx" in event_cls.__struct_fields__


def _has_kv_cache_spec_kind(event_cls):
    return "kv_cache_spec_kind" in event_cls.__struct_fields__


def _has_kv_cache_spec_sliding_window(event_cls):
    return "kv_cache_spec_sliding_window" in event_cls.__struct_fields__


class TestVllmKvEventsApi:
    """Test vLLM KV events API compatibility."""

    def test_block_stored_fields(self):
        """Verify BlockStored has expected fields in expected order.

        The Rust deserializer expects these fields in this exact order:
        1. block_hashes
        2. parent_block_hash
        3. token_ids
        4. block_size
        5. lora_id
        6. medium
        7. lora_name (added in vLLM 0.14.0)
        8. extra_keys (added in vLLM 0.17.0)
        9. group_idx (added for hybrid KV cache groups; optional for older vLLM)
        10. kv_cache_spec_kind (semantic cache type; optional for older vLLM)
        11. kv_cache_spec_sliding_window (semantic cache window; optional for older vLLM)

        If vLLM adds/removes/reorders fields, this test will fail.
        """
        expected_fields = [
            "block_hashes",
            "parent_block_hash",
            "token_ids",
            "block_size",
            "lora_id",
            "medium",
            "lora_name",
            "extra_keys",
        ]
        if _has_group_idx(BlockStored):
            expected_fields.append("group_idx")
        if _has_kv_cache_spec_kind(BlockStored):
            expected_fields.append("kv_cache_spec_kind")
        if _has_kv_cache_spec_sliding_window(BlockStored):
            expected_fields.append("kv_cache_spec_sliding_window")
        expected_fields = tuple(expected_fields)

        actual_fields = BlockStored.__struct_fields__
        assert actual_fields == expected_fields, (
            f"BlockStored fields changed!\n"
            f"Expected: {expected_fields}\n"
            f"Actual:   {actual_fields}\n"
            f"Required follow-up:\n"
            f"  - Update lib/kv-router/src/zmq_wire.rs to match the new BlockStored wire format.\n"
            f"  - Update this test's expected_fields and msgpack shape checks.\n"
            f"  - If needed, add or update a regression test in lib/llm/src/kv_router/publisher.rs."
        )

    def test_block_removed_fields(self):
        """Verify BlockRemoved has expected fields in expected order."""
        expected_fields = [
            "block_hashes",
            "medium",
        ]
        if _has_group_idx(BlockRemoved):
            expected_fields.append("group_idx")
        if _has_kv_cache_spec_kind(BlockRemoved):
            expected_fields.append("kv_cache_spec_kind")
        if _has_kv_cache_spec_sliding_window(BlockRemoved):
            expected_fields.append("kv_cache_spec_sliding_window")
        expected_fields = tuple(expected_fields)

        actual_fields = BlockRemoved.__struct_fields__
        assert actual_fields == expected_fields, (
            f"BlockRemoved fields changed!\n"
            f"Expected: {expected_fields}\n"
            f"Actual:   {actual_fields}\n"
            f"Required follow-up:\n"
            f"  - Update lib/kv-router/src/zmq_wire.rs RawKvEvent::BlockRemoved seq deserializer.\n"
            f"  - Update this test's expected_fields."
        )

    def test_event_batch_fields(self):
        """Verify EventBatch/KVEventBatch has expected fields."""
        expected_fields = (
            "ts",
            "events",
            "data_parallel_rank",
        )

        actual_fields = EventBatch.__struct_fields__
        assert actual_fields == expected_fields, (
            f"EventBatch fields changed!\n"
            f"Expected: {expected_fields}\n"
            f"Actual:   {actual_fields}\n"
            f"Required follow-up:\n"
            f"  - Update lib/kv-router/src/zmq_wire.rs KvEventBatch Deserialize impl.\n"
            f"  - Update subscriber.rs VllmEventBatch tuple if batch field order changes.\n"
            f"  - Update this test's expected_fields."
        )

    def test_kv_cache_event_uses_tagged_map(self):
        """Verify KVCacheEvent uses tagged-map serialization."""
        struct_config = getattr(KVCacheEvent, "__struct_config__", None)
        assert struct_config is not None, "KVCacheEvent is not a msgspec Struct"
        assert struct_config.array_like is False, (
            "KVCacheEvent changed away from tagged-map serialization. "
            "Check lib/kv-router/src/zmq_wire/deserialize.rs compatibility."
        )

    def test_kv_cache_event_uses_tag(self):
        """Verify KVCacheEvent uses tag=True for variant identification.

        The tag is encoded in the msgpack map's 'type' field.
        """
        struct_config = getattr(KVCacheEvent, "__struct_config__", None)
        assert struct_config is not None, "KVCacheEvent is not a msgspec Struct"
        # When tag=True is set, struct_config.tag contains the tag string (class name)
        # or True. A falsy value (None/False) means no tagging.
        assert struct_config.tag, (
            "KVCacheEvent no longer uses tag=True! "
            "This will break Rust deserialization."
        )

    def test_block_stored_serialization_format(self):
        """Verify BlockStored serializes to expected msgpack map format.

        This is the ultimate test - if the serialized format changes,
        Rust deserialization will fail.
        """
        import msgspec

        event_kwargs = {
            "block_hashes": [123, 456],
            "parent_block_hash": 789,
            "token_ids": [1, 2, 3, 4],
            "block_size": 16,
            "lora_id": None,
            "medium": "GPU",
            "lora_name": None,
            "extra_keys": None,
        }
        if _has_group_idx(BlockStored):
            event_kwargs["group_idx"] = 0
        if _has_kv_cache_spec_kind(BlockStored):
            event_kwargs["kv_cache_spec_kind"] = "full_attention"
        if _has_kv_cache_spec_sliding_window(BlockStored):
            event_kwargs["kv_cache_spec_sliding_window"] = 128
        event = BlockStored(**event_kwargs)

        encoded = msgspec.msgpack.encode(event)
        decoded = msgspec.msgpack.decode(encoded)

        assert isinstance(decoded, dict), f"Expected dict, got {type(decoded)}"
        assert decoded["type"] == "BlockStored"
        assert decoded["block_hashes"] == [123, 456]
        assert decoded["parent_block_hash"] == 789
        assert decoded["token_ids"] == [1, 2, 3, 4]
        assert decoded["block_size"] == 16
        assert decoded["lora_id"] is None
        assert decoded["medium"] == "GPU"
        assert decoded["lora_name"] is None
        assert decoded.get("extra_keys") is None
        if _has_group_idx(BlockStored):
            assert decoded["group_idx"] == 0
        if _has_kv_cache_spec_kind(BlockStored):
            assert decoded["kv_cache_spec_kind"] == "full_attention"
        if _has_kv_cache_spec_sliding_window(BlockStored):
            assert decoded["kv_cache_spec_sliding_window"] == 128

    def test_block_stored_tuple_extra_keys_serialization_format(self):
        """Verify multimodal tuple extra_keys keep the vLLM 0.19 wire shape."""
        import msgspec

        mm_hash = "0123456789abcdef00112233445566778899aabbccddeefffedcba9876543210"
        event_kwargs = {
            "block_hashes": [123],
            "parent_block_hash": None,
            "token_ids": [1, 2, 3, 4],
            "block_size": 16,
            "lora_id": None,
            "medium": "GPU",
            "lora_name": None,
            "extra_keys": [((mm_hash, 7),)],
        }
        if _has_group_idx(BlockStored):
            event_kwargs["group_idx"] = 0
        if _has_kv_cache_spec_kind(BlockStored):
            event_kwargs["kv_cache_spec_kind"] = "full_attention"
        if _has_kv_cache_spec_sliding_window(BlockStored):
            event_kwargs["kv_cache_spec_sliding_window"] = 128
        event = BlockStored(**event_kwargs)

        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(event))

        assert decoded["type"] == "BlockStored"
        assert decoded["extra_keys"] == [[[mm_hash, 7]]], (
            "vLLM multimodal extra_keys no longer serialize as nested tuple/list "
            f"payloads. Decoded: {decoded['extra_keys']!r}"
        )
        if _has_group_idx(BlockStored):
            assert decoded["group_idx"] == 0
        if _has_kv_cache_spec_kind(BlockStored):
            assert decoded["kv_cache_spec_kind"] == "full_attention"
        if _has_kv_cache_spec_sliding_window(BlockStored):
            assert decoded["kv_cache_spec_sliding_window"] == 128

    def test_block_removed_serialization_format(self):
        """Verify BlockRemoved serializes to expected msgpack map format."""
        import msgspec

        event_kwargs = {
            "block_hashes": [123, 456],
            "medium": "GPU",
        }
        if _has_group_idx(BlockRemoved):
            event_kwargs["group_idx"] = 0
        if _has_kv_cache_spec_kind(BlockRemoved):
            event_kwargs["kv_cache_spec_kind"] = "full_attention"
        if _has_kv_cache_spec_sliding_window(BlockRemoved):
            event_kwargs["kv_cache_spec_sliding_window"] = 128
        event = BlockRemoved(**event_kwargs)

        decoded = msgspec.msgpack.decode(msgspec.msgpack.encode(event))

        assert decoded["type"] == "BlockRemoved"
        assert decoded["block_hashes"] == [123, 456]
        assert decoded["medium"] == "GPU"
        if _has_group_idx(BlockRemoved):
            assert decoded["group_idx"] == 0
        if _has_kv_cache_spec_kind(BlockRemoved):
            assert decoded["kv_cache_spec_kind"] == "full_attention"
        if _has_kv_cache_spec_sliding_window(BlockRemoved):
            assert (
                decoded["kv_cache_spec_sliding_window"] == 128
            ), "kv_cache_spec_sliding_window has wrong value"
