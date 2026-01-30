#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests to verify vLLM KV events API compatibility.

These tests check that the vLLM KV events classes have the expected fields
that our Rust deserializers depend on. If vLLM changes their API, these tests
will fail early, before hitting runtime deserialization errors.

The Rust code in kv_router/publisher.rs and kv_consolidator/subscriber.rs
deserializes vLLM's msgpack-encoded KV events. Since vLLM uses msgspec with
array_like=True, the field ORDER matters - fields are serialized positionally.
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
]


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

        If vLLM adds/removes/reorders fields, this test will fail.
        """
        expected_fields = (
            "block_hashes",
            "parent_block_hash",
            "token_ids",
            "block_size",
            "lora_id",
            "medium",
            "lora_name",
        )

        actual_fields = BlockStored.__struct_fields__
        assert actual_fields == expected_fields, (
            f"BlockStored fields changed!\n"
            f"Expected: {expected_fields}\n"
            f"Actual:   {actual_fields}\n"
            f"If vLLM changed the API, update the Rust deserializers in:\n"
            f"  - lib/llm/src/kv_router/publisher.rs (RawKvEvent::BlockStored)\n"
            f"  - lib/llm/src/block_manager/kv_consolidator/subscriber.rs (VllmRawEvent::BlockStored)"
        )

    def test_block_removed_fields(self):
        """Verify BlockRemoved has expected fields in expected order."""
        expected_fields = (
            "block_hashes",
            "medium",
        )

        actual_fields = BlockRemoved.__struct_fields__
        assert actual_fields == expected_fields, (
            f"BlockRemoved fields changed!\n"
            f"Expected: {expected_fields}\n"
            f"Actual:   {actual_fields}\n"
            f"If vLLM changed the API, update the Rust deserializers."
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
            f"Actual:   {actual_fields}"
        )

    def test_kv_cache_event_uses_array_like(self):
        """Verify KVCacheEvent uses array_like=True serialization.

        Our Rust deserializers expect msgpack arrays, not objects.
        If this changes, deserialization will break.
        """
        # msgspec structs with array_like=True have this attribute
        struct_config = getattr(KVCacheEvent, "__struct_config__", None)
        assert struct_config is not None, "KVCacheEvent is not a msgspec Struct"
        assert struct_config.array_like is True, (
            "KVCacheEvent no longer uses array_like=True! "
            "This will break Rust deserialization."
        )

    def test_kv_cache_event_uses_tag(self):
        """Verify KVCacheEvent uses tag=True for variant identification.

        The tag (e.g., 'BlockStored') is the first element in the msgpack array.
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
        """Verify BlockStored serializes to expected msgpack array format.

        This is the ultimate test - if the serialized format changes,
        Rust deserialization will fail.
        """
        import msgspec

        event = BlockStored(
            block_hashes=[123, 456],
            parent_block_hash=789,
            token_ids=[1, 2, 3, 4],
            block_size=16,
            lora_id=None,
            medium="GPU",
            lora_name=None,
        )

        encoded = msgspec.msgpack.encode(event)
        decoded = msgspec.msgpack.decode(encoded)

        # Should be an array with tag as first element
        assert isinstance(decoded, list), f"Expected list, got {type(decoded)}"
        assert (
            decoded[0] == "BlockStored"
        ), f"Expected tag 'BlockStored', got {decoded[0]}"

        # Verify field count (tag + 7 fields = 8 elements)
        assert len(decoded) == 8, (
            f"Expected 8 elements (tag + 7 fields), got {len(decoded)}.\n"
            f"Decoded: {decoded}\n"
            f"If field count changed, update Rust deserializers."
        )

        # Verify field positions
        assert decoded[1] == [123, 456], f"block_hashes at wrong position: {decoded[1]}"
        assert decoded[2] == 789, f"parent_block_hash at wrong position: {decoded[2]}"
        assert decoded[3] == [1, 2, 3, 4], f"token_ids at wrong position: {decoded[3]}"
        assert decoded[4] == 16, f"block_size at wrong position: {decoded[4]}"
        assert decoded[5] is None, f"lora_id at wrong position: {decoded[5]}"
        assert decoded[6] == "GPU", f"medium at wrong position: {decoded[6]}"
        assert decoded[7] is None, f"lora_name at wrong position: {decoded[7]}"
