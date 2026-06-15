#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests to verify vLLM KV events API compatibility.

These tests check that the vLLM KV events classes have the expected fields
that our Rust deserializers depend on. If vLLM changes their API, these tests
will fail early, before hitting runtime deserialization errors.

This test is the early warning for vLLM KV-event wire-format changes.

In the normal case, if this fails, update `lib/kv-router/src/zmq_wire.rs` to
match the new upstream vLLM event shape, then update this test.

That file is Dynamo's compatibility layer for vLLM KV events:
- it decodes vLLM's msgpack `array_like=True` wire format
- it handles field order changes in `BlockStored` / `BlockRemoved` / `EventBatch`
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


# vLLM's msgspec KVCacheEvent base struct historically used array_like=True,
# which serializes each event as a tagged msgpack array (tag at index 0,
# remaining fields positional). Some vLLM builds drop array_like and instead
# serialize a tagged map (fields keyed by name plus a "type" tag). Dynamo's
# Rust decoder (lib/kv-router/src/zmq_wire/deserialize.rs) accepts BOTH shapes
# via serde's deserialize_any (visit_seq + visit_map), and the wire-format
# README documents both. These helpers let the guard tests validate either
# encoding so they track the real Rust contract rather than one specific shape.


def _event_tag(decoded):
    """Return the event tag regardless of wire shape."""
    if isinstance(decoded, list):
        return decoded[0] if decoded else None
    if isinstance(decoded, dict):
        return decoded.get("type")
    raise AssertionError(f"Unexpected decoded event type: {type(decoded)}")


def _event_field(decoded, array_index, map_key):
    """Read a field from a decoded event regardless of wire shape.

    For the tagged-map shape, missing keys resolve to None: msgspec may omit
    None-valued fields, which is fine because the Rust decoder also treats
    absent optional fields as None.
    """
    if isinstance(decoded, list):
        return decoded[array_index]
    if isinstance(decoded, dict):
        return decoded.get(map_key)
    raise AssertionError(f"Unexpected decoded event type: {type(decoded)}")


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
            f"  - Update this test's expected_fields and msgpack position checks.\n"
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

    def test_kv_cache_event_uses_supported_wire_shape(self):
        """Verify KVCacheEvent serializes to a shape the Rust decoder accepts.

        Dynamo's Rust decoder (lib/kv-router/src/zmq_wire/deserialize.rs) handles
        both msgspec wire shapes via deserialize_any:
          - tagged array (array_like=True): tag at index 0, positional fields
          - tagged map  (array_like=False): fields keyed by name, plus a "type" tag
        The only hard requirement is that the struct is tagged so the decoder can
        discriminate the event variant; array_like may be True or False.
        """
        struct_config = getattr(KVCacheEvent, "__struct_config__", None)
        assert struct_config is not None, "KVCacheEvent is not a msgspec Struct"
        assert struct_config.tag, (
            "KVCacheEvent is no longer tagged! The Rust decoder needs the tag to "
            "discriminate event variants; this will break deserialization."
        )
        assert isinstance(struct_config.array_like, bool), (
            "KVCacheEvent.array_like is not a bool; unexpected msgspec config. "
            f"Got: {struct_config.array_like!r}"
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

        # Either wire shape is accepted (the Rust decoder handles both).
        assert isinstance(
            decoded, (list, dict)
        ), f"Expected list or dict, got {type(decoded)}"
        assert (
            _event_tag(decoded) == "BlockStored"
        ), f"Expected tag 'BlockStored', got {_event_tag(decoded)}"

        if isinstance(decoded, list):
            expected_len = (
                9
                + int(_has_group_idx(BlockStored))
                + int(_has_kv_cache_spec_kind(BlockStored))
                + int(_has_kv_cache_spec_sliding_window(BlockStored))
            )
            assert len(decoded) == expected_len, (
                f"Expected {expected_len} elements, got {len(decoded)}.\n"
                f"Decoded: {decoded}\n"
                f"If field count changed, update Rust deserializers."
            )

        # Verify field values (positional in arrays, by-name in maps)
        assert _event_field(decoded, 1, "block_hashes") == [
            123,
            456,
        ], f"block_hashes wrong: {_event_field(decoded, 1, 'block_hashes')}"
        assert (
            _event_field(decoded, 2, "parent_block_hash") == 789
        ), f"parent_block_hash wrong: {_event_field(decoded, 2, 'parent_block_hash')}"
        assert _event_field(decoded, 3, "token_ids") == [
            1,
            2,
            3,
            4,
        ], f"token_ids wrong: {_event_field(decoded, 3, 'token_ids')}"
        assert (
            _event_field(decoded, 4, "block_size") == 16
        ), f"block_size wrong: {_event_field(decoded, 4, 'block_size')}"
        assert (
            _event_field(decoded, 5, "lora_id") is None
        ), f"lora_id wrong: {_event_field(decoded, 5, 'lora_id')}"
        assert (
            _event_field(decoded, 6, "medium") == "GPU"
        ), f"medium wrong: {_event_field(decoded, 6, 'medium')}"
        assert (
            _event_field(decoded, 7, "lora_name") is None
        ), f"lora_name wrong: {_event_field(decoded, 7, 'lora_name')}"
        assert (
            _event_field(decoded, 8, "extra_keys") is None
        ), f"extra_keys wrong: {_event_field(decoded, 8, 'extra_keys')}"
        next_idx = 9
        if _has_group_idx(BlockStored):
            assert (
                _event_field(decoded, next_idx, "group_idx") == 0
            ), f"group_idx wrong: {_event_field(decoded, next_idx, 'group_idx')}"
            next_idx += 1
        if _has_kv_cache_spec_kind(BlockStored):
            assert _event_field(decoded, next_idx, "kv_cache_spec_kind") == (
                "full_attention"
            ), f"kv_cache_spec_kind wrong: {_event_field(decoded, next_idx, 'kv_cache_spec_kind')}"
            next_idx += 1
        if _has_kv_cache_spec_sliding_window(BlockStored):
            assert (
                _event_field(decoded, next_idx, "kv_cache_spec_sliding_window") == 128
            ), f"kv_cache_spec_sliding_window wrong: {_event_field(decoded, next_idx, 'kv_cache_spec_sliding_window')}"

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

        assert _event_tag(decoded) == "BlockStored"
        extra_keys = _event_field(decoded, 8, "extra_keys")
        assert extra_keys == [[[mm_hash, 7]]], (
            "vLLM multimodal extra_keys no longer serialize as nested tuple/list "
            f"payloads. Decoded: {extra_keys!r}"
        )
        if _has_group_idx(BlockStored):
            assert (
                _event_field(decoded, 9, "group_idx") == 0
            ), f"group_idx wrong: {_event_field(decoded, 9, 'group_idx')}"
        if _has_kv_cache_spec_kind(BlockStored):
            kind_idx = 10 if _has_group_idx(BlockStored) else 9
            assert _event_field(decoded, kind_idx, "kv_cache_spec_kind") == (
                "full_attention"
            ), f"kv_cache_spec_kind wrong: {_event_field(decoded, kind_idx, 'kv_cache_spec_kind')}"
        if _has_kv_cache_spec_sliding_window(BlockStored):
            window_idx = 9
            if _has_group_idx(BlockStored):
                window_idx += 1
            if _has_kv_cache_spec_kind(BlockStored):
                window_idx += 1
            assert (
                _event_field(decoded, window_idx, "kv_cache_spec_sliding_window") == 128
            ), f"kv_cache_spec_sliding_window wrong: {_event_field(decoded, window_idx, 'kv_cache_spec_sliding_window')}"

    def test_block_removed_serialization_format(self):
        """Verify BlockRemoved serializes to a Rust-decodable msgpack shape.

        Accepts both the tagged-array (array_like=True) and tagged-map wire
        shapes; the Rust decoder handles both.
        """
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

        assert isinstance(
            decoded, (list, dict)
        ), f"Expected list or dict, got {type(decoded)}"
        assert _event_tag(decoded) == "BlockRemoved"
        if isinstance(decoded, list):
            expected_len = (
                3
                + int(_has_group_idx(BlockRemoved))
                + int(_has_kv_cache_spec_kind(BlockRemoved))
                + int(_has_kv_cache_spec_sliding_window(BlockRemoved))
            )
            assert len(decoded) == expected_len, (
                f"Expected {expected_len} elements, got {len(decoded)}.\n"
                f"Decoded: {decoded}\n"
                f"If field count changed, update Rust deserializers."
            )
        assert _event_field(decoded, 1, "block_hashes") == [
            123,
            456,
        ], f"block_hashes wrong: {_event_field(decoded, 1, 'block_hashes')}"
        assert (
            _event_field(decoded, 2, "medium") == "GPU"
        ), f"medium wrong: {_event_field(decoded, 2, 'medium')}"
        next_idx = 3
        if _has_group_idx(BlockRemoved):
            assert (
                _event_field(decoded, next_idx, "group_idx") == 0
            ), f"group_idx wrong: {_event_field(decoded, next_idx, 'group_idx')}"
            next_idx += 1
        if _has_kv_cache_spec_kind(BlockRemoved):
            assert _event_field(decoded, next_idx, "kv_cache_spec_kind") == (
                "full_attention"
            ), f"kv_cache_spec_kind wrong: {_event_field(decoded, next_idx, 'kv_cache_spec_kind')}"
            next_idx += 1
        if _has_kv_cache_spec_sliding_window(BlockRemoved):
            assert (
                _event_field(decoded, next_idx, "kv_cache_spec_sliding_window") == 128
            ), f"kv_cache_spec_sliding_window wrong: {_event_field(decoded, next_idx, 'kv_cache_spec_sliding_window')}"
