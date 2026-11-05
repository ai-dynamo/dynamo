# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Retained GPU event namespaces use real request metadata and upstream events."""
import hashlib
from types import SimpleNamespace

import msgspec
import pytest
import torch
from kvbm.vllm_integration.gpu_offloading import (
    GPUOffloadingConnector,
    _GPUOffloadingEventsTracker,
)
from vllm.distributed.kv_events import BlockRemoved, BlockStored
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core import kv_cache_utils
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
)
from vllm.v1.kv_offload.base import Medium, OffloadingEvent, make_offload_key

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
]


def request(salt="dynamo-cache-salt:tenant-a", **changes):
    value = SimpleNamespace(
        cache_salt=salt,
        lora_request=None,
        mm_features=[],
        prompt_embeds=None,
        _prompt_embeds_per_block_hashes={},
        all_token_ids=list(range(256)),
        block_hashes=[BlockHash(bytes([n]) * 32) for n in range(1, 5)],
    )
    value.__dict__.update(changes)
    return value


def tracker():
    return _GPUOffloadingEventsTracker(
        SimpleNamespace(enable_kv_cache_events=True, self_describing_kv_events=True)
    )


def group():
    return SimpleNamespace(
        tokens_per_chunk=128,
        hashes_per_chunk=2,
        tokens_per_block=64,
        sliding_window_size_in_chunks=None,
        group_idx=0,
        kv_event_group_spec=OffloadingEventGroupSpec("full_attention", None),
    )


def record(value, req, chunk=0):
    key = make_offload_key(req.block_hashes[2 * chunk + 1], 0)
    value.record_store(req, group(), chunk, key)
    return key


def events(value, keys, removed=False):
    connector = GPUOffloadingConnector.__new__(GPUOffloadingConnector)
    connector.connector_scheduler = SimpleNamespace(
        take_events=lambda: value.take_events(
            [OffloadingEvent(keys=keys, medium=Medium.CPU, removed=removed)]
        )
    )
    return list(connector.take_events())


def test_root_salt_is_snapshotted_and_serialized_per_block():
    value, req = tracker(), request()
    key = record(value, req)
    req.cache_salt = "dynamo-cache-salt:changed-after-snapshot"
    (stored,) = events(value, [key])
    assert isinstance(stored, BlockStored) and stored.medium == "gpu_direct"
    assert stored.parent_block_hash is None and stored.block_size == 64
    assert stored.token_ids == list(range(128))
    assert stored.extra_keys == [("dynamo-cache-salt:tenant-a",), None]
    wire = msgspec.msgpack.decode(msgspec.msgpack.encode(stored))
    assert wire["extra_keys"] == [["dynamo-cache-salt:tenant-a"], None]


def test_two_salts_keep_identical_token_requests_isolated(monkeypatch):
    monkeypatch.setattr(
        kv_cache_utils, "NONE_HASH", BlockHash(b"root" * 8), raising=False
    )
    value = tracker()
    requests = [request("dynamo-cache-salt:" + salt) for salt in ["a", "b"]]
    keys = []
    for req in requests:
        parent = None
        hashes = []
        for start in range(0, 256, 64):
            extra, _ = generate_block_hash_extra_keys(req, start, start + 64, 0)
            parent = hash_block_tokens(
                lambda x: hashlib.sha256(msgspec.msgpack.encode(x)).digest(),
                parent,
                req.all_token_ids[start : start + 64],
                extra,
            )
            hashes.append(parent)
        req.block_hashes = hashes
        keys.append(record(value, req))
    assert requests[0].block_hashes[0] != requests[1].block_hashes[0]
    assert len(value._pending_event_metadata) == 2
    first, second = events(value, keys)
    assert first.block_hashes[0] != second.block_hashes[0]
    assert first.extra_keys[0] == ("dynamo-cache-salt:a",)
    assert second.extra_keys[0] == ("dynamo-cache-salt:b",)


def test_removal_revokes_snapshotted_hashes_and_unknown_metadata_stays_unknown():
    value, req = tracker(), request()
    key = record(value, req)
    (stored,) = events(value, [key])
    (removed,) = events(value, [key], removed=True)
    assert isinstance(removed, BlockRemoved) and removed.medium == "gpu_direct"
    assert removed.block_hashes == stored.block_hashes
    assert value._pending_event_metadata == {}
    unknown = make_offload_key(BlockHash(b"unknown" * 4), 0)
    (placeholder,) = events(value, [unknown])
    assert placeholder.parent_block_hash is None and placeholder.block_size == 0
    assert placeholder.token_ids == [] and placeholder.extra_keys is None
    assert value._pending_event_metadata == {}


@pytest.mark.parametrize("context", ["plain", "lora", "multimodal", "embeddings"])
def test_extra_context_uses_the_original_block_spans(context):
    value, req = tracker(), request()
    if context == "lora":
        req.lora_request = SimpleNamespace(
            adapter_id=7, name="adapter", lora_name="adapter", lora_path="/a"
        )
    elif context == "multimodal":
        req.mm_features = [
            SimpleNamespace(
                identifier="image", mm_position=SimpleNamespace(offset=96, length=64)
            )
        ]
    elif context == "embeddings":
        req.prompt_embeds = torch.arange(1024, dtype=torch.float32).reshape(256, 4)
    key0, key1 = record(value, req, 0), record(value, req, 1)
    first, second = events(value, [key0, key1])
    if context == "plain":
        assert first.extra_keys == [("dynamo-cache-salt:tenant-a",), None]
        assert second.extra_keys == [None, None]
    elif context == "lora":
        assert first.extra_keys == [
            ("adapter", "dynamo-cache-salt:tenant-a"),
            ("adapter",),
        ]
        assert second.extra_keys == [("adapter",), ("adapter",)]
    elif context == "multimodal":
        assert first.extra_keys == [("dynamo-cache-salt:tenant-a",), (("image", 32),)]
        assert second.extra_keys == [(("image", -32),), None]
    else:
        assert len(req._prompt_embeds_per_block_hashes) == 4
        assert first.extra_keys[0][0] == "dynamo-cache-salt:tenant-a"
        assert all(
            isinstance(keys[-1], bytes) and len(keys[-1]) == 32
            for event in [first, second]
            for keys in event.extra_keys
        )
        assert first.extra_keys[0][-1] != second.extra_keys[0][-1]


@pytest.mark.parametrize("scheduler", [False, True])
def test_connector_installs_tracker_only_for_scheduler(monkeypatch, scheduler):
    original = OffloadingEventsTracker(
        SimpleNamespace(enable_kv_cache_events=True, self_describing_kv_events=True)
    )
    state = SimpleNamespace(_events_tracker=original) if scheduler else None
    monkeypatch.setattr(
        OffloadingConnector,
        "__init__",
        lambda self, *args: setattr(self, "connector_scheduler", state),
    )
    connector = GPUOffloadingConnector(None, None, None)
    if scheduler:
        assert isinstance(state._events_tracker, _GPUOffloadingEventsTracker)
        assert state._events_tracker.config is original.config
    else:
        assert connector.connector_scheduler is None


def test_connector_rejects_installation_after_metadata_was_recorded(monkeypatch):
    original = tracker()
    record(original, request())
    state = SimpleNamespace(_events_tracker=original)
    monkeypatch.setattr(
        OffloadingConnector,
        "__init__",
        lambda self, *args: setattr(self, "connector_scheduler", state),
    )
    with pytest.raises(RuntimeError, match="precede request admission"):
        GPUOffloadingConnector(None, None, None)
    assert state._events_tracker is original and original._pending_event_metadata


@pytest.mark.parametrize("boundary", [64, 192])
def test_partial_chunks_preserve_only_complete_hash_metadata(boundary):
    value, req = tracker(), request()
    key = make_offload_key(req.block_hashes[boundary // 64 - 1], 0)
    value.record_partial_store(req, group(), boundary, key)
    (stored,) = events(value, [key])
    assert stored.block_size == 64 and len(stored.block_hashes) == 1
    assert stored.token_ids == list(range(boundary - 64, boundary))
    assert stored.extra_keys == (
        [("dynamo-cache-salt:tenant-a",)] if boundary == 64 else [None]
    )
    (removed,) = events(value, [key], removed=True)
    assert removed.block_hashes == stored.block_hashes
    assert not value._pending_event_metadata


def test_unaligned_partial_hash_is_not_published_as_a_complete_block():
    value, req = tracker(), request()
    key = make_offload_key(req.block_hashes[0], 0)
    with pytest.raises(AssertionError):
        value.record_partial_store(req, group(), 65, key)
    assert not value._pending_event_metadata


def test_physical_block_metadata_covers_all_constituent_hash_units():
    value = tracker()
    req = request(
        mm_features=[
            SimpleNamespace(
                identifier="image", mm_position=SimpleNamespace(offset=96, length=16)
            )
        ]
    )
    config = group()
    config.tokens_per_block = 128
    key = make_offload_key(req.block_hashes[1], 0)
    value.record_store(req, config, 0, key)
    (stored,) = events(value, [key])
    assert stored.block_size == 128 and len(stored.block_hashes) == 1
    assert stored.token_ids == list(range(128))
    assert stored.extra_keys == [(("image", 96), "dynamo-cache-salt:tenant-a")]
