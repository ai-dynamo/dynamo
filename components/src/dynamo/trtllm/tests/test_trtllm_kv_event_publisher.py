# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for TensorRT-LLM KV-event publication and V2 multimodal normalization."""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dynamo.common.multimodal.routing_utils import (
    build_mm_routing_info_from_features,
    pad_value_for_mm_hash,
)
from dynamo.trtllm import publisher as publisher_mod

pytestmark = [
    pytest.mark.unit,
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]


def _publisher_for_kv_event_test():
    pub = publisher_mod.Publisher.__new__(publisher_mod.Publisher)
    pub.additional_metrics = None
    pub._last_engine_event_id_by_rank = {}
    pub.processing_initial_created_events = False
    pub.partial_block_hashes = set()
    pub.kv_block_size = 4
    pub.mm_token_id_offset = 1000
    pub.max_window_size = 128
    return pub


def _stored_kv_event(cache_salt="tenant-a"):
    return {
        "event_id": 1,
        "attention_dp_rank": 0,
        "data": {
            "type": "stored",
            "parent_hash": None,
            "blocks": [
                {
                    "block_hash": 123,
                    "cache_salt": cache_salt,
                    "tokens": [
                        {"token_id": 1},
                        {"token_id": 2},
                        {"token_id": 3},
                        {"token_id": 4},
                    ],
                }
            ],
        },
    }


def _load_v2_mm_fixture():
    fixture_path = Path(__file__).parent / "fixtures" / "trtllm_v2_mm_kv_events.json"
    return json.loads(fixture_path.read_text())


@pytest.mark.multimodal
def test_v2_mm_fixture_matches_trtllm_uuid_event_contract():
    fixture = _load_v2_mm_fixture()
    source = fixture["forward_source"]
    event = fixture["forward_event"]

    assert source == {
        "repository": "NVIDIA/TensorRT-LLM",
        "pull_request": 19529,
        "head_commit": "687d95a29106e642d3f38b81ae193ac5d0aa6ee2",
        "producer": "KVCacheEventManager + KVCacheEventSerializer",
        "contract": "buffered serialized event with additive UUID identity",
    }
    assert event["event_id"] == 0
    assert event["window_size"] == 128
    assert event["layer_group_id"] == 0
    assert event["attention_dp_rank"] == 0
    assert event["hash_algo"] == "v1_block_key"


@pytest.mark.multimodal
def test_text_only_event_remains_unchanged():
    pub = _publisher_for_kv_event_test()
    event = _stored_kv_event()

    _, normalized = pub._normalize_kv_event(event)

    assert normalized["token_ids"] == [1, 2, 3, 4]
    assert normalized["num_block_tokens"] == [4]
    assert normalized["block_hashes"] == [123]
    assert normalized["block_mm_infos"] == [None]


@pytest.mark.multimodal
def test_v2_mm_fixture_matches_frontend_routing_identity():
    fixture = _load_v2_mm_fixture()
    pub = _publisher_for_kv_event_test()
    pub.mm_token_id_offset = fixture["mm_token_id_offset"]

    _, normalized = pub._normalize_kv_event(copy.deepcopy(fixture["forward_event"]))

    key_a = fixture["forward_event"]["data"]["blocks"][0]["mm_keys"][0]
    key_b = fixture["forward_event"]["data"]["blocks"][2]["mm_keys"][1]
    digest_a = key_a["hash"]
    digest_b = key_b["hash"]
    routing_uuid_a = key_a["uuid"]
    routing_uuid_b = key_b["uuid"]
    assert pad_value_for_mm_hash(int(routing_uuid_a, 16)) != pad_value_for_mm_hash(
        int(digest_a[:16], 16)
    )
    assert pad_value_for_mm_hash(int(routing_uuid_b, 16)) != pad_value_for_mm_hash(
        int(digest_b[:16], 16)
    )
    request_tokens = [1, 99, 99, 99, 99, 7, 99, 99, 99, 88, 88, 9]
    request_features = [
        SimpleNamespace(
            mm_hash=routing_uuid_a,
            mm_position=SimpleNamespace(
                offset=1,
                length=8,
                is_embed=[True, True, True, True, False, True, True, True],
            ),
        ),
        SimpleNamespace(
            mm_hash=routing_uuid_b,
            mm_position=SimpleNamespace(offset=9, length=2, is_embed=None),
        ),
    ]
    request_routing = build_mm_routing_info_from_features(
        request_features, request_tokens
    )

    assert request_routing is not None
    assert normalized["token_ids"] == request_routing["routing_token_ids"]
    assert normalized["num_block_tokens"] == [4, 4, 4]
    assert normalized["block_hashes"] == [101, 102, 103]
    assert normalized["parent_hash"] == 77
    assert normalized["block_mm_infos"] == [None, None, None]


@pytest.mark.multimodal
def test_v2_mm_fixture_preserves_cross_block_and_text_separated_offsets():
    fixture = _load_v2_mm_fixture()
    blocks = fixture["forward_event"]["data"]["blocks"]

    assert [[key["start_offset"] for key in block["mm_keys"]] for block in blocks] == [
        [0],
        [3, 4],
        [6, 0],
    ]

    pub = _publisher_for_kv_event_test()
    _, normalized = pub._normalize_kv_event(copy.deepcopy(fixture["forward_event"]))
    routing_uuid_a = blocks[0]["mm_keys"][0]["uuid"]
    pad_a = pad_value_for_mm_hash(int(routing_uuid_a, 16))
    assert normalized["token_ids"][1:5] == [pad_a] * 4
    assert normalized["token_ids"][5] == 7
    assert normalized["token_ids"][6:9] == [pad_a] * 3


@pytest.mark.multimodal
def test_v2_mm_fixture_normalizes_distinct_image_and_video_items():
    fixture = _load_v2_mm_fixture()
    blocks = fixture["forward_event"]["data"]["blocks"]
    routing_uuid_a = blocks[0]["mm_keys"][0]["uuid"]
    routing_uuid_b = blocks[2]["mm_keys"][1]["uuid"]
    pub = _publisher_for_kv_event_test()

    _, normalized = pub._normalize_kv_event(copy.deepcopy(fixture["forward_event"]))

    pad_a = pad_value_for_mm_hash(int(routing_uuid_a, 16))
    pad_b = pad_value_for_mm_hash(int(routing_uuid_b, 16))
    assert pad_a != pad_b
    assert normalized["token_ids"][-4:] == [pad_a, pad_b, pad_b, 9]


@pytest.mark.parametrize(
    "bad_digest",
    ["abcd", "z" * 64, " " * 64],
    ids=["wrong-length", "non-hex", "whitespace"],
)
@pytest.mark.multimodal
def test_v2_mm_malformed_digest_is_dropped_without_logging_digest(bad_digest, caplog):
    fixture = _load_v2_mm_fixture()
    event = copy.deepcopy(fixture["forward_event"])
    first_block = event["data"]["blocks"][0]
    first_block["tokens"][1]["token_id"] = bad_digest
    first_block["mm_keys"][0]["hash"] = bad_digest
    pub = _publisher_for_kv_event_test()

    with caplog.at_level(logging.WARNING):
        normalized = pub._normalize_kv_event(event)

    assert normalized is None
    assert "Dropping unsupported multimodal stored KV event" in caplog.text
    assert bad_digest not in caplog.text


@pytest.mark.parametrize(
    "malformation",
    ["missing", "missing-uuid", "inconsistent", "out-of-order"],
)
@pytest.mark.multimodal
def test_v2_mm_incomplete_or_inconsistent_keys_fail_closed(malformation, caplog):
    fixture = _load_v2_mm_fixture()
    event = copy.deepcopy(fixture["forward_event"])
    blocks = event["data"]["blocks"]
    if malformation == "missing":
        blocks[0]["mm_keys"] = []
    elif malformation == "missing-uuid":
        del blocks[0]["mm_keys"][0]["uuid"]
    elif malformation == "inconsistent":
        blocks[0]["mm_keys"][0]["hash"] = blocks[2]["mm_keys"][1]["hash"]
    else:
        blocks[1]["mm_keys"][0]["start_offset"] = 2
    pub = _publisher_for_kv_event_test()

    with caplog.at_level(logging.WARNING):
        normalized = pub._normalize_kv_event(event)

    assert normalized is None
    assert "Dropping unsupported multimodal stored KV event" in caplog.text


@pytest.mark.parametrize(
    "bad_token_id",
    [1001.5, None, True],
    ids=["float", "none", "bool"],
)
@pytest.mark.multimodal
def test_invalid_token_id_is_skipped_and_next_text_event_is_published(
    bad_token_id, caplog
):
    fixture = _load_v2_mm_fixture()
    invalid_event = copy.deepcopy(fixture["forward_event"])
    invalid_event["data"]["blocks"][0]["tokens"][0]["token_id"] = bad_token_id
    next_event = _stored_kv_event()
    next_event["event_id"] = invalid_event["event_id"] + 1
    pub = _publisher_for_kv_event_test()
    publisher = MagicMock()
    pub.zmq_kv_event_publisher = None
    pub.kv_event_publishers = {0: publisher}

    with caplog.at_level(logging.WARNING):
        pub._handle_kv_event_batch([invalid_event, next_event])

    publisher.publish_batch.assert_called_once()
    assert publisher.publish_batch.call_args.args[0][0]["token_ids"] == [1, 2, 3, 4]
    assert "reason=invalid token ID type" in caplog.text


@pytest.mark.multimodal
def test_legacy_digest_event_is_skipped_and_next_text_event_is_published(caplog):
    fixture = _load_v2_mm_fixture()
    pub = _publisher_for_kv_event_test()
    publisher = MagicMock()
    pub.zmq_kv_event_publisher = None
    pub.kv_event_publishers = {0: publisher}
    next_event = _stored_kv_event()
    next_event["event_id"] = fixture["legacy_event"]["event_id"] + 1

    with caplog.at_level(logging.WARNING):
        pub._handle_kv_event_batch([copy.deepcopy(fixture["legacy_event"]), next_event])

    publisher.publish_batch.assert_called_once()
    assert publisher.publish_batch.call_args.args[0] == [
        {
            "type": "stored",
            "token_ids": [1, 2, 3, 4],
            "num_block_tokens": [4],
            "block_hashes": [123],
            "parent_hash": None,
            "block_mm_infos": [None],
            "lora_name": None,
            "cache_salt": "tenant-a",
        }
    ]
    assert "Dropping unsupported multimodal stored KV event" in caplog.text


@pytest.mark.multimodal
def test_direct_and_consolidator_paths_receive_identical_v2_mm_content():
    fixture = _load_v2_mm_fixture()
    event = fixture["forward_event"]

    direct = _publisher_for_kv_event_test()
    direct_publisher = MagicMock()
    direct.zmq_kv_event_publisher = None
    direct.kv_event_publishers = {0: direct_publisher}
    direct._handle_kv_event_batch([copy.deepcopy(event)])
    direct_event = direct_publisher.publish_batch.call_args.args[0][0]

    consolidator = _publisher_for_kv_event_test()
    consolidator.zmq_kv_event_publisher = MagicMock()
    consolidator.kv_event_publishers = None
    consolidator._handle_zmq_kv_event(copy.deepcopy(event))
    zmq_args = consolidator.zmq_kv_event_publisher.publish_stored.call_args.args
    consolidator_event = {
        "type": "stored",
        "token_ids": zmq_args[0],
        "num_block_tokens": zmq_args[1],
        "block_hashes": zmq_args[2],
        "parent_hash": zmq_args[3],
        "block_mm_infos": zmq_args[4],
        "lora_name": zmq_args[6],
        "cache_salt": zmq_args[7],
    }

    assert zmq_args[5] == 0
    assert consolidator_event == direct_event
