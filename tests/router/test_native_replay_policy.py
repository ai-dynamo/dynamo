# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the installed native policy API without importing any simulator."""

import json

import pytest

from dynamo.llm import NativeReplayPolicy

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def policy(role="aggregated", affinity="sibling_group", **workers):
    config = {"policy": "kv_router"}
    if affinity:
        config["affinity"] = {"mode": affinity, "ttl_seconds": 10}
    return NativeReplayPolicy(
        role,
        json.dumps(config),
        json.dumps(
            {
                "api_version": 1,
                "block_size": 4,
                "total_kv_blocks": 128,
                "max_num_batched_tokens": 128,
                "dp_size": 2,
                "workers": [{"worker_id": 3}, {"worker_id": 9}],
                "capture_decisions": True,
                **workers,
            }
        ),
    )


def request(request_id, session="child", parent="parent", scope="play-a", **fields):
    return json.dumps(
        {
            "request_id": request_id,
            "input_tokens": 8,
            "output_tokens": 1,
            "local_block_hashes": [101, 102],
            "sequence_hashes": [201, 202],
            "prompt_token_source": "materialized",
            "identity": {
                "scope": scope,
                "session": session,
                "root": "root",
                "parent": parent,
                "lineage_available": True,
            },
            **fields,
        }
    )


def cache_event(event_id=1, removed=False):
    data = (
        {"removed": {"block_hashes": [201, 202]}}
        if removed
        else {
            "stored": {
                "parent_hash": None,
                "start_position": 0,
                "blocks": [
                    {"block_hash": 201, "tokens_hash": 101},
                    {"block_hash": 202, "tokens_hash": 102},
                ],
            }
        }
    )
    return json.dumps(
        [{"worker_id": 9, "event": {"event_id": event_id, "dp_rank": 1, "data": data}}]
    )


@pytest.mark.parametrize("role", ["aggregated", "prefill", "decode"])
def test_physical_cache_and_sibling_dispatch_binding(role):
    """Native scoring consumes physical events and siblings keep the actual worker/DP."""
    host = policy(role)
    assert NativeReplayPolicy.contract()["api_version"] == 1
    assert json.loads(host.observe(cache_event(), 0)) == []
    first = json.loads(host.place(request("first"), 0))["decision"]
    assert (first["worker_id"], first["dp_rank"]) == (9, 1)
    assert first["overlap_blocks"] == first["best_available_overlap_blocks"] == 2
    assert first["cached_tokens"] == 8
    assert (
        json.loads(host.place(request("second", session="sibling"), 0))["decision"]
        is None
    )
    assert host.pending_count() == 1
    assert host.next_wakeup_ms() is None
    host.dispatch_committed("first", 0)
    assert host.next_wakeup_ms() == 0
    (second,) = json.loads(host.advance_clock(0))
    assert (second["worker_id"], second["dp_rank"]) == (9, 1)
    host.dispatch_committed("second", 0)
    assert host.pending_count() == 0
    evidence = json.loads(host.evidence())
    assert evidence["physical_kv_events"] == 1
    assert evidence["decision_count"] == 2
    assert (
        evidence["decisions"][0]["group_key"] == evidence["decisions"][1]["group_key"]
    )
    assert [item["binding_reused"] for item in evidence["decisions"]] == [False, True]
    host.prefill_completed("first", 1)
    host.request_terminal("first", 2)
    host.request_terminal("second", 2)
    host.observe(cache_event(event_id=2, removed=True), 3)
    after_remove = json.loads(host.place(request("third"), 3))["decision"]
    assert after_remove["overlap_blocks"] == after_remove["cached_tokens"] == 0
    assert (after_remove["worker_id"], after_remove["dp_rank"]) == (9, 1)
    host.dispatch_aborted("third", 3)


def test_abort_releases_initializer_and_cancel_removes_waiter():
    """Failed dispatch never establishes an affinity binding or strands its siblings."""
    host = policy()
    host.place(request("aborted"), 0)
    host.place(request("cancelled"), 0)
    host.place(request("retry"), 0)
    assert host.cancel_pending("cancelled")
    assert not host.cancel_pending("missing")
    host.dispatch_aborted("aborted", 0)
    (released,) = json.loads(host.advance_clock(0))
    assert released["request_id"] == "retry"
    assert host.pending_count() == 0
    assert [d["binding_reused"] for d in json.loads(host.evidence())["decisions"]] == [
        False,
        False,
    ]
    host.dispatch_aborted("retry", 0)


@pytest.mark.parametrize("mode", ["session", "sibling_group"])
def test_virtual_ttl_starts_at_last_release_and_scope_isolated(mode):
    """Active native leases outlive TTL; exact idle expiry permits a fresh binding."""
    host = policy(affinity=mode)
    host.place(request("first"), 0)
    host.dispatch_committed("first", 0)
    host.place(request("active"), 20_000)
    host.dispatch_committed("active", 20_000)
    host.request_terminal("first", 20_000)
    host.place(request("other-play", scope="play-b"), 20_000)
    host.dispatch_committed("other-play", 20_000)
    host.place(request("parent", session="parent", parent=None), 20_000)
    host.dispatch_committed("parent", 20_000)
    host.request_terminal("active", 21_000)
    host.place(request("expired"), 31_000)
    evidence = json.loads(host.evidence())["decisions"]
    assert [d["binding_reused"] for d in evidence] == [False, True, False, False, False]
    assert evidence[0]["group_key"] != evidence[2]["group_key"]
    assert evidence[0]["group_key"] != evidence[3]["group_key"]
    with pytest.raises(ValueError, match="monotonic"):
        host.advance_clock(30_999)
    with pytest.raises(ValueError, match="duplicate"):
        host.place(request("expired"), 31_000)
    host.dispatch_aborted("expired", 31_000)
    host.request_terminal("other-play", 31_000)
    host.request_terminal("parent", 31_000)


@pytest.mark.parametrize(
    "fields,match",
    [
        ({"prompt_token_source": "length_only_synthetic"}, "materialized"),
        ({"preferred_dp_rank": 1}, "DP pins"),
        ({"policy_class": "custom"}, "policy classes"),
        ({"identity": {"session": "s", "lineage_available": False}}, "lineage"),
    ],
)
def test_unsupported_requests_fail_explicitly(fields, match):
    """Unsupported request semantics cannot silently select a different policy."""
    host = policy()
    with pytest.raises(ValueError, match=match):
        host.place(request("bad", **fields), 0)
    assert json.loads(host.evidence())["decision_count"] == 0


def test_policy_without_affinity_and_capture_is_bounded():
    """KV selection works alone and disabling evidence avoids per-request accumulation."""
    host = policy(affinity=None, capture_decisions=False)
    host.observe(cache_event(), 0)
    result = json.loads(host.place(request("plain", identity={}), 0))["decision"]
    assert (result["worker_id"], result["dp_rank"], result["cached_tokens"]) == (
        9,
        1,
        8,
    )
    host.dispatch_committed("plain", 0)
    host.request_terminal("plain", 1)
    evidence = json.loads(host.evidence())
    assert evidence["decision_count"] == 1
    assert evidence["decisions"] == []
    assert not evidence["decisions_captured"]
    with pytest.raises(ValueError, match="device KV"):
        policy(host_offload=True)


def test_wire_version_tier_and_long_identity_are_not_silently_lost():
    """Wire validation preserves tier semantics and bounded namespaced affinity IDs."""
    with pytest.raises(ValueError, match="API version"):
        policy(api_version=2)
    host = policy()
    event = json.loads(cache_event())
    event[0]["event"]["tier"] = "host_pinned"
    with pytest.raises(ValueError, match="device KV"):
        host.observe(json.dumps(event), 0)
    assert json.loads(host.evidence())["physical_kv_events"] == 0
    # Each upstream ID may be valid even though its namespaced tuple exceeds 256 bytes.
    host.place(request("long", session="s" * 256, parent="p" * 256, scope="x" * 256), 0)
    key = json.loads(host.evidence())["decisions"][0]["group_key"]
    assert len(key.encode()) <= 256
    host.dispatch_aborted("long", 0)
