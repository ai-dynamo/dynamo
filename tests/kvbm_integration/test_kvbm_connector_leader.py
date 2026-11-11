#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for how KvConnectorLeader.build_connector_meta hands vLLM's
scheduled requests to the Rust connector. The Rust connector and its
SchedulerOutput are replaced by recorders, so no GPU or engine is needed.
"""

import dataclasses
import types

import pytest

from .common import check_module_available

HAS_VLLM = check_module_available("vllm")
HAS_KVBM = check_module_available("kvbm")

if HAS_VLLM and HAS_KVBM:
    from kvbm.vllm_integration import connector_leader
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput

pytestmark = [
    pytest.mark.kvbm,
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
    pytest.mark.skipif(not (HAS_VLLM and HAS_KVBM), reason="requires vllm and kvbm"),
]


class _RecordingSchedulerOutput:
    """Stands in for the Rust SchedulerOutput and records what the leader adds."""

    def __init__(self):
        self.new_requests = []
        self.cached_requests = []
        self.num_scheduled_tokens = {}

    def add_new_request(
        self, request_id, prompt_token_ids, block_ids, num_computed_tokens
    ):
        self.new_requests.append((request_id, list(block_ids), num_computed_tokens))

    def add_cached_request(
        self,
        request_id,
        resumed_from_preemption,
        new_token_ids,
        new_block_ids,
        num_computed_tokens,
    ):
        self.cached_requests.append(
            (
                request_id,
                resumed_from_preemption,
                list(new_token_ids),
                list(new_block_ids),
                num_computed_tokens,
            )
        )

    def add_num_scheduled_tokens(self, num_scheduled_tokens):
        self.num_scheduled_tokens = dict(num_scheduled_tokens)

    def get_num_scheduled_tokens(self):
        return sum(self.num_scheduled_tokens.values())


class _StubRustLeader:
    """Stands in for the Rust connector and tracks which requests have a slot."""

    def __init__(self):
        self.slots = set()

    def has_slot(self, request_id):
        return request_id in self.slots

    def create_slot(self, request, tokens):
        self.slots.add(request.request_id)

    def get_num_new_matched_tokens(self, request_id, num_tokens, num_computed_tokens):
        return 0, False

    def reset_cache(self):
        self.slots.clear()
        return True

    def build_connector_metadata(self, output):
        return output


@pytest.fixture
def leader(monkeypatch):
    monkeypatch.setattr(
        connector_leader, "RustSchedulerOutput", _RecordingSchedulerOutput
    )
    monkeypatch.setattr(connector_leader, "KvbmRequest", types.SimpleNamespace)
    # Skip __init__, which needs a VllmConfig and a KVBM leader.
    leader = object.__new__(connector_leader.KvConnectorLeader)
    leader._connector = _StubRustLeader()
    leader._preempted_req_ids = set()
    leader._new_slot_req_ids = set()
    return leader


def _match(leader, req_id):
    """vLLM asks the connector for matched tokens before it schedules a waiting request."""
    request = types.SimpleNamespace(
        request_id=req_id,
        all_token_ids=[0] * 16,
        num_tokens=16,
        lora_request=None,
        cache_salt=None,
    )
    leader.get_num_new_matched_tokens(request, 0)


def _new_request(req_id, block_ids, num_computed_tokens):
    required = {
        f.name: None
        for f in dataclasses.fields(NewRequestData)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    }
    required.update(
        req_id=req_id,
        prompt_token_ids=[0] * 16 * len(block_ids),
        mm_features=[],
        block_ids=(block_ids,),
        num_computed_tokens=num_computed_tokens,
    )
    return NewRequestData(**required)


def _step(new_requests=(), preempted=(), finished=(), num_scheduled_tokens=None):
    num_scheduled_tokens = num_scheduled_tokens or {}
    return dataclasses.replace(
        SchedulerOutput.make_empty(),
        scheduled_new_reqs=list(new_requests),
        preempted_req_ids=set(preempted),
        finished_req_ids=set(finished),
        num_scheduled_tokens=num_scheduled_tokens,
        total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
    )


def _admit(req_id):
    return _step(
        new_requests=[_new_request(req_id, [5], 0)], num_scheduled_tokens={req_id: 16}
    )


def test_request_resumed_as_new_takes_the_resumed_path(leader):
    """Model Runner V2 schedules a preempted request again in scheduled_new_reqs."""
    _match(leader, "r0")
    leader.build_connector_meta(_admit("r0"))
    leader.build_connector_meta(_step(preempted=["r0"]))

    _match(leader, "r0")
    _match(leader, "r1")
    out = leader.build_connector_meta(
        _step(
            new_requests=[
                _new_request("r0", [5, 6, 7], 16),
                _new_request("r1", [8], 0),
            ],
            num_scheduled_tokens={"r0": 32, "r1": 16},
        )
    )

    assert out.cached_requests == [("r0", True, [], [5, 6, 7], 16)]
    assert out.new_requests == [("r1", [8], 0)]


@pytest.mark.parametrize("reset_while_waiting", [False, True])
def test_request_preempted_by_a_cache_reset_takes_the_new_path(
    leader, reset_while_waiting
):
    """A request whose slot reset_cache dropped gets a new slot and takes the new path.

    reset_prefix_cache(reset_running_requests=True, reset_connector=True) preempts
    the running requests and then calls reset_cache; the next step reports them
    preempted and schedules them again. With reset_while_waiting, the request was
    preempted earlier and the reset comes while it waits.
    """
    _match(leader, "r0")
    leader.build_connector_meta(_admit("r0"))
    if reset_while_waiting:
        leader.build_connector_meta(_step(preempted=["r0"]))
    assert leader.reset_cache()

    _match(leader, "r0")
    out = leader.build_connector_meta(
        _step(
            new_requests=[_new_request("r0", [5, 6, 7], 0)],
            preempted=[] if reset_while_waiting else ["r0"],
            num_scheduled_tokens={"r0": 48},
        )
    )

    assert out.cached_requests == []
    assert out.new_requests == [("r0", [5, 6, 7], 0)]


def test_request_aborted_after_a_cache_reset_is_forgotten(leader):
    """A request that reset_prefix_cache preempted and that is aborted before the
    next step is reported both preempted and finished; the leader keeps no id."""
    _match(leader, "r0")
    leader.build_connector_meta(_admit("r0"))
    assert leader.reset_cache()

    leader.build_connector_meta(_step(preempted=["r0"], finished=["r0"]))

    assert leader._preempted_req_ids == set()
