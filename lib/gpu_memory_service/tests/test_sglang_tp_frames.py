# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The compact transport must preserve all agreement and failure semantics."""

import struct

import pytest
import torch.distributed as dist
from gpu_memory_service.integrations.sglang.tp_consistency import (
    GmsTPConsistencyError,
    TPConsistency,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_candidate_read_and_intersection_use_one_vote(monkeypatch):
    cohort = TPConsistency(world_size=2)
    votes = []

    def gather(value):
        votes.append(value)
        return [value, ("eligible", True, [7, 4])]

    monkeypatch.setattr(cohort, "_gather", gather)
    local, common = cohort.run_intersection("eligible", lambda: [5, 4, 7])
    assert local == [5, 4, 7]
    assert common == [4, 7]
    assert votes == [("eligible", True, [5, 4, 7])]


@pytest.mark.parametrize("failure", ["local", "peer", "stage"])
def test_candidate_read_failure_never_returns_eligibility(monkeypatch, failure):
    cohort = TPConsistency(world_size=2)
    votes = []

    def gather(value):
        votes.append(value)
        return [
            value,
            (
                "wrong" if failure == "stage" else "eligible",
                failure != "peer",
                [4],
            ),
        ]

    def candidates():
        if failure == "local":
            raise RuntimeError("native read failed")
        return [4]

    monkeypatch.setattr(cohort, "_gather", gather)
    with pytest.raises(GmsTPConsistencyError, match="eligible"):
        cohort.run_intersection("eligible", candidates)
    assert len(votes) == 1
    assert votes[0][1] == (failure != "local")


def test_small_vote_uses_one_collective(monkeypatch):
    calls = []

    def gather(outputs, vote, *, group):
        calls.append(vote.numel())
        for output in outputs:
            output.copy_(vote)

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(
        dist, "all_gather_object", lambda *a, **kw: pytest.fail("size exchange")
    )
    value = ("reserve:vote", True, [1, 7, 42])
    assert TPConsistency(world_size=2)._gather(value) == [value, value]
    assert calls == [4096]


@pytest.mark.parametrize("oversized_peer", [False, True])
def test_large_vote_collectively_falls_back(monkeypatch, oversized_peer):
    calls = []
    value = ("vote", "x" * (10000 if not oversized_peer else 1))

    def gather(outputs, vote, *, group):
        for output in outputs:
            output.copy_(vote)
        if oversized_peer:
            outputs[1][:4].copy_(
                __import__("torch").tensor(list(struct.pack("<I", 10000)))
            )

    def objects(outputs, payload, *, group):
        calls.append(payload)
        outputs[:] = [payload, ("vote", "peer")]

    monkeypatch.setattr(dist, "all_gather", gather)
    monkeypatch.setattr(dist, "all_gather_object", objects)
    assert TPConsistency(world_size=2)._gather(value) == [value, ("vote", "peer")]
    assert calls == [value]


def test_encoding_failure_still_votes(monkeypatch):
    calls = []

    class Unencodable:
        def __reduce__(self):
            raise ValueError("encode failed")

    def gather(outputs, vote, *, group):
        calls.append(bytes(vote[:4].tolist()))
        for output in outputs:
            output.copy_(vote)

    monkeypatch.setattr(dist, "all_gather", gather)
    with pytest.raises(GmsTPConsistencyError, match="channel failed"):
        TPConsistency(world_size=2)._gather(Unencodable())
    assert calls == [b"\0\0\0\0"]


def test_small_vote_transport_failure_is_fatal(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("peer died")

    monkeypatch.setattr(dist, "all_gather", fail)
    with pytest.raises(GmsTPConsistencyError, match="channel failed"):
        TPConsistency(world_size=2)._gather(("vote", True))
