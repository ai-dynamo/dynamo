# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise native primitive ownership across the Python boundary."""

import json
import subprocess
import sys

import pytest

from dynamo._core import (
    OfflineRouter,
    RadixTree,
    SessionAffinity,
    compute_block_hash_for_seq,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.fixture(scope="module", autouse=True)
def nats_and_etcd():
    """These primitives do not use the bindings suite's service fixture."""
    yield


def test_affinity_wait_commit_abort_and_virtual_idle_ttl():
    table = SessionAffinity(1)
    first = table.try_acquire("session")
    assert first.target is None
    assert table.try_acquire("session") is None
    first.release()
    first = table.try_acquire("session")
    lease = table.commit(first, 7, 2)
    table.advance_clock(2000)
    bound = table.try_acquire("session")
    assert bound.target == (7, 2)  # An active native lease prevents idle expiry.
    bound.release()
    lease.release()
    table.advance_clock(3001)
    assert table.try_acquire("session").target is None
    with pytest.raises(ValueError):
        table.advance_clock(3000)
    with pytest.raises(ValueError):
        table.commit(first, 7, 2)  # Ownership cannot be consumed twice.


def test_native_indexer_selection_and_booking_lifecycle():
    tokens = [1, 2, 3, 4]
    hashes = compute_block_hash_for_seq(tokens, 4)
    tree = RadixTree()
    tree.apply_event(
        1,
        json.dumps(
            {
                "event_id": 1,
                "dp_rank": 0,
                "data": {
                    "stored": {
                        "parent_hash": None,
                        "blocks": [{"block_hash": 100, "tokens_hash": hashes[0]}],
                    }
                },
            }
        ).encode(),
    )
    router = OfflineRouter(4, 2, 1, 128, selector_seed=7)
    router.submit("request", 4, 2, tree.find_matches(hashes), hashes, tokens=tokens)
    assert router.poll() == [("request", 1, 0, 1, 1)]
    with pytest.raises(ValueError, match="duplicate"):
        router.submit("request", 4, 2, tree.find_matches(hashes), hashes, tokens=tokens)
    router.advance_clock(10)
    router.prefill_complete("request")
    router.free("request")
    router.remove_worker(1)
    router.unregister_worker(1)
    tree.remove_worker(1)
    router.add_worker(3)
    router.submit(
        "next",
        4,
        2,
        tree.find_matches(hashes),
        hashes,
        tokens=tokens,
        affinity_target=(3, 0),
    )
    assert router.poll()[0][1:3] == (3, 0)


def test_native_pending_cancellation_and_clock_validation():
    tree = RadixTree()
    router = OfflineRouter(4, 1, 1, 128, 64, '{"router_queue_threshold":0.5}')
    tokens = [1] * 64
    hashes = compute_block_hash_for_seq(tokens, 4)
    for request_id in ("active", "cancelled"):
        router.submit(
            request_id, 64, 1, tree.find_matches(hashes), hashes, tokens=tokens
        )
    assert [row[0] for row in router.poll()] == ["active"]
    assert router.pending_count == 1
    assert router.cancel("cancelled")
    assert router.poll() == []
    router.submit("waiting", 64, 1, tree.find_matches(hashes), hashes, tokens=tokens)
    assert router.poll() == []
    router.advance_clock(5)
    router.prefill_complete("active")
    assert [row[0] for row in router.poll()] == ["waiting"]
    router.free("active")
    router.free("waiting")
    assert router.pending_count == 0
    for invalid in (4, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            router.advance_clock(invalid)


def test_seeded_native_selections_are_repeatable():
    tree = RadixTree()
    choices = []
    for _ in range(2):
        router = OfflineRouter(4, 4, 1, 128, selector_seed=19)
        choices.append([])
        for i in range(8):
            router.submit(str(i), 0, 1, tree.find_matches([]), [], [])
            choices[-1].append(router.poll()[0][1:3])
            router.free(str(i))
    assert choices[0] == choices[1]


@pytest.mark.timeout(30)
def test_scheduler_does_not_import_a_simulation_executor():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys
class BlockSimulator(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'aisimulate', 'aisimulate_core', 'aiconfigurator', 'aiconfigurator_core'}:
            raise AssertionError('unexpected simulator import: ' + fullname)
sys.meta_path.insert(0, BlockSimulator())
from dynamo._core import OfflineRouter, RadixTree
router = OfflineRouter(4, 1, 1, 128)
router.submit('request', 0, 1, RadixTree().find_matches([]), [], [])
assert router.poll()[0][:3] == ('request', 0, 0)
router.free('request')
""",
        ],
        check=True,
        timeout=20,
    )
