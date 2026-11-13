# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the hello_engine example's KV block publishing
(examples/custom_backend/hello_world/engine).

The contract under test: on every request the engine publishes the
prompt's full run of complete blocks from position 0, with per-worker
unique node IDs — the router matches on the token content it re-hashes
itself, so republishing is idempotent and needs no bookkeeping.
"""

import os
import sys

import pytest

pytestmark = [
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.unit,
]

_ENGINE_SRC = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../examples/custom_backend/hello_world/engine/src",
    )
)


class _FakePublisher:
    def __init__(self):
        self.calls = []

    def publish_stored(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture()
def engine():
    sys.path.insert(0, _ENGINE_SRC)
    try:
        from hello_engine.engine import HelloEngine

        eng = HelloEngine.__new__(HelloEngine)
        eng._publisher = _FakePublisher()
        eng._next_block_id = 0
        yield eng
    finally:
        sys.path.remove(_ENGINE_SRC)


def _block_size():
    from hello_engine.engine import BLOCK_SIZE

    return BLOCK_SIZE


def test_publishes_full_run_from_position_zero(engine):
    b = _block_size()
    prompt = list(range(3 * b))
    engine._publish_prompt_blocks(prompt)

    (call,) = engine._publisher.calls
    assert call["parent_hash"] is None
    assert call["token_ids"] == prompt
    assert call["num_block_tokens"] == [b, b, b]
    assert len(call["block_hashes"]) == 3


def test_repeat_republishes_idempotently_with_fresh_ids(engine):
    b = _block_size()
    prompt = list(range(3 * b))
    engine._publish_prompt_blocks(prompt)
    engine._publish_prompt_blocks(prompt)

    first, second = engine._publisher.calls
    # Same tokens (that is what the router matches on) ...
    assert second["token_ids"] == first["token_ids"]
    # ... but node IDs never repeat within the worker.
    assert not set(first["block_hashes"]) & set(second["block_hashes"])


def test_block_ids_are_unique_across_prompts(engine):
    b = _block_size()
    engine._publish_prompt_blocks(list(range(2 * b)))
    engine._publish_prompt_blocks(list(range(1000, 1000 + 2 * b)))

    seen = [h for c in engine._publisher.calls for h in c["block_hashes"]]
    assert len(seen) == len(set(seen)) == 4


def test_partial_final_block_is_not_published(engine):
    b = _block_size()
    prompt = list(range(b + b // 2))  # one full block + a partial
    engine._publish_prompt_blocks(prompt)

    (call,) = engine._publisher.calls
    assert len(call["block_hashes"]) == 1
    assert call["token_ids"] == prompt[:b]


def test_short_prompt_publishes_nothing(engine):
    b = _block_size()
    engine._publish_prompt_blocks(list(range(b - 1)))
    assert engine._publisher.calls == []


def test_null_publisher_is_safe(engine):
    engine._publisher = None
    engine._publish_prompt_blocks(list(range(64)))  # must not raise
