# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the hello_engine example's KV block publishing
(examples/custom_backend/hello_world/engine).

These cover the chained-hash contract the example exists to teach:
block hashes fold in their parent's hash, and publishes carry
``parent_hash`` so the router's radix tree links new blocks under the
shared prefix instead of anchoring everything at the root.
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
        eng._published_blocks = set()
        yield eng
    finally:
        sys.path.remove(_ENGINE_SRC)


def _block_size():
    from hello_engine.engine import BLOCK_SIZE

    return BLOCK_SIZE


def test_fresh_prompt_publishes_full_chain_from_root(engine):
    b = _block_size()
    prompt = list(range(3 * b))
    engine._publish_prompt_blocks(prompt)

    (call,) = engine._publisher.calls
    assert len(call["block_hashes"]) == 3
    assert call["parent_hash"] is None
    assert call["token_ids"] == prompt
    assert call["num_block_tokens"] == [b, b, b]


def test_exact_repeat_publishes_nothing(engine):
    b = _block_size()
    prompt = list(range(3 * b))
    engine._publish_prompt_blocks(prompt)
    engine._publish_prompt_blocks(prompt)

    assert len(engine._publisher.calls) == 1


def test_shared_prefix_publishes_tail_under_parent(engine):
    b = _block_size()
    prompt_a = list(range(3 * b))
    engine._publish_prompt_blocks(prompt_a)
    first = engine._publisher.calls[0]

    # Shares the first two blocks, then diverges for two new ones.
    prompt_b = prompt_a[: 2 * b] + list(range(1000, 1000 + 2 * b))
    engine._publish_prompt_blocks(prompt_b)

    assert len(engine._publisher.calls) == 2
    tail = engine._publisher.calls[1]
    assert len(tail["block_hashes"]) == 2
    # The tail must chain under the last shared block, not the root.
    assert tail["parent_hash"] == first["block_hashes"][1]
    assert tail["token_ids"] == prompt_b[2 * b : 4 * b]


def test_same_content_at_different_position_does_not_collide(engine):
    b = _block_size()
    prompt_a = list(range(3 * b))
    engine._publish_prompt_blocks(prompt_a)

    # prompt_a's first block content appears at position 2 here. With
    # chained hashes it must be treated as a different block, not
    # deduplicated away.
    prompt_c = list(range(5000, 5000 + b)) + prompt_a[:b]
    engine._publish_prompt_blocks(prompt_c)

    tail = engine._publisher.calls[1]
    assert len(tail["block_hashes"]) == 2
    assert tail["parent_hash"] is None


def test_partial_final_block_is_not_published(engine):
    b = _block_size()
    prompt = list(range(b + b // 2))  # one full block + a partial
    engine._publish_prompt_blocks(prompt)

    (call,) = engine._publisher.calls
    assert len(call["block_hashes"]) == 1
    assert call["token_ids"] == prompt[:b]
