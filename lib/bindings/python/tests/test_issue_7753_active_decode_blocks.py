# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Test for issue #7753: active_decode_blocks_threshold broken by dual publishers.

Two separate code paths both publish ActiveLoad events with active_decode_blocks to the
same NATS subject (KV_METRICS_SUBJECT):
  1. WorkerMetricsPublisher (engine-side) - the authoritative source
  2. RuntimeSequencePublisher::publish_load() (router-side, via publish_active_load_for_worker)

The last-write-wins behavior means the KvWorkerMonitor sees an inconsistent mix of values,
making active_decode_blocks_threshold-based request rejection unreliable.

Fix: RuntimeSequencePublisher should set active_decode_blocks=None in the ActiveLoad it emits,
so only the engine-side publisher controls this field.
"""

import os
import re

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.gpu_0,
]

# Path to the router-side sequence publisher (relative to this file)
MULTI_WORKER_RS = os.path.normpath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../kv-router/src/sequences/multi_worker.rs",
    )
)


def test_router_does_not_publish_active_decode_blocks():
    """
    Verify that the router-side publisher (publish_active_load_for_worker) does NOT
    include active_decode_blocks in its ActiveLoad payload.

    Before the fix, publish_active_load_for_worker() set:
        active_decode_blocks: Some(active_blocks as u64),

    This raced with the engine-side WorkerMetricsPublisher which publishes the
    authoritative KV cache occupancy. The KvWorkerMonitor (worker_monitor.rs) applies
    last-write-wins, so whichever publisher fires last wins - making the
    active_decode_blocks_threshold-based busy detection unreliable.

    After the fix, publish_active_load_for_worker() must set:
        active_decode_blocks: None,

    So only WorkerMetricsPublisher (engine-side) controls this field.
    """
    assert os.path.exists(MULTI_WORKER_RS), (
        f"Source file not found: {MULTI_WORKER_RS}\n"
        "Run this test from the repository root."
    )

    with open(MULTI_WORKER_RS) as f:
        source = f.read()

    # Find the publish_active_load_for_worker function body
    fn_match = re.search(
        r"fn publish_active_load_for_worker\b.*?(?=\n    /// |\n    pub fn |\n    fn |\Z)",
        source,
        re.DOTALL,
    )
    assert fn_match is not None, (
        "Could not find publish_active_load_for_worker function in multi_worker.rs"
    )
    fn_body = fn_match.group(0)

    # The fix: active_decode_blocks must be None (not Some(...))
    # Before fix: active_decode_blocks: Some(active_blocks as u64),
    # After fix:  active_decode_blocks: None,
    assert "active_decode_blocks: None," in fn_body, (
        "Bug #7753: publish_active_load_for_worker() in multi_worker.rs still sets "
        "active_decode_blocks to Some(...) instead of None.\n\n"
        "This causes the router-side publisher to race with the engine-side "
        "WorkerMetricsPublisher on the KV_METRICS_SUBJECT NATS subject. "
        "The KvWorkerMonitor applies last-write-wins, so the "
        "active_decode_blocks_threshold-based busy detection sees an inconsistent mix "
        "of engine-reported and router-bookkeeping values.\n\n"
        "Fix: set active_decode_blocks: None in the ActiveLoad emitted by "
        "publish_active_load_for_worker(), so only WorkerMetricsPublisher controls "
        "this field.\n\n"
        f"Found in function:\n{fn_body}"
    )

    # Also confirm that active_prefill_tokens is still published by the router
    # (this is the router's authoritative value - engine publishes None for this field)
    assert "active_prefill_tokens: Some(" in fn_body, (
        "publish_active_load_for_worker() should still publish active_prefill_tokens "
        "since the router is the authoritative source for that field."
    )


def test_worker_load_state_is_busy_logic():
    """
    Regression test for the is_busy() decision logic.

    Simulates the race condition: engine publishes active_decode_blocks=90 (busy at 90%),
    then router overwrites with active_decode_blocks=10 (its own bookkeeping, much lower).
    Without the fix, the monitor sees 10 blocks → not busy → requests NOT rejected.

    With the fix (router sets active_decode_blocks=None), the engine's value (90) is
    preserved and busy detection works correctly.
    """

    class WorkerLoadState:
        """Python mirror of the Rust WorkerLoadState struct for logic verification."""

        def __init__(self):
            self.active_decode_blocks = {}
            self.kv_total_blocks = {}
            self.active_prefill_tokens = {}

        def apply_active_load(
            self,
            dp_rank: int,
            active_decode_blocks,
            active_prefill_tokens,
        ):
            """Apply an ActiveLoad event (None fields are ignored, matching Rust behavior)."""
            if active_decode_blocks is not None:
                self.active_decode_blocks[dp_rank] = active_decode_blocks
            if active_prefill_tokens is not None:
                self.active_prefill_tokens[dp_rank] = active_prefill_tokens

        def is_busy(self, active_decode_blocks_threshold: float) -> bool:
            """Simplified is_busy() matching the Rust logic for the blocks threshold."""
            all_dp_ranks = set(self.active_decode_blocks.keys())
            if not all_dp_ranks:
                return False
            return all(
                (self.kv_total_blocks.get(dp) or 0) > 0
                and self.active_decode_blocks.get(dp, 0)
                > active_decode_blocks_threshold * (self.kv_total_blocks.get(dp) or 1)
                for dp in all_dp_ranks
            )

    # Scenario: 100 total blocks, threshold=0.85 (busy when >85 blocks used)
    threshold = 0.85

    # --- BUG SCENARIO (before fix) ---
    buggy_state = WorkerLoadState()
    buggy_state.kv_total_blocks[0] = 100

    # Engine publishes authoritative value: 90 blocks in use (should be busy)
    buggy_state.apply_active_load(
        0, active_decode_blocks=90, active_prefill_tokens=None
    )
    assert buggy_state.is_busy(threshold), "Engine-only: should be busy at 90/100"

    # Router OVERWRITES with its own bookkeeping value: 10 blocks (router's view is stale/different)
    buggy_state.apply_active_load(0, active_decode_blocks=10, active_prefill_tokens=50)
    # BUG: now the monitor sees 10 blocks → not busy → requests NOT rejected even though
    # the engine has 90% KV cache usage!
    assert not buggy_state.is_busy(threshold), (
        "Confirms bug: after router overwrites, incorrectly appears not busy (10/100 < 0.85)"
    )

    # --- FIXED SCENARIO (after fix) ---
    fixed_state = WorkerLoadState()
    fixed_state.kv_total_blocks[0] = 100

    # Engine publishes authoritative value: 90 blocks in use (should be busy)
    fixed_state.apply_active_load(
        0, active_decode_blocks=90, active_prefill_tokens=None
    )

    # Router publishes with active_decode_blocks=None (fixed) - only updates prefill tokens
    fixed_state.apply_active_load(
        0, active_decode_blocks=None, active_prefill_tokens=50
    )

    # FIXED: engine's value (90) is preserved, busy detection works correctly
    assert fixed_state.is_busy(threshold), (
        "After fix: engine's active_decode_blocks=90 preserved (router sent None), "
        "correctly detected as busy at 90/100 > 0.85"
    )
    # Prefill tokens were updated by the router
    assert fixed_state.active_prefill_tokens.get(0) == 50
