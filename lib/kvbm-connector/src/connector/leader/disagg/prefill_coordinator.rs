// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared prefill-side observer and status types used by the
//! new [`super::coordinator::ConditionalDisaggCoordinator`].
//!
//! The legacy `PrefillCoordinator` trait and `PrefillCoordinatorImpl`
//! production struct have been removed; the unified
//! `ConditionalDisaggCoordinator` now owns all prefill-side per-request
//! state and drives this observer directly.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Weak};

use kvbm_logical::blocks::ImmutableBlock;
use parking_lot::Mutex;

use crate::{G2, SequenceHash};

// ============================================================================
// PrefillStatus
// ============================================================================

/// Per-request prefill-side state machine state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrefillStatus {
    Attaching,
    Pulling,
    Registered,
    OnboardingScheduled,
    OnboardingComplete,
    SlotDone,
    Released,
}

// ============================================================================
// ConditionalDecodeG2Observer
// ============================================================================
//
// One observer instance, registered ONCE with the offload pipeline at
// coordinator construction. Holds per-request residual hash sets; as the
// pipeline registers G2 blocks, the observer pops matched hashes from the
// matching request's set and forwards the matched blocks to
// `commit_output_blocks`. When a request's residual goes empty, its entry
// is auto-dropped.
//
// `break;` on first hash match: if two simultaneous requests share an
// expected output hash (rare; would mean identical continuation blocks),
// the first iterator-ordered request claims. Document and revisit only
// if production hits the case.

/// Closure invoked by the observer once expected output blocks
/// have been registered for a request.  Signature mirrors
/// `commit_output_blocks(request_id, blocks)`.  Decoupled from any
/// specific coordinator type so the new `ConditionalDisaggCoordinator`
/// can drive the same observer.
pub type CommitOutputBlocksFn =
    dyn Fn(&str, Vec<ImmutableBlock<G2>>) + Send + Sync + 'static;

pub struct ConditionalDecodeG2Observer {
    /// Per-request residual hashset: hashes we still expect to
    /// see registered for this request. Entries removed as
    /// matches land. Empty entry → dropped.
    pending: Mutex<HashMap<String, HashSet<SequenceHash>>>,
    /// Closure that forwards matched blocks to the coordinator's
    /// `commit_output_blocks`.  Held as `Arc` so the closure can
    /// capture the coord via `Weak<Self>` without forcing the
    /// observer to know the coord's concrete type.  `None` until a
    /// coordinator installs itself via [`install_commit_fn`].
    commit_fn: Mutex<Option<Arc<CommitOutputBlocksFn>>>,
}

impl ConditionalDecodeG2Observer {
    pub fn new() -> Arc<Self> {
        Arc::new(Self {
            pending: Mutex::new(HashMap::new()),
            commit_fn: Mutex::new(None),
        })
    }

    /// Install the coordinator's commit-output-blocks dispatcher.
    /// Called by the coordinator's constructor (`Arc::new_cyclic`)
    /// with a closure that captures `Weak<coord>` and re-upgrades
    /// per call.  Idempotent — overwrites any prior installation
    /// (production wiring sets it exactly once; tests may swap).
    pub fn install_commit_fn(&self, f: Arc<CommitOutputBlocksFn>) {
        *self.commit_fn.lock() = Some(f);
    }

    /// Track a request's expected output hashes; returns an RAII
    /// [`ObserverHandle`] whose drop evicts the residual entry.
    /// Production calls this at `ensure_started` time and stashes
    /// the handle in per-request state. Re-tracking with a different
    /// set replaces the prior entry.
    pub fn track(
        self: &Arc<Self>,
        request_id: String,
        expected: HashSet<SequenceHash>,
    ) -> ObserverHandle {
        self.pending.lock().insert(request_id.clone(), expected);
        ObserverHandle {
            request_id,
            observer: Arc::downgrade(self),
        }
    }

    /// Drop residual entry for a request. Production no longer
    /// calls this directly — the [`ObserverHandle`] returned by
    /// [`track`](Self::track) handles it on drop. Kept public for
    /// test scenarios that exercise manual eviction; idempotent
    /// against auto-drop-on-full-match.
    pub fn untrack_request(&self, request_id: &str) {
        self.pending.lock().remove(request_id);
    }

    /// Whether the observer is still waiting for at least one hash
    /// to be registered for `request_id`.  Returns `false` when the
    /// entry is absent (already drained by full-match auto-remove,
    /// untracked, or never tracked).
    pub fn has_pending(&self, request_id: &str) -> bool {
        self.pending
            .lock()
            .get(request_id)
            .map(|hashes| !hashes.is_empty())
            .unwrap_or(false)
    }

    /// Test accessor: snapshot of remaining hashes for `request_id`.
    #[cfg(any(test, feature = "testing"))]
    pub fn pending_for(&self, request_id: &str) -> Option<HashSet<SequenceHash>> {
        self.pending.lock().get(request_id).cloned()
    }

    /// Test accessor: count of tracked requests.
    #[cfg(any(test, feature = "testing"))]
    pub fn tracked_count(&self) -> usize {
        self.pending.lock().len()
    }

    /// Called by the offload pipeline's register-observer path
    /// after each batch of G2 blocks is registered.
    ///
    /// Lock split: matches are computed under the `pending`
    /// guard, then the guard is dropped before
    /// `commit_output_blocks` is called (which acquires session
    /// locks). Avoids cross-lock hazards.
    pub fn observe(&self, blocks: &[ImmutableBlock<G2>]) {
        let mut by_request: HashMap<String, Vec<ImmutableBlock<G2>>> = HashMap::new();
        let mut empty_after: Vec<String> = Vec::new();
        {
            let mut pending = self.pending.lock();
            for block in blocks {
                let hash = block.sequence_hash();
                // Linear scan over active CD requests. N expected
                // small; if profiling shows hot, swap for a
                // SequenceHash → request_id reverse index updated
                // alongside `pending` mutations.
                for (request_id, hashes) in pending.iter_mut() {
                    if hashes.remove(&hash) {
                        by_request
                            .entry(request_id.clone())
                            .or_default()
                            .push(block.clone());
                        break;
                    }
                }
            }
            for (request_id, hashes) in pending.iter() {
                if hashes.is_empty() {
                    empty_after.push(request_id.clone());
                }
            }
            for request_id in &empty_after {
                pending.remove(request_id);
            }
        }

        let dispatch = self.commit_fn.lock().clone();
        if let Some(f) = dispatch {
            for (request_id, blocks) in by_request {
                f(&request_id, blocks);
            }
        }
    }
}

/// RAII handle returned by [`ConditionalDecodeG2Observer::track`].
/// On drop, evicts the per-request residual entry. Held by
/// per-request state in the coordinator, so any path that drops the
/// request's state (success, failure, panic mid-pipeline) cleans up
/// automatically.
pub struct ObserverHandle {
    request_id: String,
    observer: Weak<ConditionalDecodeG2Observer>,
}

impl Drop for ObserverHandle {
    fn drop(&mut self) {
        if let Some(observer) = self.observer.upgrade() {
            observer.untrack_request(&self.request_id);
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::G2;
    use kvbm_engine::testing::managers::{TestManagerBuilder, TestRegistryBuilder};
    use kvbm_engine::testing::token_blocks::create_token_sequence;
    use kvbm_logical::manager::BlockManager;

    const BLOCK_SIZE: usize = 16;

    fn make_g2_manager(capacity: usize) -> Arc<BlockManager<G2>> {
        let registry = TestRegistryBuilder::new().build();
        Arc::new(
            TestManagerBuilder::<G2>::new()
                .block_count(capacity)
                .block_size(BLOCK_SIZE)
                .registry(registry)
                .build(),
        )
    }

    fn make_blocks(
        g2: &Arc<BlockManager<G2>>,
        count: usize,
        start_token: u32,
    ) -> Vec<ImmutableBlock<G2>> {
        let token_sequence = create_token_sequence(count, BLOCK_SIZE, start_token);
        let mutables = g2.allocate_blocks(count).expect("alloc");
        let completes: Vec<_> = mutables
            .into_iter()
            .zip(token_sequence.blocks().iter())
            .map(|(m, tb)| m.complete(tb).expect("complete"))
            .collect();
        g2.register_blocks(completes)
    }

    /// Mixed-batch dispatch: two requests tracked with disjoint
    /// expected-hash sets; one batch contains blocks for both
    /// plus an unrelated block. Each request's residual should
    /// be drained only by its own matches; the unrelated block
    /// is silently ignored.
    #[test]
    fn mixed_batch_routes_per_request_and_ignores_unrelated() {
        let observer = ConditionalDecodeG2Observer::new();
        // No coordinator installed — observe()'s commit_output_blocks
        // dispatch is a silent no-op. We test the pending-state
        // transitions, not the downstream dispatch.

        let mgr = make_g2_manager(8);
        let a_blocks = make_blocks(&mgr, 2, 0); // hashes for "a"
        let b_blocks = make_blocks(&mgr, 1, 100); // hashes for "b"
        let unrelated = make_blocks(&mgr, 1, 9000); // not tracked

        let a_hashes: HashSet<_> = a_blocks.iter().map(|b| b.sequence_hash()).collect();
        let b_hashes: HashSet<_> = b_blocks.iter().map(|b| b.sequence_hash()).collect();

        let _h_a = observer.track("a".into(), a_hashes.clone());
        let _h_b = observer.track("b".into(), b_hashes.clone());
        assert_eq!(observer.tracked_count(), 2);

        // Mixed batch: 1 of "a"'s blocks, "b"'s block, unrelated.
        let batch: Vec<_> = vec![
            a_blocks[0].clone(),
            b_blocks[0].clone(),
            unrelated[0].clone(),
        ];
        observer.observe(&batch);

        // "a" still has 1 hash residual (a_blocks[1] not yet seen).
        let pending_a = observer.pending_for("a").expect("a still tracked");
        assert_eq!(pending_a.len(), 1);
        assert!(pending_a.contains(&a_blocks[1].sequence_hash()));

        // "b" auto-dropped (its only hash matched).
        assert!(
            observer.pending_for("b").is_none(),
            "b should be auto-dropped after full match"
        );
        assert_eq!(observer.tracked_count(), 1);

        // Second batch with "a"'s remaining block — "a" should auto-drop.
        observer.observe(&[a_blocks[1].clone()]);
        assert!(observer.pending_for("a").is_none());
        assert_eq!(observer.tracked_count(), 0);
    }

    /// untrack_request evicts a residual entry that didn't fully
    /// drain (failure path).
    #[test]
    fn untrack_request_evicts_partial_residual() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(4);
        let blocks = make_blocks(&mgr, 3, 200);
        let hashes: HashSet<_> = blocks.iter().map(|b| b.sequence_hash()).collect();

        // Hold the handle alive so RAII drop doesn't pre-empt
        // the manual untrack we're testing.
        let _h = observer.track("req-fail".into(), hashes);
        // Observe only 1 of 3 — residual non-empty.
        observer.observe(&[blocks[0].clone()]);
        assert_eq!(observer.pending_for("req-fail").unwrap().len(), 2);

        observer.untrack_request("req-fail");
        assert!(observer.pending_for("req-fail").is_none());
    }

    /// RAII observer handle: dropping the handle evicts the
    /// residual entry from `pending`. Models the production drop
    /// path where the coordinator's per-request state holds the
    /// handle and is dropped alongside the per-request state.
    #[test]
    fn observer_handle_drop_evicts_pending_entry() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(4);
        let blocks = make_blocks(&mgr, 2, 400);
        let hashes: HashSet<_> = blocks.iter().map(|b| b.sequence_hash()).collect();

        let handle = observer.track("req-raii".into(), hashes);
        assert_eq!(observer.tracked_count(), 1);
        assert!(observer.pending_for("req-raii").is_some());

        drop(handle);

        assert_eq!(observer.tracked_count(), 0);
        assert!(observer.pending_for("req-raii").is_none());
    }

    /// Dropping a handle whose entry was already auto-removed
    /// (full match) is a safe no-op — the underlying
    /// `untrack_request` is idempotent.
    #[test]
    fn observer_handle_drop_after_auto_remove_is_noop() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(2);
        let blocks = make_blocks(&mgr, 1, 500);
        let hash = blocks[0].sequence_hash();

        let handle = observer.track("a".into(), [hash].into_iter().collect());
        // Full match — entry auto-drops from `pending`.
        observer.observe(&[blocks[0].clone()]);
        assert_eq!(observer.tracked_count(), 0);

        // Handle drop should not panic, observer state unchanged.
        drop(handle);
        assert_eq!(observer.tracked_count(), 0);
    }

    /// re-observing an already-popped block is a no-op
    /// (idempotent against offload P-rule re-emission).
    #[test]
    fn re_observe_is_idempotent() {
        let observer = ConditionalDecodeG2Observer::new();
        let mgr = make_g2_manager(2);
        let blocks = make_blocks(&mgr, 1, 300);
        let hash = blocks[0].sequence_hash();

        let _h = observer.track("a".into(), [hash].into_iter().collect());
        observer.observe(&[blocks[0].clone()]);
        // Auto-dropped after full match.
        assert_eq!(observer.tracked_count(), 0);

        // Re-observing the same block: no panic, no resurrection
        // of the dropped entry.
        observer.observe(&[blocks[0].clone()]);
        assert_eq!(observer.tracked_count(), 0);
    }
}
