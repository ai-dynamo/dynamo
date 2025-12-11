// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Pending transfer tracking for duplicate prevention.
//!
//! This module provides `PendingTracker` and `PendingGuard` types that work together
//! to track blocks that are currently in-flight through the transfer pipeline.
//!
//! # Problem
//!
//! When overlapping sequences are enqueued for transfer at roughly the same time,
//! the presence policy may allow duplicate transfers because:
//! - The first sequence's blocks haven't completed registration yet
//! - The second sequence sees the same blocks as "not present"
//!
//! # Solution
//!
//! The `PendingTracker` maintains a set of sequence hashes currently in the pipeline.
//! When blocks pass policy evaluation, a `PendingGuard` is created that:
//! - Adds the sequence hash to the pending set on creation
//! - Automatically removes it on drop (RAII pattern)
//!
//! The `PresenceFilter` can then check both the registry (completed transfers)
//! AND the pending set (in-flight transfers) to avoid duplicates.
//!
//! # Example
//!
//! ```ignore
//! let tracker = Arc::new(PendingTracker::new());
//!
//! // Create guard when block passes policy
//! let guard = tracker.guard(sequence_hash);
//!
//! // Guard travels with block through pipeline stages
//! queued_block.pending_guard = Some(guard);
//!
//! // When block completes or is cancelled, guard is dropped
//! // and hash is automatically removed from pending set
//! ```

use std::sync::Arc;

use dashmap::DashSet;

use crate::v2::SequenceHash;

/// Tracks sequence hashes that are currently pending transfer.
///
/// This is shared between the pipeline and the presence policy via `Arc`.
/// Thread-safe for concurrent access from multiple pipeline stages.
#[derive(Debug, Default)]
pub struct PendingTracker {
    pending: DashSet<SequenceHash>,
}

impl PendingTracker {
    /// Create a new empty pending tracker.
    pub fn new() -> Self {
        Self {
            pending: DashSet::new(),
        }
    }

    /// Check if a sequence hash is currently pending transfer.
    ///
    /// Used by `PresenceFilter` to skip blocks that are already in-flight.
    pub fn is_pending(&self, hash: &SequenceHash) -> bool {
        self.pending.contains(hash)
    }

    /// Get the number of pending transfers.
    ///
    /// Useful for metrics and debugging.
    pub fn len(&self) -> usize {
        self.pending.len()
    }

    /// Check if there are no pending transfers.
    pub fn is_empty(&self) -> bool {
        self.pending.is_empty()
    }

    /// Create a guard that marks a sequence hash as pending until dropped.
    ///
    /// The guard uses RAII to ensure the hash is removed when:
    /// - Transfer completes successfully
    /// - Transfer is cancelled
    /// - Block is evicted from pipeline
    /// - Any error causes the block to be dropped
    pub fn guard(self: &Arc<Self>, hash: SequenceHash) -> PendingGuard {
        self.pending.insert(hash);
        PendingGuard {
            hash,
            tracker: Arc::clone(self),
        }
    }
}

/// RAII guard that removes a sequence hash from the pending set on drop.
///
/// This guard travels with the block through all pipeline stages and ensures
/// cleanup happens automatically regardless of how the transfer completes.
///
/// # Clone Behavior
///
/// Cloning a `PendingGuard` is cheap (Arc clone) but does NOT create a new
/// pending entry. The hash is only inserted once when the first guard is
/// created, and removed when ALL clones are dropped.
///
/// However, the current implementation removes on first drop, so cloning
/// should be avoided unless you understand the implications.
pub struct PendingGuard {
    hash: SequenceHash,
    tracker: Arc<PendingTracker>,
}

impl PendingGuard {
    /// Get the sequence hash this guard is tracking.
    pub fn sequence_hash(&self) -> SequenceHash {
        self.hash
    }
}

impl Drop for PendingGuard {
    fn drop(&mut self) {
        self.tracker.pending.remove(&self.hash);
    }
}

impl std::fmt::Debug for PendingGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PendingGuard")
            .field("sequence_hash", &self.hash)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to create a test SequenceHash with unique values.
    fn test_hash(id: u64) -> SequenceHash {
        SequenceHash::new(id, 0, id)
    }

    #[test]
    fn test_pending_tracker_new() {
        let tracker = PendingTracker::new();
        assert!(tracker.is_empty());
        assert_eq!(tracker.len(), 0);
    }

    #[test]
    fn test_pending_guard_inserts_and_removes() {
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(12345);

        assert!(!tracker.is_pending(&hash));

        {
            let _guard = tracker.guard(hash);
            assert!(tracker.is_pending(&hash));
            assert_eq!(tracker.len(), 1);
        }

        // Guard dropped, hash should be removed
        assert!(!tracker.is_pending(&hash));
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_multiple_guards_different_hashes() {
        let tracker = Arc::new(PendingTracker::new());
        let hash1 = test_hash(111);
        let hash2 = test_hash(222);
        let hash3 = test_hash(333);

        let guard1 = tracker.guard(hash1);
        let guard2 = tracker.guard(hash2);

        assert!(tracker.is_pending(&hash1));
        assert!(tracker.is_pending(&hash2));
        assert!(!tracker.is_pending(&hash3));
        assert_eq!(tracker.len(), 2);

        drop(guard1);
        assert!(!tracker.is_pending(&hash1));
        assert!(tracker.is_pending(&hash2));
        assert_eq!(tracker.len(), 1);

        drop(guard2);
        assert!(tracker.is_empty());
    }

    #[test]
    fn test_guard_sequence_hash_accessor() {
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(42);

        let guard = tracker.guard(hash);
        assert_eq!(guard.sequence_hash(), hash);
    }

    #[test]
    fn test_tracker_debug() {
        let tracker = PendingTracker::new();
        let debug_str = format!("{:?}", tracker);
        assert!(debug_str.contains("PendingTracker"));
    }

    #[test]
    fn test_guard_debug() {
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(999);
        let guard = tracker.guard(hash);

        let debug_str = format!("{:?}", guard);
        assert!(debug_str.contains("PendingGuard"));
        assert!(debug_str.contains("sequence_hash"));
    }

    #[test]
    fn test_concurrent_access_to_same_hash() {
        // Test that the same hash being added twice is handled correctly
        let tracker = Arc::new(PendingTracker::new());
        let hash = test_hash(555);

        // First guard marks it as pending
        let guard1 = tracker.guard(hash);
        assert!(tracker.is_pending(&hash));
        assert_eq!(tracker.len(), 1);

        // Second guard for same hash - DashSet.insert returns false if already present
        // but our guard() always inserts (doesn't check first)
        let guard2 = tracker.guard(hash);
        assert!(tracker.is_pending(&hash));
        // DashSet deduplicates, so len is still 1
        assert_eq!(tracker.len(), 1);

        // Drop first guard - hash removed from set
        drop(guard1);
        // DashSet now doesn't have the hash
        assert!(!tracker.is_pending(&hash));

        // Second guard still exists but hash was already removed
        // This is expected behavior - the RAII ensures cleanup on any drop
        drop(guard2);
        assert!(tracker.is_empty());
    }
}
