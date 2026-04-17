// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared stale-child cleanup machinery for rooted tree structures.
//!
//! Provides a throttled, one-in-flight sweep that unlinks empty child nodes
//! from their parent. It is used by [`ConcurrentRadixTree`](super::concurrent_radix_tree),
//! [`ConcurrentRadixTreeCompressed`](super::concurrent_radix_tree_compressed)
//! and the sequence-side
//! [`PromptMembershipTrie`](super::sequences::prompt_membership_trie::PromptMembershipTrie),
//! each of which embeds a [`CleanupState`] and implements [`CleanableNode`]
//! for its node type.
//!
//! # Sweep semantics
//!
//! [`sweep_stale_children`] is a reverse-BFS prune:
//! - BFS from the root under read locks, collecting `(parent_weak, key, child_weak)` edges.
//! - Iterate edges deepest-first so children are swept before parents.
//! - For each edge: upgrade weaks, take the parent write lock, verify the
//!   child pointer still matches, `try_write` the child, and unlink only when
//!   the child has no workers, no children, and `Arc::strong_count == 2`
//!   (parent map ref + our local upgrade). The strong-count gate is what
//!   prevents reclaiming a node that a concurrent `find_matches` is currently
//!   traversing — such edges are skipped and retried on the next sweep.

use std::collections::VecDeque;
use std::hash::Hash;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Weak};
use std::time::Instant;

use parking_lot::RwLock;
use rustc_hash::FxHashMap;

pub const CLEANUP_INTERVAL_MS: u64 = 5 * 60 * 1000;

/// Node type that participates in the reverse-BFS cleanup sweep.
pub trait CleanableNode: Sized + Send + Sync + 'static {
    /// Key type used in this node's children map (e.g. `LocalBlockHash`,
    /// `SequenceHash`).
    type ChildKey: Copy + Eq + Hash + Send + Sync + 'static;

    /// True if this node still carries worker state that pins it in the tree.
    fn has_any_workers(&self) -> bool;

    /// Read-only view of this node's children keyed by the first edge element.
    fn children(&self) -> &FxHashMap<Self::ChildKey, Arc<RwLock<Self>>>;

    /// Unlink a child edge.
    fn remove_child(&mut self, key: &Self::ChildKey);
}

pub struct CleanupState {
    clock_origin: Instant,
    last_cleanup_elapsed_ms: AtomicU64,
    scheduled: AtomicBool,
}

impl CleanupState {
    pub fn new() -> Self {
        Self {
            clock_origin: Instant::now(),
            last_cleanup_elapsed_ms: AtomicU64::new(0),
            scheduled: AtomicBool::new(false),
        }
    }

    pub fn elapsed_ms(&self) -> u64 {
        self.clock_origin.elapsed().as_millis() as u64
    }

    pub fn try_schedule(&self) -> bool {
        let now_ms = self.elapsed_ms();
        let last_ms = self.last_cleanup_elapsed_ms.load(Ordering::Relaxed);
        if now_ms.saturating_sub(last_ms) < CLEANUP_INTERVAL_MS {
            return false;
        }

        self.scheduled
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    pub fn cancel(&self) {
        self.scheduled.store(false, Ordering::Release);
    }
}

impl Default for CleanupState {
    fn default() -> Self {
        Self::new()
    }
}

pub struct CleanupGuard<'a> {
    state: &'a CleanupState,
    completed_elapsed_ms: Option<u64>,
}

impl<'a> CleanupGuard<'a> {
    pub fn new(state: &'a CleanupState) -> Self {
        Self {
            state,
            completed_elapsed_ms: None,
        }
    }

    pub fn mark_completed(&mut self) {
        self.completed_elapsed_ms = Some(self.state.elapsed_ms());
    }
}

impl Drop for CleanupGuard<'_> {
    fn drop(&mut self) {
        if let Some(elapsed_ms) = self.completed_elapsed_ms {
            self.state
                .last_cleanup_elapsed_ms
                .store(elapsed_ms, Ordering::Relaxed);
        }
        self.state.scheduled.store(false, Ordering::Release);
    }
}

struct CleanupEdge<N: CleanableNode> {
    parent: Weak<RwLock<N>>,
    key: N::ChildKey,
    child: Weak<RwLock<N>>,
}

/// Reverse-BFS sweep that unlinks empty, unreferenced leaf nodes from the tree.
pub fn sweep_stale_children<N: CleanableNode>(root: &Arc<RwLock<N>>) {
    let mut queue: VecDeque<Arc<RwLock<N>>> = VecDeque::from([root.clone()]);
    let mut edges: Vec<CleanupEdge<N>> = Vec::new();

    while let Some(parent) = queue.pop_front() {
        let guard = parent.read();
        for (&key, child) in guard.children() {
            queue.push_back(child.clone());
            edges.push(CleanupEdge {
                parent: Arc::downgrade(&parent),
                key,
                child: Arc::downgrade(child),
            });
        }
    }

    for edge in edges.into_iter().rev() {
        let (Some(parent), Some(child)) = (edge.parent.upgrade(), edge.child.upgrade()) else {
            continue;
        };

        let mut parent_guard = parent.write();
        let still_attached = parent_guard
            .children()
            .get(&edge.key)
            .is_some_and(|current| Arc::ptr_eq(current, &child));
        if !still_attached {
            continue;
        }

        let Some(child_guard) = child.try_write() else {
            continue;
        };
        if child_guard.has_any_workers() || !child_guard.children().is_empty() {
            continue;
        }
        if Arc::strong_count(&child) != 2 {
            continue;
        }

        parent_guard.remove_child(&edge.key);
        drop(child_guard);
    }
}
