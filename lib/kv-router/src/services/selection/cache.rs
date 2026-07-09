// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-flight selection cache.
//!
//! `/select` computes everything needed to later book a reservation (the
//! chosen worker, the normalized sequence hashes, and the prefill and
//! output-token accounting). Caching that here lets a follow-up
//! `/reservations` call replay it by `selection_id` instead of re-sending
//! the prompt.
//!
//! Entries that are never claimed by a `create_reservation` (an abandoned or
//! failed request) are evicted inline: every [`SelectionCache::insert`] sweeps
//! expired entries and enforces a size cap.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, Instant};

use dynamo_tokens::SequenceHash;
use parking_lot::Mutex;

use crate::protocols::WorkerWithDpRank;

use super::types::SelectionKey;

/// How long a pending selection lives before it is evicted.
const SELECTION_CACHE_TTL: Duration = Duration::from_secs(120);

/// Upper bound on resident pending selections, evicting oldest first. Only
/// reached when many selects go unclaimed (e.g. a crashed client).
const SELECTION_CACHE_MAX_ENTRIES: usize = 4096;

/// Booking inputs captured by `select`, replayed by a later `create_reservation`
/// without re-sending the prompt. Single-use, hence not `Clone`.
pub(super) struct PendingSelection {
    pub key: SelectionKey,
    pub worker: WorkerWithDpRank,
    pub sequence_hashes: Vec<SequenceHash>,
    pub isl_tokens: usize,
    pub effective_prefill_tokens: usize,
    pub expected_output_tokens: Option<u32>,
    pub track_prefill_tokens: bool,
    pub lora_name: Option<String>,
}

struct Entry {
    selection: PendingSelection,
    inserted_at: Instant,
    generation: u64,
}

/// Pending selections are scoped to their (model, tenant), like every other
/// selection-service structure.
type CacheKey = (SelectionKey, String);

struct State {
    entries: HashMap<CacheKey, Entry>,
    /// Insertion-ordered `(key, generation)`. The TTL is uniform, so the front
    /// is always the oldest. A tuple goes stale when its entry is taken or its
    /// key re-inserted; eviction detects that by comparing `generation`.
    order: VecDeque<(CacheKey, u64)>,
    next_generation: u64,
}

impl State {
    /// Remove the entry for a tuple popped from `order`, unless the tuple is
    /// stale (entry already taken, or key re-inserted with a newer generation).
    fn remove_if_current(&mut self, key: &CacheKey, generation: u64) {
        if self
            .entries
            .get(key)
            .is_some_and(|entry| entry.generation == generation)
        {
            self.entries.remove(key);
        }
    }
}

/// Maps `(SelectionKey, selection_id)` -> the booking inputs computed during `select`.
pub(super) struct SelectionCache {
    ttl: Duration,
    max_entries: usize,
    state: Mutex<State>,
}

impl Default for SelectionCache {
    fn default() -> Self {
        Self::new(SELECTION_CACHE_TTL, SELECTION_CACHE_MAX_ENTRIES)
    }
}

impl SelectionCache {
    fn new(ttl: Duration, max_entries: usize) -> Self {
        Self {
            ttl,
            max_entries,
            state: Mutex::new(State {
                entries: HashMap::new(),
                order: VecDeque::new(),
                next_generation: 0,
            }),
        }
    }

    /// Record (or replace) the pending selection under `(selection.key, selection_id)`,
    /// sweeping expired entries and evicting oldest-first if over the cap.
    pub(super) fn insert(&self, selection_id: String, selection: PendingSelection, now: Instant) {
        let cache_key = (selection.key.clone(), selection_id);
        let mut state = self.state.lock();
        // Drop stale and expired front tuples so `order` stays bounded by the
        // number of live entries.
        while let Some((key, generation)) = state.order.front().cloned() {
            let live = state.entries.get(&key).is_some_and(|entry| {
                entry.generation == generation && now.duration_since(entry.inserted_at) <= self.ttl
            });
            if live {
                break;
            }
            state.order.pop_front();
            state.remove_if_current(&key, generation);
        }
        let generation = state.next_generation;
        state.next_generation += 1;
        state.order.push_back((cache_key.clone(), generation));
        state.entries.insert(
            cache_key,
            Entry {
                selection,
                inserted_at: now,
                generation,
            },
        );
        // Every live entry has a matching tuple, so this terminates.
        while state.entries.len() > self.max_entries {
            let Some((key, generation)) = state.order.pop_front() else {
                break;
            };
            state.remove_if_current(&key, generation);
        }
    }

    /// Remove and return the pending selection for `(key, reservation_id)`. An
    /// entry older than the TTL is treated as already gone (and dropped).
    pub(super) fn take(
        &self,
        key: &SelectionKey,
        reservation_id: &str,
        now: Instant,
    ) -> Option<PendingSelection> {
        let cache_key = (key.clone(), reservation_id.to_string());
        let entry = self.state.lock().entries.remove(&cache_key)?;
        if now.duration_since(entry.inserted_at) > self.ttl {
            return None;
        }
        Some(entry.selection)
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.state.lock().entries.len()
    }

    #[cfg(test)]
    fn order_len(&self) -> usize {
        self.state.lock().order.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TTL: Duration = Duration::from_secs(10);
    const CAP: usize = 2;

    fn cache() -> SelectionCache {
        SelectionCache::new(TTL, CAP)
    }

    fn key() -> SelectionKey {
        SelectionKey::new("model", "default")
    }

    fn pending(worker_id: u64) -> PendingSelection {
        PendingSelection {
            key: key(),
            worker: WorkerWithDpRank::new(worker_id, 0),
            sequence_hashes: vec![1, 2, 3],
            isl_tokens: 12,
            effective_prefill_tokens: 8,
            expected_output_tokens: Some(16),
            track_prefill_tokens: true,
            lora_name: None,
        }
    }

    #[test]
    fn take_returns_entry_exactly_once() {
        let cache = cache();
        let now = Instant::now();
        cache.insert("req-1".to_string(), pending(1), now);

        let taken = cache.take(&key(), "req-1", now).expect("entry present");
        assert_eq!(taken.worker.worker_id, 1);
        assert_eq!(taken.isl_tokens, 12);
        assert!(cache.take(&key(), "req-1", now).is_none());
    }

    #[test]
    fn take_misses_unknown_id() {
        assert!(cache().take(&key(), "nope", Instant::now()).is_none());
    }

    #[test]
    fn take_requires_matching_key() {
        let cache = cache();
        let now = Instant::now();
        cache.insert("req-1".to_string(), pending(1), now);

        let other = SelectionKey::new("other-model", "default");
        assert!(cache.take(&other, "req-1", now).is_none());
        // The mismatch does not consume the entry.
        assert!(cache.take(&key(), "req-1", now).is_some());
    }

    #[test]
    fn take_refuses_expired_entry() {
        let cache = cache();
        let inserted = Instant::now();
        cache.insert("req-1".to_string(), pending(1), inserted);

        let later = inserted + TTL + Duration::from_secs(1);
        assert!(cache.take(&key(), "req-1", later).is_none());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn insert_sweeps_expired_entries() {
        let cache = cache();
        let t0 = Instant::now();
        cache.insert("old".to_string(), pending(1), t0);

        // The next insert reclaims everything past the TTL.
        let later = t0 + TTL + Duration::from_secs(1);
        cache.insert("new".to_string(), pending(2), later);
        assert_eq!(cache.len(), 1);
        assert!(cache.take(&key(), "old", later).is_none());
        assert!(cache.take(&key(), "new", later).is_some());
    }

    #[test]
    fn sweep_skips_stale_tuples_of_taken_entries() {
        let cache = cache();
        let t0 = Instant::now();
        cache.insert("req-1".to_string(), pending(1), t0);
        assert!(cache.take(&key(), "req-1", t0).is_some());

        // The sweep skips req-1's leftover tuple without touching the live entry.
        let later = t0 + TTL + Duration::from_secs(1);
        cache.insert("req-2".to_string(), pending(2), later);
        assert_eq!(cache.len(), 1);
        assert!(cache.take(&key(), "req-2", later).is_some());
    }

    #[test]
    fn cap_evicts_oldest_first() {
        let cache = cache(); // CAP = 2
        let t = Instant::now();
        cache.insert("a".to_string(), pending(1), t);
        cache.insert("b".to_string(), pending(2), t + Duration::from_millis(1));
        cache.insert("c".to_string(), pending(3), t + Duration::from_millis(2));

        let now = t + Duration::from_millis(3);
        assert_eq!(cache.len(), 2);
        assert!(cache.take(&key(), "a", now).is_none());
        assert!(cache.take(&key(), "b", now).is_some());
        assert!(cache.take(&key(), "c", now).is_some());
    }

    #[test]
    fn cap_eviction_skips_stale_tuples_of_reinserted_ids() {
        let cache = cache(); // CAP = 2
        let t = Instant::now();
        cache.insert("x".to_string(), pending(1), t);
        // Re-inserting "x" leaves a stale ("x", t) tuple behind the live one.
        cache.insert("x".to_string(), pending(2), t + Duration::from_millis(1));
        cache.insert("y".to_string(), pending(3), t + Duration::from_millis(2));
        // Over the cap: "x" is evicted via its live tuple, not the stale one.
        cache.insert("z".to_string(), pending(4), t + Duration::from_millis(3));

        let now = t + Duration::from_millis(4);
        assert_eq!(cache.len(), 2);
        assert!(cache.take(&key(), "x", now).is_none());
        assert!(cache.take(&key(), "y", now).is_some());
        assert!(cache.take(&key(), "z", now).is_some());
    }

    #[test]
    fn cap_eviction_identifies_entries_by_generation() {
        // All ops share one instant, so a re-inserted id and its stale tuple
        // carry the same timestamp; generation distinguishes them.
        let cache = cache(); // CAP = 2
        let t = Instant::now();
        cache.insert("a".to_string(), pending(1), t);
        cache.insert("b".to_string(), pending(2), t);
        assert!(cache.take(&key(), "a", t).is_some());
        cache.insert("a".to_string(), pending(3), t);
        cache.insert("c".to_string(), pending(4), t);

        // The oldest live entry "b" is evicted; refreshed "a" and "c" remain.
        assert_eq!(cache.len(), 2);
        assert!(cache.take(&key(), "b", t).is_none());
        assert!(cache.take(&key(), "a", t).is_some());
        assert!(cache.take(&key(), "c", t).is_some());
    }

    #[test]
    fn insert_drains_stale_order_tuples() {
        let cache = SelectionCache::new(TTL, 1024);
        // Every insert is claimed immediately, so `entries` stays near empty.
        // The order queue must not accumulate a tuple per request.
        let t = Instant::now();
        for i in 0..50 {
            let id = format!("req-{i}");
            cache.insert(id.clone(), pending(i as u64), t);
            assert!(cache.take(&key(), &id, t).is_some());
        }
        cache.insert("last".to_string(), pending(99), t);
        assert_eq!(cache.len(), 1);
        // Bounded by live entries
        assert!(
            cache.order_len() <= 2,
            "order grew to {}",
            cache.order_len()
        );
    }
}
