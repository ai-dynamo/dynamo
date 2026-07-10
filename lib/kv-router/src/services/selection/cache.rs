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

use std::collections::{BTreeMap, HashMap};
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

/// Upper bound on resident bytes (approximate: sequence hashes plus id
/// strings). Bounds memory when prompts are large, since request bodies can
/// reach several MiB and the entry cap alone does not.
const SELECTION_CACHE_MAX_BYTES: usize = 256 * 1024 * 1024;

/// Runtime-tunable bounds for the pending-selection cache.
#[derive(Debug, Clone)]
pub struct SelectionCacheConfig {
    /// Lifetime of an unclaimed pending selection.
    pub ttl: Duration,
    /// Maximum number of resident pending selections.
    pub max_entries: usize,
    /// Approximate byte budget across resident pending selections.
    pub max_bytes: usize,
}

impl Default for SelectionCacheConfig {
    fn default() -> Self {
        Self {
            ttl: SELECTION_CACHE_TTL,
            max_entries: SELECTION_CACHE_MAX_ENTRIES,
            max_bytes: SELECTION_CACHE_MAX_BYTES,
        }
    }
}

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
    bytes: usize,
}

/// Pending selections are scoped by `(SelectionKey, selection_id)`, where
/// `SelectionKey` is the (model, routing_group) scope.
type CacheKey = (SelectionKey, String);

/// Approximate resident bytes of an entry: the variable-length sequence hashes
/// plus the id and scope strings.
fn entry_bytes(key: &CacheKey, selection: &PendingSelection) -> usize {
    selection.sequence_hashes.len() * std::mem::size_of::<SequenceHash>()
        + key.1.len()
        + key.0.model_name.len()
        + key.0.routing_group.len()
        + selection.lora_name.as_deref().map_or(0, str::len)
}

struct State {
    entries: HashMap<CacheKey, Entry>,
    /// Live entries keyed by `(inserted_at, generation)` for oldest-first
    /// eviction; `generation` breaks `Instant` ties.
    order: BTreeMap<(Instant, u64), CacheKey>,
    next_generation: u64,
    total_bytes: usize,
}

/// Maps `(SelectionKey, selection_id)` -> the booking inputs computed during `select`.
pub(super) struct SelectionCache {
    ttl: Duration,
    max_entries: usize,
    max_bytes: usize,
    state: Mutex<State>,
}

impl SelectionCache {
    pub(super) fn new(config: &SelectionCacheConfig) -> Self {
        Self {
            ttl: config.ttl,
            max_entries: config.max_entries,
            max_bytes: config.max_bytes,
            state: Mutex::new(State {
                entries: HashMap::new(),
                order: BTreeMap::new(),
                next_generation: 0,
                total_bytes: 0,
            }),
        }
    }

    /// Record (or replace) the pending selection under `(selection.key, selection_id)`,
    /// sweeping expired entries and evicting oldest-first if over the cap.
    pub(super) fn insert(&self, selection_id: String, selection: PendingSelection, now: Instant) {
        self.insert_at(selection_id, selection, now, now);
    }

    /// Re-insert a selection taken earlier for a failed booking, keeping its
    /// original TTL anchor so retries cannot extend its life. A newer select
    /// for the same id wins, so this is a no-op when the id is already present.
    pub(super) fn reinsert(
        &self,
        selection_id: String,
        selection: PendingSelection,
        inserted_at: Instant,
        now: Instant,
    ) {
        let cache_key = (selection.key.clone(), selection_id);
        let mut state = self.state.lock();
        if state.entries.contains_key(&cache_key) {
            return;
        }
        self.insert_locked(&mut state, cache_key, selection, inserted_at, now);
    }

    /// `inserted_at` anchors the entry's TTL (equal to `now` for a fresh insert,
    /// the original time for a re-insert); `now` drives the sweep.
    fn insert_at(
        &self,
        selection_id: String,
        selection: PendingSelection,
        inserted_at: Instant,
        now: Instant,
    ) {
        let cache_key = (selection.key.clone(), selection_id);
        let mut state = self.state.lock();
        self.insert_locked(&mut state, cache_key, selection, inserted_at, now);
    }

    fn insert_locked(
        &self,
        state: &mut State,
        cache_key: CacheKey,
        selection: PendingSelection,
        inserted_at: Instant,
        now: Instant,
    ) {
        // Sweep expired entries from the front (oldest by anchor).
        while let Some(oldest) = state.order.keys().next().copied() {
            if now.duration_since(oldest.0) <= self.ttl {
                break;
            }
            if let Some(key) = state.order.remove(&oldest)
                && let Some(evicted) = state.entries.remove(&key)
            {
                state.total_bytes = state.total_bytes.saturating_sub(evicted.bytes);
            }
        }
        let bytes = entry_bytes(&cache_key, &selection);
        let generation = state.next_generation;
        state.next_generation += 1;
        state
            .order
            .insert((inserted_at, generation), cache_key.clone());
        state.total_bytes = state.total_bytes.saturating_add(bytes);
        // Replacing a live selection drops its old order entry and byte count.
        if let Some(old) = state.entries.insert(
            cache_key,
            Entry {
                selection,
                inserted_at,
                generation,
                bytes,
            },
        ) {
            state.order.remove(&(old.inserted_at, old.generation));
            state.total_bytes = state.total_bytes.saturating_sub(old.bytes);
        }
        // Evict oldest-first while over the entry cap or byte budget. An entry
        // larger than the whole budget is evicted immediately (not cached).
        while state.entries.len() > self.max_entries || state.total_bytes > self.max_bytes {
            let Some((_, key)) = state.order.pop_first() else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&key) {
                state.total_bytes = state.total_bytes.saturating_sub(evicted.bytes);
            }
        }
    }

    /// Remove and return the pending selection for `(key, selection_id)` with
    /// its original TTL anchor (for a possible [`reinsert`](Self::reinsert)). An
    /// entry older than the TTL is treated as already gone (and dropped).
    pub(super) fn take(
        &self,
        key: &SelectionKey,
        selection_id: &str,
        now: Instant,
    ) -> Option<(PendingSelection, Instant)> {
        let cache_key = (key.clone(), selection_id.to_string());
        let mut state = self.state.lock();
        let entry = state.entries.remove(&cache_key)?;
        state.order.remove(&(entry.inserted_at, entry.generation));
        state.total_bytes = state.total_bytes.saturating_sub(entry.bytes);
        if now.duration_since(entry.inserted_at) > self.ttl {
            return None;
        }
        Some((entry.selection, entry.inserted_at))
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

    fn cache_with(max_entries: usize, max_bytes: usize) -> SelectionCache {
        SelectionCache::new(&SelectionCacheConfig {
            ttl: TTL,
            max_entries,
            max_bytes,
        })
    }

    fn cache() -> SelectionCache {
        cache_with(CAP, usize::MAX)
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

        let (taken, _) = cache.take(&key(), "req-1", now).expect("entry present");
        assert_eq!(taken.worker.worker_id, 1);
        assert_eq!(taken.isl_tokens, 12);
        assert!(cache.take(&key(), "req-1", now).is_none());
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
    fn byte_budget_evicts_oldest() {
        // Budget holds two entries; the entry cap is not the binding limit.
        let per = entry_bytes(&(key(), "a".to_string()), &pending(1));
        let cache = cache_with(1024, 2 * per);
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
    fn reinsert_preserves_original_ttl_anchor() {
        let cache = cache();
        let t0 = Instant::now();
        cache.insert("req-1".to_string(), pending(1), t0);
        let (pending, inserted_at) = cache.take(&key(), "req-1", t0).expect("entry present");
        assert_eq!(inserted_at, t0);

        // The failed-booking path re-inserts later but keeps the original
        // anchor, so a retry loop cannot extend the entry past its TTL.
        let later = t0 + Duration::from_secs(5);
        cache.reinsert("req-1".to_string(), pending, inserted_at, later);

        let past_original_ttl = t0 + TTL + Duration::from_secs(1);
        assert!(cache.take(&key(), "req-1", past_original_ttl).is_none());
    }

    #[test]
    fn reinsert_yields_to_newer_selection() {
        let cache = cache();
        let t = Instant::now();
        cache.insert("req-1".to_string(), pending(1), t);
        let (old, old_at) = cache.take(&key(), "req-1", t).expect("entry present");
        // A newer select repopulates the id while the failed replay is in flight.
        cache.insert(
            "req-1".to_string(),
            pending(2),
            t + Duration::from_millis(1),
        );
        // The failed replay must not clobber the newer selection.
        cache.reinsert(
            "req-1".to_string(),
            old,
            old_at,
            t + Duration::from_millis(2),
        );

        let (survivor, _) = cache
            .take(&key(), "req-1", t + Duration::from_millis(3))
            .expect("entry present");
        assert_eq!(survivor.worker.worker_id, 2);
    }

    #[test]
    fn order_index_bounded_by_live_entries_under_pinned_front() {
        let cache = cache_with(1024, usize::MAX);
        let t = Instant::now();
        // One live entry pinned at the front, never taken.
        cache.insert("pinned".to_string(), pending(0), t);
        // Heavy insert/take traffic behind it must not grow the index.
        for i in 1..=50u64 {
            let id = format!("req-{i}");
            let at = t + Duration::from_millis(i);
            cache.insert(id.clone(), pending(i), at);
            assert!(cache.take(&key(), &id, at).is_some());
        }
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.order_len(), 1);
    }

    #[test]
    fn cap_evicts_oldest_by_anchor_including_reinserted() {
        let cache = cache(); // CAP = 2
        let t = Instant::now();
        // "old" is selected first (oldest anchor) and taken.
        cache.insert("old".to_string(), pending(1), t);
        let (old, old_at) = cache.take(&key(), "old", t).expect("old present");
        // A newer selection arrives, then "old" is reinserted with its original anchor.
        cache.insert("mid".to_string(), pending(2), t + Duration::from_millis(5));
        cache.reinsert("old".to_string(), old, old_at, t + Duration::from_millis(6));
        // A third selection pushes over the cap.
        cache.insert("new".to_string(), pending(3), t + Duration::from_millis(7));

        // "old" has the oldest anchor and is evicted first, though it was
        // reinserted most recently.
        let now = t + Duration::from_millis(8);
        assert_eq!(cache.len(), 2);
        assert!(cache.take(&key(), "old", now).is_none());
        assert!(cache.take(&key(), "mid", now).is_some());
        assert!(cache.take(&key(), "new", now).is_some());
    }
}
