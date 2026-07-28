// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! In-flight selection cache: holds the booking inputs a `/select` computed so
//! a later `/reservations` replays them by `selection_id` without re-sending
//! the prompt. Unclaimed entries are swept and capped on every insert.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_tokens::SequenceHash;
use parking_lot::Mutex;

use crate::identity::RoutingPartitionId;
use crate::protocols::WorkerWithDpRank;

/// How long a pending selection lives before it is evicted.
const SELECTION_CACHE_TTL: Duration = Duration::from_secs(120);

/// Max resident pending selections; oldest-first eviction.
const SELECTION_CACHE_MAX_ENTRIES: usize = 4096;

/// Approximate resident-byte budget; the entry cap alone does not bound memory
/// when prompts are large.
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

/// Booking inputs captured by `select`, replayed by a later `create_reservation`.
pub(super) struct PendingSelection {
    pub key: RoutingPartitionId,
    pub worker: WorkerWithDpRank,
    pub sequence_hashes: Vec<SequenceHash>,
    pub isl_tokens: usize,
    pub effective_prefill_tokens: usize,
    pub expected_output_tokens: Option<u32>,
    pub track_prefill_tokens: bool,
    pub lora_name: Option<String>,
}

struct Entry {
    selection: Arc<PendingSelection>,
    inserted_at: Instant,
    generation: u64,
    bytes: usize,
}

/// Scoped by `(RoutingPartitionId, selection_id)`.
type CacheKey = (RoutingPartitionId, String);

/// Approximate resident bytes: sequence hashes plus id and scope strings.
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

/// Maps `(RoutingPartitionId, selection_id)` to the booking inputs computed during `select`.
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

    /// Insert (or replace) the selection, sweeping expired entries and enforcing the cap.
    pub(super) fn insert(&self, selection_id: String, selection: PendingSelection, now: Instant) {
        self.insert_at(selection_id, selection, now, now);
    }

    /// `inserted_at` anchors the entry's TTL; `now` drives the sweep.
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
        // A replaced entry drops its old order key and bytes.
        if let Some(old) = state.entries.insert(
            cache_key,
            Entry {
                selection: Arc::new(selection),
                inserted_at,
                generation,
                bytes,
            },
        ) {
            state.order.remove(&(old.inserted_at, old.generation));
            state.total_bytes = state.total_bytes.saturating_sub(old.bytes);
        }
        // Evict oldest-first over the cap or byte budget (an over-budget entry evicts itself).
        while state.entries.len() > self.max_entries || state.total_bytes > self.max_bytes {
            let Some((_, key)) = state.order.pop_first() else {
                break;
            };
            if let Some(evicted) = state.entries.remove(&key) {
                state.total_bytes = state.total_bytes.saturating_sub(evicted.bytes);
            }
        }
    }

    /// Return a shared `Arc` handle to the selection and its `generation` without
    /// removing it (past-TTL is treated as gone); the caller `remove`s it only
    /// once the booking lands.
    pub(super) fn peek(
        &self,
        key: &RoutingPartitionId,
        selection_id: &str,
        now: Instant,
    ) -> Option<(Arc<PendingSelection>, u64)> {
        let cache_key = (key.clone(), selection_id.to_string());
        let state = self.state.lock();
        let entry = state.entries.get(&cache_key)?;
        if now.duration_since(entry.inserted_at) > self.ttl {
            return None;
        }
        Some((Arc::clone(&entry.selection), entry.generation))
    }

    /// Consume the entry after a booking lands, only if its `generation` still
    /// matches the peek; a newer `select` for the id survives.
    pub(super) fn remove(&self, key: &RoutingPartitionId, selection_id: &str, generation: u64) {
        let cache_key = (key.clone(), selection_id.to_string());
        let mut state = self.state.lock();
        if state
            .entries
            .get(&cache_key)
            .is_some_and(|entry| entry.generation == generation)
        {
            self.remove_locked(&mut state, &cache_key);
        }
    }

    /// Drop any cached selection for the id (an explicit booking supersedes it).
    pub(super) fn discard(&self, key: &RoutingPartitionId, selection_id: &str) {
        let cache_key = (key.clone(), selection_id.to_string());
        let mut state = self.state.lock();
        self.remove_locked(&mut state, &cache_key);
    }

    fn remove_locked(&self, state: &mut State, cache_key: &CacheKey) {
        if let Some(entry) = state.entries.remove(cache_key) {
            state.order.remove(&(entry.inserted_at, entry.generation));
            state.total_bytes = state.total_bytes.saturating_sub(entry.bytes);
        }
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

    fn key() -> RoutingPartitionId {
        RoutingPartitionId::new("model", "default")
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

    /// Peek an entry and consume it (as a successful booking would).
    fn peek_and_remove(
        cache: &SelectionCache,
        id: &str,
        now: Instant,
    ) -> Option<Arc<PendingSelection>> {
        let (selection, generation) = cache.peek(&key(), id, now)?;
        cache.remove(&key(), id, generation);
        Some(selection)
    }

    #[test]
    fn peek_then_remove_consumes_once() {
        let cache = cache();
        let now = Instant::now();
        cache.insert("req-1".to_string(), pending(1), now);

        let (peeked, generation) = cache.peek(&key(), "req-1", now).expect("entry present");
        assert_eq!(peeked.worker.worker_id, 1);
        assert_eq!(peeked.isl_tokens, 12);
        // A peek leaves the entry; only remove consumes it.
        assert!(cache.peek(&key(), "req-1", now).is_some());
        cache.remove(&key(), "req-1", generation);
        assert!(cache.peek(&key(), "req-1", now).is_none());
    }

    #[test]
    fn peek_refuses_expired_entry() {
        let cache = cache();
        let inserted = Instant::now();
        cache.insert("req-1".to_string(), pending(1), inserted);

        let later = inserted + TTL + Duration::from_secs(1);
        assert!(cache.peek(&key(), "req-1", later).is_none());
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
        assert!(cache.peek(&key(), "old", later).is_none());
        assert!(cache.peek(&key(), "new", later).is_some());
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
        assert!(cache.peek(&key(), "a", now).is_none());
        assert!(cache.peek(&key(), "b", now).is_some());
        assert!(cache.peek(&key(), "c", now).is_some());
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
        assert!(cache.peek(&key(), "a", now).is_none());
        assert!(cache.peek(&key(), "b", now).is_some());
        assert!(cache.peek(&key(), "c", now).is_some());
    }

    #[test]
    fn remove_yields_to_newer_selection() {
        let cache = cache();
        let t = Instant::now();
        cache.insert("req-1".to_string(), pending(1), t);
        let (_old, stale_generation) = cache.peek(&key(), "req-1", t).expect("entry present");
        // A newer select replaces the id while the first replay is in flight.
        cache.insert(
            "req-1".to_string(),
            pending(2),
            t + Duration::from_millis(1),
        );
        // A remove for the stale generation must not drop the newer selection.
        cache.remove(&key(), "req-1", stale_generation);

        let (survivor, _) = cache
            .peek(&key(), "req-1", t + Duration::from_millis(2))
            .expect("entry present");
        assert_eq!(survivor.worker.worker_id, 2);
    }

    #[test]
    fn order_index_bounded_by_live_entries_under_pinned_front() {
        let cache = cache_with(1024, usize::MAX);
        let t = Instant::now();
        // One live entry pinned at the front, never consumed.
        cache.insert("pinned".to_string(), pending(0), t);
        // Heavy insert/consume traffic behind it must not grow the index.
        for i in 1..=50u64 {
            let id = format!("req-{i}");
            let at = t + Duration::from_millis(i);
            cache.insert(id.clone(), pending(i), at);
            assert!(peek_and_remove(&cache, &id, at).is_some());
        }
        assert_eq!(cache.len(), 1);
        assert_eq!(cache.order_len(), 1);
    }
}
