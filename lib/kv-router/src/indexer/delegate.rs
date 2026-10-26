// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{hash::Hash, sync::Arc};

use parking_lot::Mutex;
use rustc_hash::{FxHashMap, FxHashSet};

use crate::protocols::{ExternalSequenceBlockHash, WorkerWithDpRank};

/// Observes the first owner and last owner of a sequence hash within one indexer.
///
/// Supply the delegate during construction. Indexers cannot replace it after construction.
/// Duplicate events do not produce notifications. Reinsertion after removal produces another create.
/// Worker clears, rank resets, and explicit evictions all release ownership.
/// Destruction does not produce removal notifications.
///
/// Callbacks run synchronously on mutation threads. Calls for one hash are serialized.
/// Calls for different hashes can overlap. Callbacks must return promptly and must not panic,
/// mutate the indexer, or wait for another indexer operation.
/// Notifications describe ownership transitions, not an atomic snapshot of concurrent query results.
///
/// Primary radix and positional backends use engine sequence hashes.
/// The exact cuckoo producer uses its canonical sequence hash type as `H`.
pub trait KvIndexerDelegate<H = ExternalSequenceBlockHash>: Send + Sync + 'static {
    /// Called when a hash gains its first owner.
    fn on_create(&self, hash: H);

    /// Called when a hash loses its last owner.
    fn on_remove(&self, hash: H);
}

type Owners<O> = FxHashMap<ExternalSequenceBlockHash, FxHashSet<O>>;

struct DelegateState<O> {
    delegate: Arc<dyn KvIndexerDelegate>,
    shards: [Mutex<Owners<O>>; 64],
}

/// Shared across mutation threads, allocated only when a delegate is configured.
#[derive(Clone)]
pub(crate) struct HashLifecycle<O = WorkerWithDpRank>(Option<Arc<DelegateState<O>>>);

impl<O> Default for HashLifecycle<O> {
    fn default() -> Self {
        Self(None)
    }
}

impl<O: Eq + Hash> HashLifecycle<O> {
    pub(crate) fn is_enabled(&self) -> bool {
        self.0.is_some()
    }

    pub(crate) fn new(delegate: Arc<dyn KvIndexerDelegate>) -> Self {
        Self(Some(Arc::new(DelegateState {
            delegate,
            shards: std::array::from_fn(|_| Mutex::new(FxHashMap::default())),
        })))
    }

    pub(crate) fn insert(&self, worker: O, hash: ExternalSequenceBlockHash) {
        let Some(state) = &self.0 else { return };
        let mut shard = state.shards[hash.0 as usize % state.shards.len()].lock();
        let owners = shard.entry(hash).or_default();
        if owners.insert(worker) && owners.len() == 1 {
            state.delegate.on_create(hash);
        }
    }

    pub(crate) fn remove(&self, worker: O, hash: ExternalSequenceBlockHash) {
        let Some(state) = &self.0 else { return };
        let mut shard = state.shards[hash.0 as usize % state.shards.len()].lock();
        if let Some(owners) = shard.get_mut(&hash)
            && owners.remove(&worker)
            && owners.is_empty()
        {
            shard.remove(&hash);
            state.delegate.on_remove(hash);
        }
    }
}

impl<H> std::fmt::Debug for dyn KvIndexerDelegate<H> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("KvIndexerDelegate")
    }
}
