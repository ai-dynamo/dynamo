// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! One indexer per routing partition, created on demand with the registry's policy and
//! metrics, plus the readiness signal consumers wait on before trusting lookups.
//!
//! Two creation paths exist because two callers have different contracts. A worker
//! registration *claims* an indexer under the lifecycle lock, so a concurrent removal of
//! an empty indexer cannot race the claim, and a block-size mismatch is the caller's
//! error. A recovery dump or an embedding host *gets or creates* one lock-free, and a
//! mismatch is only logged, because the dump already describes an indexer that exists
//! elsewhere.

use std::sync::Arc;

use dashmap::DashMap;
use dashmap::mapref::one::Ref;
use parking_lot::Mutex;
use tokio::sync::watch;

use crate::backend::{Indexer, IndexerPolicy, create_indexer_with_policy};
use crate::identity::RoutingPartitionId;
use crate::metrics::KvIndexerMetrics;

/// A partition's indexer and the block size it was created for.
pub struct IndexerEntry {
    pub indexer: Indexer,
    pub block_size: u32,
}

/// Why a claim could not hand out a partition's indexer.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum IndexClaimError {
    #[error(
        "block_size mismatch for model={model_name} routing_group={routing_group}: existing={existing}, requested={requested}"
    )]
    BlockSizeMismatch {
        model_name: String,
        routing_group: String,
        existing: u32,
        requested: u32,
    },
}

/// Proof that the caller holds the indexer lifecycle lock; see [`IndexRegistry::lock_lifecycle`].
pub struct LifecycleGuard<'a> {
    _guard: tokio::sync::MutexGuard<'a, ()>,
}

pub struct IndexRegistry {
    indexers: DashMap<RoutingPartitionId, IndexerEntry>,
    /// Serializes indexer claims against empty-indexer removal.
    lifecycle: tokio::sync::Mutex<()>,
    num_threads: usize,
    metrics: Arc<KvIndexerMetrics>,
    policy: Mutex<IndexerPolicy>,
    ready_tx: watch::Sender<bool>,
    ready_rx: watch::Receiver<bool>,
    retain_empty_indexers: bool,
}

impl IndexRegistry {
    pub fn new(num_threads: usize, metrics: Arc<KvIndexerMetrics>) -> Self {
        let (ready_tx, ready_rx) = watch::channel(false);
        Self {
            indexers: DashMap::new(),
            lifecycle: tokio::sync::Mutex::new(()),
            num_threads,
            metrics,
            policy: Mutex::new(IndexerPolicy::event_driven()),
            ready_tx,
            ready_rx,
            retain_empty_indexers: false,
        }
    }

    /// Keep an indexer when its last worker leaves; schedulers that hold the indexer
    /// across worker updates want this.
    pub fn with_retained_indexers(mut self) -> Self {
        self.retain_empty_indexers = true;
        self
    }

    /// The policy new indexers are created with; existing indexers keep theirs.
    pub fn set_indexer_policy(&self, policy: IndexerPolicy) {
        *self.policy.lock() = policy;
    }

    pub fn signal_ready(&self) {
        let _ = self.ready_tx.send(true);
    }

    pub fn ready_rx(&self) -> watch::Receiver<bool> {
        self.ready_rx.clone()
    }

    /// Number of partitions with an indexer.
    pub fn len(&self) -> usize {
        self.indexers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.indexers.is_empty()
    }

    pub fn contains(&self, key: &RoutingPartitionId) -> bool {
        self.indexers.contains_key(key)
    }

    /// Take the lifecycle lock. Hold it across a [`claim`](Self::claim) and whatever
    /// bookkeeping must not interleave with an empty-indexer removal.
    pub async fn lock_lifecycle(&self) -> LifecycleGuard<'_> {
        LifecycleGuard {
            _guard: self.lifecycle.lock().await,
        }
    }

    /// Create the partition's indexer if needed and hand it out, under the lifecycle
    /// lock the caller holds. A different block size than the existing indexer's is an
    /// error; nothing is created or changed in that case.
    pub fn claim(
        &self,
        _lifecycle: &LifecycleGuard<'_>,
        key: &RoutingPartitionId,
        block_size: u32,
    ) -> Result<Indexer, IndexClaimError> {
        let policy = self.policy.lock().clone();
        let entry = self.indexers.entry(key.clone()).or_insert_with(|| {
            tracing::info!(
                model_name = %key.model_name,
                routing_group = %key.routing_group,
                block_size,
                "Creating new indexer"
            );
            IndexerEntry {
                indexer: create_indexer_with_policy(
                    block_size,
                    self.num_threads,
                    self.metrics.clone(),
                    &policy,
                ),
                block_size,
            }
        });
        if entry.block_size != block_size {
            return Err(IndexClaimError::BlockSizeMismatch {
                model_name: key.model_name.clone(),
                routing_group: key.routing_group.clone(),
                existing: entry.block_size,
                requested: block_size,
            });
        }
        Ok(entry.indexer.clone())
    }

    /// Remove the partition's indexer when nothing names it any more, under the
    /// lifecycle lock; `in_use` is evaluated while the lock is held. A registry built
    /// with retained indexers never removes.
    pub async fn release_if_unused(&self, key: &RoutingPartitionId, in_use: impl FnOnce() -> bool) {
        if self.retain_empty_indexers {
            return;
        }
        let _lifecycle = self.lifecycle.lock().await;
        if in_use() {
            return;
        }
        self.indexers.remove(key);
    }

    pub fn get_indexer(
        &self,
        key: &RoutingPartitionId,
    ) -> Option<Ref<'_, RoutingPartitionId, IndexerEntry>> {
        self.indexers.get(key)
    }

    /// The partition's indexer handle, if it has one. Clones the handle so no shard
    /// guard outlives the call.
    pub fn indexer(&self, key: &RoutingPartitionId) -> Option<Indexer> {
        self.indexers.get(key).map(|entry| entry.indexer.clone())
    }

    pub fn block_size(&self, key: &RoutingPartitionId) -> Option<u32> {
        self.indexers.get(key).map(|entry| entry.block_size)
    }

    /// The lock-free creation path: a recovery dump or an embedding host asks for the
    /// partition's indexer and gets one created with the registry's policy if none
    /// exists. A block-size mismatch is logged, not rejected.
    pub fn get_or_create_indexer(&self, key: RoutingPartitionId, block_size: u32) -> Indexer {
        let policy = self.policy.lock().clone();
        let entry = self.indexers.entry(key.clone()).or_insert_with(|| {
            tracing::info!(
                model_name = %key.model_name,
                routing_group = %key.routing_group,
                block_size,
                "Creating indexer from recovery dump"
            );
            IndexerEntry {
                indexer: create_indexer_with_policy(
                    block_size,
                    self.num_threads,
                    self.metrics.clone(),
                    &policy,
                ),
                block_size,
            }
        });
        if entry.block_size != block_size {
            tracing::warn!(
                model_name = %key.model_name,
                routing_group = %key.routing_group,
                existing_block_size = entry.block_size,
                requested_block_size = block_size,
                "Block size mismatch for existing indexer"
            );
        }
        entry.indexer.clone()
    }

    pub fn all_indexers_with_block_size(&self) -> Vec<(RoutingPartitionId, Indexer, u32)> {
        self.indexers
            .iter()
            .map(|entry| {
                (
                    entry.key().clone(),
                    entry.value().indexer.clone(),
                    entry.value().block_size,
                )
            })
            .collect()
    }

    /// Whether the partition's map shard can be written right now; tests use it to
    /// prove removal does not hold the shard across an await.
    #[cfg(any(test, feature = "testing"))]
    pub fn shard_is_writable(&self, key: &RoutingPartitionId) -> bool {
        !matches!(
            self.indexers.try_get_mut(key),
            dashmap::try_result::TryResult::Locked
        )
    }
}
