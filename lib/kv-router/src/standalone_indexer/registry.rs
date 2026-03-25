// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dashmap::mapref::one::Ref;
use parking_lot::Mutex;
use serde::Serialize;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::protocols::WorkerId;

use super::indexer::{Indexer, create_indexer};
use super::listener::spawn_zmq_listener;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct IndexerKey {
    pub model_name: String,
    pub tenant_id: String,
}

pub struct IndexerEntry {
    pub indexer: Indexer,
    pub block_size: u32,
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ListenerStatus {
    Pending,
    Active,
    Paused,
    Failed,
}

impl ListenerStatus {
    pub const ALL: [Self; 4] = [Self::Pending, Self::Active, Self::Paused, Self::Failed];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Paused => "paused",
            Self::Failed => "failed",
        }
    }

    pub fn metric_index(self) -> usize {
        match self {
            Self::Pending => 0,
            Self::Active => 1,
            Self::Paused => 2,
            Self::Failed => 3,
        }
    }

    pub fn aggregate(statuses: impl IntoIterator<Item = Self>) -> Self {
        let mut saw_pending = false;
        let mut saw_active = false;

        for status in statuses {
            match status {
                Self::Failed => return Self::Failed,
                Self::Pending => saw_pending = true,
                Self::Active => saw_active = true,
                Self::Paused => {}
            }
        }

        if saw_pending {
            Self::Pending
        } else if saw_active {
            Self::Active
        } else {
            Self::Paused
        }
    }
}

impl fmt::Display for ListenerStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WorkerSource {
    Zmq,
    Discovery,
}

#[derive(Debug, Clone, Serialize)]
pub struct ListenerInfo {
    endpoint: String,
    status: ListenerStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    last_error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WorkerInfo {
    instance_id: WorkerId,
    source: WorkerSource,
    status: ListenerStatus,
    endpoints: HashMap<u32, String>,
    listeners: HashMap<u32, ListenerInfo>,
}

#[derive(Debug, thiserror::Error)]
pub enum ListenerControlError {
    #[error("instance {instance_id} not found")]
    WorkerNotFound { instance_id: WorkerId },

    #[error("instance {instance_id} dp_rank {dp_rank} not found")]
    ListenerNotFound { instance_id: WorkerId, dp_rank: u32 },

    #[error("instance {instance_id} is discovery-managed; no ZMQ listener to control")]
    DiscoveryManaged { instance_id: WorkerId },

    #[error("instance {instance_id} dp_rank {dp_rank} cannot be paused from status {status}")]
    InvalidPauseState {
        instance_id: WorkerId,
        dp_rank: u32,
        status: ListenerStatus,
    },

    #[error("instance {instance_id} dp_rank {dp_rank} cannot be resumed from status {status}")]
    InvalidResumeState {
        instance_id: WorkerId,
        dp_rank: u32,
        status: ListenerStatus,
    },
}

struct ListenerRuntime {
    status: ListenerStatus,
    last_error: Option<String>,
    cancel_token: Option<CancellationToken>,
    generation: u64,
}

pub struct ListenerRecord {
    endpoint: String,
    replay_endpoint: Option<String>,
    block_size: u32,
    indexer: Indexer,
    watermark: Arc<AtomicU64>,
    runtime: Mutex<ListenerRuntime>,
}

impl ListenerRecord {
    fn new(
        endpoint: String,
        replay_endpoint: Option<String>,
        block_size: u32,
        indexer: Indexer,
        watermark: Arc<AtomicU64>,
    ) -> Self {
        Self {
            endpoint,
            replay_endpoint,
            block_size,
            indexer,
            watermark,
            runtime: Mutex::new(ListenerRuntime {
                status: ListenerStatus::Pending,
                last_error: None,
                cancel_token: None,
                generation: 0,
            }),
        }
    }

    pub(super) fn endpoint(&self) -> &str {
        &self.endpoint
    }

    pub(super) fn replay_endpoint(&self) -> Option<&str> {
        self.replay_endpoint.as_deref()
    }

    pub(super) fn block_size(&self) -> u32 {
        self.block_size
    }

    pub(super) fn indexer(&self) -> Indexer {
        self.indexer.clone()
    }

    pub(super) fn watermark(&self) -> Arc<AtomicU64> {
        self.watermark.clone()
    }

    pub(super) fn start_pending(&self) -> (u64, CancellationToken) {
        let mut runtime = self.runtime.lock();
        runtime.generation += 1;
        let cancel_token = CancellationToken::new();
        runtime.status = ListenerStatus::Pending;
        runtime.last_error = None;
        runtime.cancel_token = Some(cancel_token.clone());
        (runtime.generation, cancel_token)
    }

    pub(super) fn pause(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<CancellationToken, ListenerControlError> {
        let mut runtime = self.runtime.lock();
        match runtime.status {
            ListenerStatus::Pending | ListenerStatus::Active => {
                let cancel_token =
                    runtime
                        .cancel_token
                        .take()
                        .ok_or(ListenerControlError::InvalidPauseState {
                            instance_id,
                            dp_rank,
                            status: runtime.status,
                        })?;
                runtime.status = ListenerStatus::Paused;
                runtime.last_error = None;
                Ok(cancel_token)
            }
            status => Err(ListenerControlError::InvalidPauseState {
                instance_id,
                dp_rank,
                status,
            }),
        }
    }

    pub(super) fn resume(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(u64, CancellationToken), ListenerControlError> {
        let mut runtime = self.runtime.lock();
        match runtime.status {
            ListenerStatus::Paused | ListenerStatus::Failed => {
                runtime.generation += 1;
                let cancel_token = CancellationToken::new();
                runtime.status = ListenerStatus::Pending;
                runtime.last_error = None;
                runtime.cancel_token = Some(cancel_token.clone());
                Ok((runtime.generation, cancel_token))
            }
            status => Err(ListenerControlError::InvalidResumeState {
                instance_id,
                dp_rank,
                status,
            }),
        }
    }

    pub(super) fn is_current_attempt(&self, generation: u64) -> bool {
        let runtime = self.runtime.lock();
        runtime.generation == generation && runtime.cancel_token.is_some()
    }

    pub(super) fn try_mark_active(&self, generation: u64) -> bool {
        let mut runtime = self.runtime.lock();
        if runtime.generation != generation || runtime.cancel_token.is_none() {
            return false;
        }
        runtime.status = ListenerStatus::Active;
        runtime.last_error = None;
        true
    }

    pub(super) fn try_mark_failed(&self, generation: u64, error: impl Into<String>) {
        let mut runtime = self.runtime.lock();
        if runtime.generation != generation || runtime.cancel_token.is_none() {
            return;
        }
        runtime.status = ListenerStatus::Failed;
        runtime.last_error = Some(error.into());
        runtime.cancel_token = None;
    }

    fn take_cancel(&self) -> Option<CancellationToken> {
        self.runtime.lock().cancel_token.take()
    }

    fn snapshot(&self) -> ListenerInfo {
        let runtime = self.runtime.lock();
        ListenerInfo {
            endpoint: self.endpoint.clone(),
            status: runtime.status,
            last_error: runtime.last_error.clone(),
        }
    }

    #[allow(dead_code)]
    fn status(&self) -> ListenerStatus {
        self.runtime.lock().status
    }
}

pub struct WorkerEntry {
    key: IndexerKey,
    listeners: HashMap<u32, Arc<ListenerRecord>>,
}

pub struct WorkerRegistry {
    workers: DashMap<WorkerId, WorkerEntry>,
    indexers: DashMap<IndexerKey, IndexerEntry>,
    peers: DashMap<String, ()>,
    watermarks: DashMap<(WorkerId, u32), Arc<AtomicU64>>,
    #[cfg(feature = "indexer-runtime")]
    discovered_workers: DashMap<WorkerId, IndexerKey>,
    num_threads: usize,
    ready_tx: watch::Sender<bool>,
    ready_rx: watch::Receiver<bool>,
}

impl WorkerRegistry {
    pub fn new(num_threads: usize) -> Self {
        let (ready_tx, ready_rx) = watch::channel(false);
        Self {
            workers: DashMap::new(),
            indexers: DashMap::new(),
            peers: DashMap::new(),
            watermarks: DashMap::new(),
            #[cfg(feature = "indexer-runtime")]
            discovered_workers: DashMap::new(),
            num_threads,
            ready_tx,
            ready_rx,
        }
    }

    pub fn signal_ready(&self) {
        let _ = self.ready_tx.send(true);
    }

    pub fn ready_rx(&self) -> watch::Receiver<bool> {
        self.ready_rx.clone()
    }

    pub fn register_peer(&self, url: String) {
        self.peers.entry(url).or_insert(());
    }

    pub fn deregister_peer(&self, url: &str) -> bool {
        self.peers.remove(url).is_some()
    }

    pub fn list_peers(&self) -> Vec<String> {
        self.peers.iter().map(|entry| entry.key().clone()).collect()
    }

    #[cfg(feature = "metrics")]
    pub fn refresh_metrics(&self) {
        let models = self.indexers.len();
        let workers = self.workers.len() + {
            #[cfg(feature = "indexer-runtime")]
            {
                self.discovered_workers.len()
            }
            #[cfg(not(feature = "indexer-runtime"))]
            {
                0
            }
        };

        let mut listener_counts = [0_i64; 4];
        for entry in self.workers.iter() {
            for record in entry.value().listeners.values() {
                listener_counts[record.status().metric_index()] += 1;
            }
        }

        super::metrics::set_worker_state(models, workers, listener_counts);
    }

    #[expect(clippy::too_many_arguments)]
    pub async fn register(
        &self,
        instance_id: WorkerId,
        endpoint: String,
        dp_rank: u32,
        model_name: String,
        tenant_id: String,
        block_size: u32,
        replay_endpoint: Option<String>,
    ) -> Result<()> {
        #[cfg(feature = "indexer-runtime")]
        if self.discovered_workers.contains_key(&instance_id) {
            bail!(
                "instance {instance_id} is already registered via discovery; \
                 use the Dynamo runtime to manage it"
            );
        }

        let key = IndexerKey {
            model_name,
            tenant_id,
        };

        if let Some(entry) = self.workers.get(&instance_id) {
            if entry.key != key {
                bail!(
                    "instance {instance_id} is already registered for model={} tenant={}",
                    entry.key.model_name,
                    entry.key.tenant_id
                );
            }

            if entry.listeners.contains_key(&dp_rank) {
                bail!("instance {instance_id} dp_rank {dp_rank} already registered");
            }
        }

        let indexer_entry = self.indexers.entry(key.clone()).or_insert_with(|| {
            tracing::info!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                block_size,
                "Creating new indexer"
            );
            IndexerEntry {
                indexer: create_indexer(block_size, self.num_threads),
                block_size,
            }
        });

        if indexer_entry.block_size != block_size {
            bail!(
                "block_size mismatch for model={} tenant={}: existing={}, requested={}",
                key.model_name,
                key.tenant_id,
                indexer_entry.block_size,
                block_size
            );
        }

        let indexer = indexer_entry.indexer.clone();
        let bs = indexer_entry.block_size;
        drop(indexer_entry);

        let watermark = self
            .watermarks
            .entry((instance_id, dp_rank))
            .or_insert_with(|| Arc::new(AtomicU64::new(u64::MAX)))
            .clone();

        let record = Arc::new(ListenerRecord::new(
            endpoint,
            replay_endpoint,
            bs,
            indexer,
            watermark,
        ));
        let attempt = record.start_pending();

        {
            let mut entry = self
                .workers
                .entry(instance_id)
                .or_insert_with(|| WorkerEntry {
                    key: key.clone(),
                    listeners: HashMap::new(),
                });
            entry.listeners.insert(dp_rank, record.clone());
        }

        self.spawn_listener(instance_id, dp_rank, attempt, record);
        Ok(())
    }

    pub async fn deregister(
        &self,
        instance_id: WorkerId,
        model_name: &str,
        tenant_id: &str,
    ) -> Result<()> {
        let key = IndexerKey {
            model_name: model_name.to_string(),
            tenant_id: tenant_id.to_string(),
        };

        if let Some(entry) = self.workers.get(&instance_id) {
            if entry.key != key {
                bail!(
                    "instance {instance_id} is registered for model={} tenant={}",
                    entry.key.model_name,
                    entry.key.tenant_id
                );
            }
        } else {
            #[cfg(feature = "indexer-runtime")]
            if let Some(discovered_key) = self.discovered_workers.get(&instance_id) {
                if discovered_key.value() != &key {
                    bail!(
                        "instance {instance_id} is registered for model={} tenant={}",
                        discovered_key.value().model_name,
                        discovered_key.value().tenant_id
                    );
                }
            } else {
                bail!("instance {instance_id} not found");
            }

            #[cfg(not(feature = "indexer-runtime"))]
            bail!("instance {instance_id} not found");
        }

        if let Some((_, entry)) = self.workers.remove(&instance_id) {
            for record in entry.listeners.values() {
                if let Some(cancel_token) = record.take_cancel() {
                    cancel_token.cancel();
                }
            }
        } else {
            #[cfg(feature = "indexer-runtime")]
            {
                self.discovered_workers.remove(&instance_id);
            }
        }

        if let Some(ie) = self.indexers.get(&key) {
            ie.indexer.remove_worker(instance_id).await;
        }
        self.maybe_remove_indexer(&key);
        Ok(())
    }

    pub async fn deregister_dp_rank(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
        model_name: &str,
        tenant_id: &str,
    ) -> Result<()> {
        let key = IndexerKey {
            model_name: model_name.to_string(),
            tenant_id: tenant_id.to_string(),
        };

        let (record, remove_worker) = {
            let mut entry = self
                .workers
                .get_mut(&instance_id)
                .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

            if entry.key != key {
                bail!(
                    "instance {instance_id} is registered for model={} tenant={}",
                    entry.key.model_name,
                    entry.key.tenant_id
                );
            }

            let record = entry.listeners.remove(&dp_rank).ok_or_else(|| {
                anyhow::anyhow!("instance {instance_id} dp_rank {dp_rank} not found")
            })?;
            let remove_worker = entry.listeners.is_empty();
            (record, remove_worker)
        };

        if let Some(cancel_token) = record.take_cancel() {
            cancel_token.cancel();
        }

        if remove_worker {
            self.workers.remove(&instance_id);
            if let Some(ie) = self.indexers.get(&key) {
                ie.indexer.remove_worker(instance_id).await;
            }
            self.maybe_remove_indexer(&key);
        } else if let Some(ie) = self.indexers.get(&key) {
            ie.indexer.remove_worker_dp_rank(instance_id, dp_rank).await;
        }

        Ok(())
    }

    pub async fn deregister_all_tenants(
        &self,
        instance_id: WorkerId,
        model_name: &str,
    ) -> Result<()> {
        let key = if let Some(entry) = self.workers.get(&instance_id) {
            if entry.key.model_name != model_name {
                bail!(
                    "instance {instance_id} is registered for model={} tenant={}",
                    entry.key.model_name,
                    entry.key.tenant_id
                );
            }
            entry.key.clone()
        } else {
            #[cfg(feature = "indexer-runtime")]
            if let Some(discovered_key) = self.discovered_workers.get(&instance_id) {
                if discovered_key.value().model_name != model_name {
                    bail!(
                        "instance {instance_id} is registered for model={} tenant={}",
                        discovered_key.value().model_name,
                        discovered_key.value().tenant_id
                    );
                }
                discovered_key.value().clone()
            } else {
                bail!("instance {instance_id} not found");
            }

            #[cfg(not(feature = "indexer-runtime"))]
            bail!("instance {instance_id} not found");
        };

        if let Some((_, entry)) = self.workers.remove(&instance_id) {
            for record in entry.listeners.values() {
                if let Some(cancel_token) = record.take_cancel() {
                    cancel_token.cancel();
                }
            }
        } else {
            #[cfg(feature = "indexer-runtime")]
            {
                self.discovered_workers.remove(&instance_id);
            }
        }

        if let Some(ie) = self.indexers.get(&key) {
            ie.indexer.remove_worker(instance_id).await;
        }
        self.maybe_remove_indexer(&key);
        Ok(())
    }

    pub fn pause_listener(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(), ListenerControlError> {
        let record = if let Some(entry) = self.workers.get(&instance_id) {
            entry.listeners.get(&dp_rank).cloned().ok_or(
                ListenerControlError::ListenerNotFound {
                    instance_id,
                    dp_rank,
                },
            )?
        } else {
            #[cfg(feature = "indexer-runtime")]
            if self.discovered_workers.contains_key(&instance_id) {
                return Err(ListenerControlError::DiscoveryManaged { instance_id });
            }

            return Err(ListenerControlError::WorkerNotFound { instance_id });
        };

        let cancel_token = record.pause(instance_id, dp_rank)?;
        cancel_token.cancel();
        tracing::info!(instance_id, dp_rank, "Paused ZMQ listener");
        Ok(())
    }

    pub async fn resume_listener(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
    ) -> std::result::Result<(), ListenerControlError> {
        let record = if let Some(entry) = self.workers.get(&instance_id) {
            entry.listeners.get(&dp_rank).cloned().ok_or(
                ListenerControlError::ListenerNotFound {
                    instance_id,
                    dp_rank,
                },
            )?
        } else {
            #[cfg(feature = "indexer-runtime")]
            if self.discovered_workers.contains_key(&instance_id) {
                return Err(ListenerControlError::DiscoveryManaged { instance_id });
            }

            return Err(ListenerControlError::WorkerNotFound { instance_id });
        };

        let attempt = record.resume(instance_id, dp_rank)?;
        self.spawn_listener(instance_id, dp_rank, attempt, record);
        tracing::info!(instance_id, dp_rank, "Resumed ZMQ listener");
        Ok(())
    }

    pub fn list(&self) -> Vec<WorkerInfo> {
        #[allow(unused_mut)]
        let mut result: Vec<WorkerInfo> = self
            .workers
            .iter()
            .map(|entry| {
                let listeners: HashMap<u32, ListenerInfo> = entry
                    .value()
                    .listeners
                    .iter()
                    .map(|(dp_rank, record)| (*dp_rank, record.snapshot()))
                    .collect();
                let endpoints = listeners
                    .iter()
                    .map(|(dp_rank, info)| (*dp_rank, info.endpoint.clone()))
                    .collect();
                let status = ListenerStatus::aggregate(listeners.values().map(|info| info.status));
                WorkerInfo {
                    instance_id: *entry.key(),
                    source: WorkerSource::Zmq,
                    status,
                    endpoints,
                    listeners,
                }
            })
            .collect();

        #[cfg(feature = "indexer-runtime")]
        for entry in self.discovered_workers.iter() {
            let worker_id = *entry.key();
            if self.workers.contains_key(&worker_id) {
                continue;
            }
            result.push(WorkerInfo {
                instance_id: worker_id,
                source: WorkerSource::Discovery,
                status: ListenerStatus::Active,
                endpoints: HashMap::new(),
                listeners: HashMap::new(),
            });
        }

        result
    }

    pub fn get_indexer(&self, key: &IndexerKey) -> Option<Ref<'_, IndexerKey, IndexerEntry>> {
        self.indexers.get(key)
    }

    pub fn get_or_create_indexer(&self, key: IndexerKey, block_size: u32) -> Indexer {
        let entry = self.indexers.entry(key.clone()).or_insert_with(|| {
            tracing::info!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                block_size,
                "Creating indexer from recovery dump"
            );
            IndexerEntry {
                indexer: create_indexer(block_size, self.num_threads),
                block_size,
            }
        });
        if entry.block_size != block_size {
            tracing::warn!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                existing_block_size = entry.block_size,
                requested_block_size = block_size,
                "Block size mismatch for existing indexer"
            );
        }
        entry.indexer.clone()
    }

    pub fn all_indexers_with_block_size(&self) -> Vec<(IndexerKey, Indexer, u32)> {
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

    #[cfg(feature = "indexer-runtime")]
    pub fn add_worker_from_discovery(
        &self,
        instance_id: WorkerId,
        model_name: String,
        tenant_id: String,
        block_size: u32,
    ) -> Result<()> {
        if self.workers.contains_key(&instance_id) {
            bail!(
                "instance {instance_id} is already manually registered; \
                 cannot add via discovery"
            );
        }

        let key = IndexerKey {
            model_name,
            tenant_id,
        };

        if let Some(existing) = self.discovered_workers.get(&instance_id) {
            if existing.value() != &key {
                bail!(
                    "instance {instance_id} is already registered for model={} tenant={}",
                    existing.value().model_name,
                    existing.value().tenant_id
                );
            }
            return Ok(());
        }

        let indexer_entry = self.indexers.entry(key.clone()).or_insert_with(|| {
            tracing::info!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                block_size,
                "Creating new indexer (discovery)"
            );
            IndexerEntry {
                indexer: create_indexer(block_size, self.num_threads),
                block_size,
            }
        });

        if indexer_entry.block_size != block_size {
            bail!(
                "block_size mismatch for model={} tenant={}: existing={}, requested={}",
                key.model_name,
                key.tenant_id,
                indexer_entry.block_size,
                block_size
            );
        }
        drop(indexer_entry);

        self.discovered_workers.insert(instance_id, key);
        Ok(())
    }

    #[cfg(feature = "indexer-runtime")]
    pub async fn remove_worker_from_discovery(&self, instance_id: WorkerId) {
        if let Some((_, key)) = self.discovered_workers.remove(&instance_id) {
            if let Some(ie) = self.indexers.get(&key) {
                ie.indexer.remove_worker(instance_id).await;
            }
            self.maybe_remove_indexer(&key);
        } else {
            tracing::debug!(
                instance_id,
                "remove_worker_from_discovery: worker not in discovered_workers map"
            );
        }
    }

    #[cfg(feature = "indexer-runtime")]
    pub fn get_indexer_for_worker(&self, worker_id: WorkerId) -> Option<Indexer> {
        if let Some(key) = self.discovered_workers.get(&worker_id)
            && let Some(ie) = self.indexers.get(key.value())
        {
            return Some(ie.indexer.clone());
        }

        if let Some(entry) = self.workers.get(&worker_id)
            && let Some(ie) = self.indexers.get(&entry.key)
        {
            return Some(ie.indexer.clone());
        }

        None
    }

    fn spawn_listener(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
        (generation, cancel_token): (u64, CancellationToken),
        record: Arc<ListenerRecord>,
    ) {
        spawn_zmq_listener(
            instance_id,
            dp_rank,
            record,
            self.ready_rx(),
            generation,
            cancel_token.child_token(),
        );
    }

    fn maybe_remove_indexer(&self, key: &IndexerKey) {
        if self.workers.iter().any(|entry| entry.value().key == *key) {
            return;
        }

        #[cfg(feature = "indexer-runtime")]
        if self
            .discovered_workers
            .iter()
            .any(|entry| entry.value() == key)
        {
            return;
        }

        self.indexers.remove(key);
    }
}
