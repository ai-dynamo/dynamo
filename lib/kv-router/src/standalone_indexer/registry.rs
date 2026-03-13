// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dashmap::mapref::one::Ref;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::protocols::WorkerId;

use super::indexer::{Indexer, create_indexer};
use super::listener::run_zmq_listener;

#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct IndexerKey {
    pub model_name: String,
    pub tenant_id: String,
}

pub struct IndexerEntry {
    pub indexer: Indexer,
    pub block_size: u32,
}

pub struct WorkerEntry {
    pub endpoints: HashMap<u32, String>,
    pub replay_endpoints: HashMap<u32, String>,
    cancels: HashMap<u32, CancellationToken>,
}

/// State needed to restart a paused ZMQ listener.
struct ListenerState {
    endpoint: String,
    replay_endpoint: Option<String>,
    block_size: u32,
    indexer: Indexer,
    watermark: Arc<AtomicU64>,
}

pub struct WorkerRegistry {
    workers: DashMap<WorkerId, WorkerEntry>,
    indexers: DashMap<IndexerKey, IndexerEntry>,
    peers: DashMap<String, ()>,
    /// Persists across unregister/register cycles so gap detection works after re-registration.
    watermarks: DashMap<(WorkerId, u32), Arc<AtomicU64>>,
    /// Saved listener state for pause/resume. Populated on register, kept on pause.
    listener_states: DashMap<(WorkerId, u32), ListenerState>,
    /// Workers added via MDC discovery (no ZMQ listener). Maps worker_id → indexer key.
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
            listener_states: DashMap::new(),
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
        // Reject if this worker was already added via discovery
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

        let indexer_entry = self.indexers.entry(key.clone()).or_insert_with(|| {
            tracing::info!(
                model_name = %key.model_name,
                tenant_id = %key.tenant_id,
                block_size,
                "Creating new indexer"
            );
            super::metrics::inc_models();
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

        // Check for duplicate and insert replay endpoint while holding the lock briefly.
        {
            let mut entry = self.workers.entry(instance_id).or_insert_with(|| {
                super::metrics::inc_workers();
                WorkerEntry {
                    endpoints: HashMap::new(),
                    replay_endpoints: HashMap::new(),
                    cancels: HashMap::new(),
                }
            });

            if entry.endpoints.contains_key(&dp_rank) {
                bail!("instance {instance_id} dp_rank {dp_rank} already registered");
            }

            if let Some(rep) = &replay_endpoint {
                entry.replay_endpoints.insert(dp_rank, rep.clone());
            }
        }

        // Reuse watermark if it survived a previous unregister (preserves gap detection).
        let watermark = self
            .watermarks
            .entry((instance_id, dp_rank))
            .or_insert_with(|| Arc::new(AtomicU64::new(u64::MAX)))
            .clone();

        self.listener_states.insert(
            (instance_id, dp_rank),
            ListenerState {
                endpoint: endpoint.clone(),
                replay_endpoint: replay_endpoint.clone(),
                block_size: bs,
                indexer: indexer.clone(),
                watermark: watermark.clone(),
            },
        );

        let cancel = CancellationToken::new();
        let child_cancel = cancel.child_token();
        let addr = endpoint.clone();
        let ready = self.ready_rx();

        // Connect the ZMQ socket and spawn the listener task (non-blocking).
        run_zmq_listener(
            instance_id,
            dp_rank,
            addr,
            bs,
            indexer,
            child_cancel,
            ready,
            replay_endpoint,
            watermark,
        )
        .await;

        // Re-acquire to store the endpoint and cancel token.
        let mut entry = self
            .workers
            .get_mut(&instance_id)
            .expect("worker entry disappeared during listener setup");
        entry.endpoints.insert(dp_rank, endpoint);
        entry.cancels.insert(dp_rank, cancel);
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

        // Check ZMQ-registered workers first
        if let Some((_, entry)) = self.workers.remove(&instance_id) {
            super::metrics::dec_workers();
            for cancel in entry.cancels.values() {
                cancel.cancel();
            }
        } else if self.discovered_workers.remove(&instance_id).is_some() {
            // Discovered worker — no ZMQ listeners to cancel
            super::metrics::dec_workers();
            tracing::info!(instance_id, "Deregistering discovered worker via HTTP");
        } else {
            bail!("instance {instance_id} not found");
        }

        if let Some(ie) = self.indexers.get(&key) {
            ie.indexer.remove_worker(instance_id).await;
        } else {
            tracing::warn!(
                instance_id,
                model_name,
                tenant_id,
                "indexer key not found on deregister; tree will not be cleaned up"
            );
        }

        Ok(())
    }

    pub async fn deregister_dp_rank(
        &self,
        instance_id: WorkerId,
        dp_rank: u32,
        model_name: &str,
        tenant_id: &str,
    ) -> Result<()> {
        let mut entry = self
            .workers
            .get_mut(&instance_id)
            .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

        if entry.endpoints.remove(&dp_rank).is_none() {
            bail!("instance {instance_id} dp_rank {dp_rank} not found");
        }

        if let Some(cancel) = entry.cancels.remove(&dp_rank) {
            cancel.cancel();
        }

        if entry.endpoints.is_empty() {
            drop(entry);
            return self.deregister(instance_id, model_name, tenant_id).await;
        }
        drop(entry);

        let key = IndexerKey {
            model_name: model_name.to_string(),
            tenant_id: tenant_id.to_string(),
        };
        if let Some(ie) = self.indexers.get(&key) {
            ie.indexer.remove_worker_dp_rank(instance_id, dp_rank).await;
        } else {
            tracing::warn!(
                instance_id,
                dp_rank,
                model_name,
                tenant_id,
                "indexer key not found on deregister_dp_rank; tree will not be cleaned up"
            );
        }

        Ok(())
    }

    pub async fn deregister_all_tenants(
        &self,
        instance_id: WorkerId,
        model_name: &str,
    ) -> Result<()> {
        // Check ZMQ-registered workers first
        if let Some((_, entry)) = self.workers.remove(&instance_id) {
            super::metrics::dec_workers();
            for cancel in entry.cancels.values() {
                cancel.cancel();
            }
        } else if self.discovered_workers.remove(&instance_id).is_some() {
            super::metrics::dec_workers();
            tracing::info!(
                instance_id,
                "Deregistering discovered worker (all tenants) via HTTP"
            );
        } else {
            bail!("instance {instance_id} not found");
        }

        let mut found = false;
        for ie in self.indexers.iter() {
            if ie.key().model_name == model_name {
                ie.indexer.remove_worker(instance_id).await;
                found = true;
            }
        }
        if !found {
            tracing::warn!(
                instance_id,
                model_name,
                "no indexers found for model on deregister_all_tenants; tree will not be cleaned up"
            );
        }

        Ok(())
    }

    #[cfg_attr(not(feature = "test-endpoints"), allow(dead_code))]
    pub fn pause_listener(&self, instance_id: WorkerId, dp_rank: u32) -> Result<()> {
        let mut entry = self
            .workers
            .get_mut(&instance_id)
            .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

        let cancel = entry.cancels.remove(&dp_rank).ok_or_else(|| {
            anyhow::anyhow!("instance {instance_id} dp_rank {dp_rank} not active")
        })?;
        cancel.cancel();

        tracing::info!(instance_id, dp_rank, "Paused ZMQ listener");
        Ok(())
    }

    #[cfg_attr(not(feature = "test-endpoints"), allow(dead_code))]
    pub async fn resume_listener(&self, instance_id: WorkerId, dp_rank: u32) -> Result<()> {
        {
            let entry = self
                .workers
                .get(&instance_id)
                .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

            if entry.cancels.contains_key(&dp_rank) {
                bail!("instance {instance_id} dp_rank {dp_rank} already running");
            }
        }

        let state = self
            .listener_states
            .get(&(instance_id, dp_rank))
            .ok_or_else(|| anyhow::anyhow!("no saved state for {instance_id} dp_rank {dp_rank}"))?;

        let cancel = CancellationToken::new();
        let child_cancel = cancel.child_token();
        let ready = self.ready_rx();
        let addr = state.endpoint.clone();
        let bs = state.block_size;
        let indexer = state.indexer.clone();
        let replay_ep = state.replay_endpoint.clone();
        let watermark = state.watermark.clone();
        drop(state);

        run_zmq_listener(
            instance_id,
            dp_rank,
            addr,
            bs,
            indexer,
            child_cancel,
            ready,
            replay_ep,
            watermark,
        )
        .await;

        let mut entry = self
            .workers
            .get_mut(&instance_id)
            .expect("worker entry disappeared during listener resume");
        entry.cancels.insert(dp_rank, cancel);
        Ok(())
    }

    pub fn list(&self) -> Vec<(WorkerId, HashMap<u32, String>)> {
        let mut result: Vec<(WorkerId, HashMap<u32, String>)> = self
            .workers
            .iter()
            .map(|entry| (*entry.key(), entry.value().endpoints.clone()))
            .collect();

        // Include discovered workers (no ZMQ endpoints)
        for entry in self.discovered_workers.iter() {
            let worker_id = *entry.key();
            // Skip if already in the workers map (shouldn't happen, but be safe)
            if self.workers.contains_key(&worker_id) {
                continue;
            }
            result.push((worker_id, HashMap::new()));
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

    // ---------------------------------------------------------------
    // Discovery-based worker management (no ZMQ listener)
    // ---------------------------------------------------------------

    /// Register a worker discovered via MDC. Creates the indexer if needed but
    /// does NOT start a ZMQ listener — events arrive via the event plane.
    pub fn add_worker_from_discovery(
        &self,
        instance_id: WorkerId,
        model_name: String,
        tenant_id: String,
        block_size: u32,
    ) -> Result<()> {
        // Reject if this worker is already registered via ZMQ (--workers or /register)
        if self.workers.contains_key(&instance_id) {
            bail!(
                "instance {instance_id} is already registered via ZMQ; \
                 cannot add via discovery"
            );
        }

        let key = IndexerKey {
            model_name,
            tenant_id,
        };

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

    /// Remove a worker that was discovered via MDC.
    pub async fn remove_worker_from_discovery(&self, instance_id: WorkerId) {
        if let Some((_, key)) = self.discovered_workers.remove(&instance_id) {
            if let Some(ie) = self.indexers.get(&key) {
                ie.indexer.remove_worker(instance_id).await;
            }
        } else {
            tracing::debug!(
                instance_id,
                "remove_worker_from_discovery: worker not in discovered_workers map"
            );
        }
    }

    /// Look up the indexer responsible for a given worker_id.
    /// Checks both discovery-registered and CLI-registered workers.
    pub fn get_indexer_for_worker(&self, worker_id: WorkerId) -> Option<Indexer> {
        // Check discovery workers first (more common in runtime mode)
        if let Some(key) = self.discovered_workers.get(&worker_id)
            && let Some(ie) = self.indexers.get(key.value())
        {
            return Some(ie.indexer.clone());
        }
        // Fall back for legacy --workers mode: only if this worker is actually
        // in the ZMQ-registered workers map, route to the first indexer.
        if self.workers.contains_key(&worker_id) {
            return self.indexers.iter().next().map(|ie| ie.indexer.clone());
        }
        None
    }
}
