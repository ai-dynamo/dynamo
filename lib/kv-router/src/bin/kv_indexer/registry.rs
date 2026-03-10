// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dashmap::mapref::one::Ref;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::WorkerId;

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
    cancels: HashMap<u32, CancellationToken>,
}

pub struct WorkerRegistry {
    workers: DashMap<WorkerId, WorkerEntry>,
    indexers: DashMap<IndexerKey, IndexerEntry>,
    peers: DashMap<String, ()>,
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

    pub fn register(
        &self,
        instance_id: WorkerId,
        endpoint: String,
        dp_rank: u32,
        model_name: String,
        tenant_id: String,
        block_size: u32,
    ) -> Result<()> {
        let key = IndexerKey {
            model_name,
            tenant_id,
        };

        // Get or create the indexer for this (model, tenant) pair.
        // Use the entry API for atomic check-and-insert.
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

        let mut entry = self
            .workers
            .entry(instance_id)
            .or_insert_with(|| WorkerEntry {
                endpoints: HashMap::new(),
                cancels: HashMap::new(),
            });

        if entry.endpoints.contains_key(&dp_rank) {
            bail!("instance {instance_id} dp_rank {dp_rank} already registered");
        }

        let cancel = CancellationToken::new();
        let child_cancel = cancel.child_token();
        let addr = endpoint.clone();
        let ready = self.ready_rx();

        tokio::spawn(async move {
            run_zmq_listener(instance_id, dp_rank, addr, bs, indexer, child_cancel, ready).await;
        });

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
        let (_, entry) = self
            .workers
            .remove(&instance_id)
            .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

        for cancel in entry.cancels.values() {
            cancel.cancel();
        }

        let key = IndexerKey {
            model_name: model_name.to_string(),
            tenant_id: tenant_id.to_string(),
        };
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
        let (_, entry) = self
            .workers
            .remove(&instance_id)
            .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

        for cancel in entry.cancels.values() {
            cancel.cancel();
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

    pub fn list(&self) -> Vec<(WorkerId, HashMap<u32, String>)> {
        self.workers
            .iter()
            .map(|entry| (*entry.key(), entry.value().endpoints.clone()))
            .collect()
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
}
