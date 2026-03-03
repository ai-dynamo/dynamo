// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Result, bail};
use dashmap::DashMap;
use dashmap::mapref::one::Ref;
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
    pub lora_name: Option<String>,
}

pub struct WorkerEntry {
    pub endpoints: HashMap<u32, String>,
    cancel: CancellationToken,
}

pub struct WorkerRegistry {
    workers: DashMap<WorkerId, WorkerEntry>,
    indexers: DashMap<IndexerKey, IndexerEntry>,
    num_threads: usize,
}

impl WorkerRegistry {
    pub fn new(num_threads: usize) -> Self {
        Self {
            workers: DashMap::new(),
            indexers: DashMap::new(),
            num_threads,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn register(
        &self,
        instance_id: WorkerId,
        endpoint: String,
        dp_rank: u32,
        model_name: String,
        tenant_id: String,
        block_size: u32,
        lora_name: Option<String>,
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
                lora_name = ?lora_name,
                "Creating new indexer"
            );
            IndexerEntry {
                indexer: create_indexer(block_size, self.num_threads),
                block_size,
                lora_name: lora_name.clone(),
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
                cancel: CancellationToken::new(),
            });

        if entry.endpoints.contains_key(&dp_rank) {
            bail!("instance {instance_id} dp_rank {dp_rank} already registered");
        }

        let child_cancel = entry.cancel.child_token();
        let addr = endpoint.clone();

        tokio::spawn(async move {
            run_zmq_listener(instance_id, dp_rank, addr, bs, indexer, child_cancel).await;
        });

        entry.endpoints.insert(dp_rank, endpoint);
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

        entry.cancel.cancel();

        let key = IndexerKey {
            model_name: model_name.to_string(),
            tenant_id: tenant_id.to_string(),
        };
        if let Some(ie) = self.indexers.get(&key) {
            ie.indexer.remove_worker(instance_id).await;
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

        if entry.endpoints.is_empty() {
            drop(entry);
            return self.deregister(instance_id, model_name, tenant_id).await;
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

        entry.cancel.cancel();

        for ie in self.indexers.iter() {
            if ie.key().model_name == model_name {
                ie.indexer.remove_worker(instance_id).await;
            }
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

    pub fn all_indexers(&self) -> Vec<(IndexerKey, Indexer)> {
        self.indexers
            .iter()
            .map(|entry| (entry.key().clone(), entry.value().indexer.clone()))
            .collect()
    }
}
