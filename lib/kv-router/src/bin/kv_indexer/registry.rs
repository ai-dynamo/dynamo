// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Result, bail};
use dashmap::DashMap;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::WorkerId;

use super::indexer::Indexer;
use super::listener::run_zmq_listener;

pub struct EndpointEntry {
    pub endpoint: String,
}

pub struct WorkerEntry {
    pub endpoints: HashMap<u32, EndpointEntry>,
    cancel: CancellationToken,
}

pub struct WorkerRegistry {
    workers: DashMap<WorkerId, WorkerEntry>,
    indexer: Indexer,
    block_size: u32,
}

impl WorkerRegistry {
    pub fn new(indexer: Indexer, block_size: u32) -> Self {
        Self {
            workers: DashMap::new(),
            indexer,
            block_size,
        }
    }

    pub fn register(&self, instance_id: WorkerId, endpoint: String, dp_rank: u32) -> Result<()> {
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
        let indexer = self.indexer.clone();
        let block_size = self.block_size;
        let addr = endpoint.clone();

        tokio::spawn(async move {
            run_zmq_listener(
                instance_id,
                dp_rank,
                addr,
                block_size,
                indexer,
                child_cancel,
            )
            .await;
        });

        entry.endpoints.insert(dp_rank, EndpointEntry { endpoint });
        Ok(())
    }

    pub async fn deregister(&self, instance_id: WorkerId) -> Result<()> {
        let (_, entry) = self
            .workers
            .remove(&instance_id)
            .ok_or_else(|| anyhow::anyhow!("instance {instance_id} not found"))?;

        entry.cancel.cancel();
        self.indexer.remove_worker(instance_id).await;
        Ok(())
    }

    pub fn list(&self) -> Vec<(WorkerId, HashMap<u32, String>)> {
        self.workers
            .iter()
            .map(|entry| {
                let endpoints: HashMap<u32, String> = entry
                    .value()
                    .endpoints
                    .iter()
                    .map(|(&dp_rank, e)| (dp_rank, e.endpoint.clone()))
                    .collect();
                (*entry.key(), endpoints)
            })
            .collect()
    }

    pub fn indexer(&self) -> &Indexer {
        &self.indexer
    }
}
