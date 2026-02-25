// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Result, bail};
use dashmap::DashMap;
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::protocols::WorkerId;

use super::indexer::Indexer;
use super::listener::run_zmq_listener;

pub struct WorkerEntry {
    pub zmq_addresses: HashMap<u32, String>,
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

    pub fn register(&self, worker_id: WorkerId, zmq_addresses: HashMap<u32, String>) -> Result<()> {
        let entry = self.workers.entry(worker_id);
        let dashmap::mapref::entry::Entry::Vacant(vacant) = entry else {
            bail!("worker {worker_id} already registered");
        };

        let cancel = CancellationToken::new();

        for (&dp_rank, addr) in &zmq_addresses {
            let indexer = self.indexer.clone();
            let block_size = self.block_size;
            let addr = addr.clone();
            let cancel_clone = cancel.clone();

            tokio::spawn(async move {
                run_zmq_listener(worker_id, dp_rank, addr, block_size, indexer, cancel_clone).await;
            });
        }

        vacant.insert(WorkerEntry {
            zmq_addresses,
            cancel,
        });
        Ok(())
    }

    pub async fn deregister(&self, worker_id: WorkerId) -> Result<()> {
        let (_, entry) = self
            .workers
            .remove(&worker_id)
            .ok_or_else(|| anyhow::anyhow!("worker {worker_id} not found"))?;

        entry.cancel.cancel();
        self.indexer.remove_worker(worker_id).await;
        Ok(())
    }

    pub fn list(&self) -> Vec<(WorkerId, HashMap<u32, String>)> {
        self.workers
            .iter()
            .map(|entry| (*entry.key(), entry.value().zmq_addresses.clone()))
            .collect()
    }

    pub fn indexer(&self) -> &Indexer {
        &self.indexer
    }
}
