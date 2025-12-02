// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::worker::ConnectorWorkerClient;
use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::distributed::worker::LeaderLayoutConfig;
use crate::{InstanceId, KvbmRuntime};

use dynamo_nova_backend::{PeerInfo, WorkerAddress};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use std::sync::{Arc, OnceLock};

pub trait ConnectorLeaderInterface: Send + Sync {}

pub struct ConnectorLeader {
    pub(crate) runtime: KvbmRuntime,
    state: Arc<Mutex<ConnectorLeaderState>>,
    instance_leader: OnceLock<InstanceLeader>,
}

#[derive(Default)]
struct ConnectorLeaderState {
    worker_instance_ids: Vec<InstanceId>,
    worker_clients: Vec<ConnectorWorkerClient>,
}

impl ConnectorLeader {
    pub fn new(runtime: KvbmRuntime) -> Self {
        Self {
            runtime,
            state: Arc::new(Mutex::new(ConnectorLeaderState::default())),
            instance_leader: OnceLock::new(),
        }
    }

    /// Access the InstanceLeader (available after initialize_workers()).
    pub fn instance_leader(&self) -> Option<&InstanceLeader> {
        self.instance_leader.get()
    }

    /// Set the InstanceLeader (called by test infrastructure after worker initialization).
    ///
    /// This is typically called by ConnectorInstance after workers are initialized
    /// and we have access to their DirectWorker instances.
    pub fn set_instance_leader(&self, leader: InstanceLeader) -> Result<()> {
        self.instance_leader
            .set(leader)
            .map_err(|_| anyhow!("InstanceLeader already set"))
    }

    pub fn register_worker(
        &self,
        rank: usize,
        instance_id: InstanceId,
        worker_address: WorkerAddress,
    ) -> Result<()> {
        let mut state = self.state.lock();

        if rank != state.worker_instance_ids.len() {
            bail!("Rank mismatch");
        }

        self.runtime
            .nova
            .register_peer(PeerInfo::new(instance_id, worker_address))?;

        state.worker_instance_ids.push(instance_id);
        state.worker_clients.push(ConnectorWorkerClient::new(
            self.runtime.nova.clone(),
            instance_id,
        ));

        Ok(())
    }

    pub fn initialize_workers(&self) -> Result<()> {
        let state = self.state.lock();

        if state.worker_clients.is_empty() {
            bail!("No workers registered");
        }

        tracing::info!(
            num_workers = state.worker_clients.len(),
            "Initializing workers"
        );

        // Step 1: Gather layout configs from all workers
        let mut layout_config_futures = Vec::with_capacity(state.worker_clients.len());
        for worker in &state.worker_clients {
            layout_config_futures.push(worker.get_layout_config()?);
        }

        let mut layout_configs = Vec::with_capacity(state.worker_clients.len());
        for (i, future) in layout_config_futures.into_iter().enumerate() {
            let config = self
                .runtime
                .tokio()
                .block_on(future)
                .map_err(|e| anyhow!("Failed to get layout config from worker {}: {}", i, e))?;
            layout_configs.push(config);
        }

        tracing::debug!(
            num_configs = layout_configs.len(),
            "Gathered layout configs from workers"
        );

        // Step 2: Validate all configs match
        let reference_config = &layout_configs[0];
        for (i, config) in layout_configs.iter().enumerate().skip(1) {
            if config.num_layers != reference_config.num_layers {
                bail!(
                    "Layout config mismatch: worker {} has {} layers, worker 0 has {}",
                    i,
                    config.num_layers,
                    reference_config.num_layers
                );
            }
            if config.outer_dim != reference_config.outer_dim {
                bail!(
                    "Layout config mismatch: worker {} has outer_dim {}, worker 0 has {}",
                    i,
                    config.outer_dim,
                    reference_config.outer_dim
                );
            }
            if config.page_size != reference_config.page_size {
                bail!(
                    "Layout config mismatch: worker {} has page_size {}, worker 0 has {}",
                    i,
                    config.page_size,
                    reference_config.page_size
                );
            }
            if config.inner_dim != reference_config.inner_dim {
                bail!(
                    "Layout config mismatch: worker {} has inner_dim {}, worker 0 has {}",
                    i,
                    config.inner_dim,
                    reference_config.inner_dim
                );
            }
            if config.dtype_width_bytes != reference_config.dtype_width_bytes {
                bail!(
                    "Layout config mismatch: worker {} has dtype_width_bytes {}, worker 0 has {}",
                    i,
                    config.dtype_width_bytes,
                    reference_config.dtype_width_bytes
                );
            }
        }

        tracing::info!("All worker layout configs match");

        // Step 3: Compute G2/G3 block counts from leader config
        let bytes_per_block = reference_config.required_bytes() / reference_config.num_blocks;

        let host_block_count = self
            .runtime
            .config()
            .cache
            .host
            .compute_num_blocks(bytes_per_block)
            .unwrap_or(0);

        let disk_block_count = self
            .runtime
            .config()
            .cache
            .disk
            .as_ref()
            .and_then(|dc| dc.compute_num_blocks(bytes_per_block));

        tracing::info!(
            host_block_count,
            ?disk_block_count,
            bytes_per_block,
            "Computed block counts for G2/G3 tiers"
        );

        // Step 4: Build leader config and send to workers
        let leader_config = LeaderLayoutConfig {
            host_block_count,
            disk_block_count,
        };

        // Step 5: Initialize all workers in parallel
        let mut initialize_futures = Vec::with_capacity(state.worker_clients.len());
        for worker in &state.worker_clients {
            initialize_futures.push(worker.initialize(leader_config.clone())?);
        }

        // Step 6: Await all initializations
        let mut worker_layouts = Vec::with_capacity(state.worker_clients.len());
        for (i, future) in initialize_futures.into_iter().enumerate() {
            let worker_layout = self
                .runtime
                .tokio()
                .block_on(future)
                .with_context(|| format!("Failed to initialize worker {}", i))?;
            worker_layouts.push(worker_layout);
        }

        // todo: build instance leader

        tracing::info!("All workers initialized successfully");

        Ok(())
    }
}
