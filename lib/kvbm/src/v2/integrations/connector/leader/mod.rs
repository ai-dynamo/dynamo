// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::worker::ConnectorWorkerClient;
use crate::distributed::leader::Leader;
use crate::distributed::worker::{NovaWorkerClient, Worker};
use crate::logical::blocks::BlockRegistry;
use crate::logical::manager::{BlockManager, FrequencyTrackingCapacity};
use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::distributed::worker::{LeaderLayoutConfig, SerializedLayout};
use crate::{G2, G3, InstanceId, KvbmRuntime};

use dynamo_nova_backend::{PeerInfo, WorkerAddress};

use anyhow::{Context, Result, anyhow, bail};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ops::Deref;
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
    worker_connector_clients: Vec<ConnectorWorkerClient>,
    worker_transfer_clients: Vec<NovaWorkerClient>,
    worker_metadata: Vec<SerializedLayout>,
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

    /// This is called by the Scheduler-side of the ConnectorAPI during the call to set_xfer_handshake_metadata.
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
        state
            .worker_connector_clients
            .push(ConnectorWorkerClient::new(
                self.runtime.nova.clone(),
                instance_id,
            ));
        state.worker_transfer_clients.push(NovaWorkerClient::new(
            self.runtime.nova.clone(),
            instance_id,
        ));

        Ok(())
    }

    /// Initialize all workers via leader-driven deferred init flow (blocking version).
    ///
    /// NOTE: This uses block_on internally and should only be called from a blocking context.
    /// For async contexts, use `initialize_workers_async`.
    pub fn initialize(self: &Arc<Self>) -> Result<()> {
        let (tx, rx) = oneshot::channel();
        let this = self.clone();
        self.runtime.tokio().spawn(async move {
            let result = this.initialize_async().await;
            if tx.send(result).is_err() {
                bail!("Failed to send result to channel");
            }
            Ok(())
        });
        rx.recv()??;
        Ok(())
    }

    /// Initialize all workers via leader-driven deferred init flow (async version).
    /// This is primarily used for use and testing outside of the ConnectorAPI.
    pub async fn initialize_async(&self) -> Result<()> {
        // Step 1: Gather layout config futures while holding the lock
        let layout_config_futures = {
            let state = self.state.lock();

            if state.worker_connector_clients.is_empty() {
                bail!("No workers registered");
            }

            tracing::info!(
                num_workers = state.worker_connector_clients.len(),
                "Initializing workers"
            );

            let mut futures = Vec::with_capacity(state.worker_connector_clients.len());
            for worker in &state.worker_connector_clients {
                futures.push(worker.get_layout_config()?);
            }
            futures
        }; // Lock released here

        let mut layout_configs = Vec::with_capacity(layout_config_futures.len());
        for (i, future) in layout_config_futures.into_iter().enumerate() {
            let config = future
                .await
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
        let initialize_futures = {
            let state = self.state.lock();
            let mut futures = Vec::with_capacity(state.worker_connector_clients.len());
            for worker in &state.worker_connector_clients {
                futures.push(worker.initialize(leader_config.clone())?);
            }
            futures
        }; // Lock released here

        // Step 6: Await all initializations and collect worker metadata
        let mut worker_layouts = HashMap::new();
        let mut collected_metadata = Vec::new();

        for (i, future) in initialize_futures.into_iter().enumerate() {
            let worker_layout = future
                .await
                .with_context(|| format!("Failed to initialize worker {}", i))?;

            // Collect metadata for later storage
            collected_metadata.push(worker_layout.metadata.clone());
            worker_layouts.insert(i, worker_layout);
        }

        // Store all metadata and configure worker handles
        {
            let mut state = self.state.lock();
            state.worker_metadata = collected_metadata.clone();

            // Configure layout handles for each NovaWorkerClient from their metadata
            for (i, (client, metadata)) in state
                .worker_transfer_clients
                .iter()
                .zip(collected_metadata.iter())
                .enumerate()
            {
                client
                    .configure_layout_handles(metadata)
                    .with_context(|| format!("Failed to configure handles for worker {}", i))?;
            }
        }

        tracing::debug!("Configured layout handles for all workers");

        let registry = BlockRegistry::with_frequency_tracker(
            FrequencyTrackingCapacity::Medium.create_tracker(),
        );

        let g2_manager = BlockManager::<G2>::builder()
            .block_count(host_block_count)
            .block_size(reference_config.page_size)
            .registry(registry.clone())
            .with_lru_backend()
            .build()
            .expect("Should build G2 manager");

        let g3_manager = disk_block_count.map(|count| {
            BlockManager::<G3>::builder()
                .block_count(count)
                .block_size(reference_config.page_size)
                .registry(registry.clone())
                .with_lru_backend()
                .build()
                .expect("Should build G3 manager")
        });

        let (worker_clients, worker_metadata) = {
            let state = self.state.lock();
            (
                state.worker_transfer_clients.clone(),
                state.worker_metadata.clone(),
            )
        };

        let leader = InstanceLeader::builder()
            .nova(self.runtime.nova.clone())
            .registry(registry)
            .with_g2_manager(Some(g2_manager))
            .with_g3_manager(g3_manager)
            .workers(
                worker_clients
                    .into_iter()
                    .map(|client| Arc::new(client) as Arc<dyn Worker>)
                    .collect(),
            )
            .with_cached_worker_metadata(worker_metadata)
            .build()?;

        leader.register_handlers()?;

        self.set_instance_leader(leader)?;

        tracing::info!("All workers initialized successfully");

        // Refresh handler lists for all workers since they registered new handlers during init
        // This clears the stale cache from the initial handshake (which only had connector handlers)
        let worker_instance_ids = {
            let state = self.state.lock();
            state.worker_instance_ids.clone()
        };

        for instance_id in &worker_instance_ids {
            self.runtime
                .nova
                .refresh_handlers(*instance_id)
                .await
                .with_context(|| {
                    format!("Failed to refresh handlers for worker {}", instance_id)
                })?;
        }

        tracing::debug!(
            num_workers = worker_instance_ids.len(),
            "Refreshed handler lists for all workers"
        );

        Ok(())
    }
}

impl Deref for ConnectorLeader {
    type Target = dyn Leader;

    fn deref(&self) -> &Self::Target {
        self.instance_leader.get().expect("InstanceLeader not set")
    }
}
