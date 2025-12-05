// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use super::ConnectorLeader;

use crate::distributed::leader::InstanceLeader;
use crate::distributed::worker::{LeaderLayoutConfig, NovaWorkerClient, Worker};
use crate::integrations::connector::worker::ConnectorWorkerClient;
use crate::logical::blocks::BlockRegistry;
use crate::logical::manager::{BlockManager, FrequencyTrackingCapacity};
use crate::{G2, G3, InstanceId};

use anyhow::{Context, Result, anyhow, bail};
use dynamo_nova_backend::{PeerInfo, WorkerAddress};

impl ConnectorLeader {
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
    #[tracing::instrument(level = "debug", skip(self))]
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
    pub(crate) async fn initialize_async(&self) -> Result<()> {
        tracing::debug!("Starting initialize_async");

        // Step 1: Gather layout config futures while holding the lock
        tracing::debug!("Step 1: Acquiring lock to gather layout config futures");
        let layout_config_futures = {
            tracing::debug!("Lock acquired, checking worker count");
            let state = self.state.lock();

            if state.worker_connector_clients.is_empty() {
                bail!("No workers registered");
            }

            tracing::info!(
                num_workers = state.worker_connector_clients.len(),
                "Initializing workers"
            );

            tracing::debug!(
                num_workers = state.worker_connector_clients.len(),
                "Creating layout config futures for all workers"
            );
            let mut futures = Vec::with_capacity(state.worker_connector_clients.len());
            for (idx, worker) in state.worker_connector_clients.iter().enumerate() {
                tracing::debug!(worker_idx = idx, "Creating layout config future for worker");
                futures.push(worker.get_layout_config()?);
            }
            tracing::debug!(
                num_futures = futures.len(),
                "Created all layout config futures"
            );
            futures
        }; // Lock released here
        tracing::debug!("Lock released, starting to await layout configs");

        tracing::debug!(
            num_futures = layout_config_futures.len(),
            "Awaiting layout configs from workers"
        );
        let mut layout_configs = Vec::with_capacity(layout_config_futures.len());
        for (i, future) in layout_config_futures.into_iter().enumerate() {
            tracing::debug!(worker_idx = i, "Awaiting layout config from worker");
            let config = future
                .await
                .map_err(|e| anyhow!("Failed to get layout config from worker {}: {}", i, e))?;
            tracing::debug!(worker_idx = i, "Received layout config from worker");
            layout_configs.push(config);
        }
        tracing::debug!(
            num_configs = layout_configs.len(),
            "Completed awaiting all layout configs"
        );

        tracing::debug!(
            num_configs = layout_configs.len(),
            "Gathered layout configs from workers"
        );

        // Step 2: Validate all configs match
        tracing::debug!("Step 2: Validating all configs match");
        let reference_config = &layout_configs[0];
        tracing::debug!(
            num_layers = reference_config.num_layers,
            outer_dim = reference_config.outer_dim,
            page_size = reference_config.page_size,
            inner_dim = reference_config.inner_dim,
            dtype_width_bytes = reference_config.dtype_width_bytes,
            "Reference config (worker 0)"
        );
        for (i, config) in layout_configs.iter().enumerate().skip(1) {
            tracing::debug!(worker_idx = i, "Validating config for worker");
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
        tracing::debug!("Step 3: Computing G2/G3 block counts");
        let bytes_per_block = reference_config.required_bytes() / reference_config.num_blocks;
        tracing::debug!(
            bytes_per_block,
            num_blocks = reference_config.num_blocks,
            "Computed bytes per block"
        );

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
        tracing::debug!("Step 4: Building leader config");
        let leader_config = LeaderLayoutConfig {
            host_block_count,
            disk_block_count,
        };
        tracing::debug!(
            host_block_count = leader_config.host_block_count,
            disk_block_count = ?leader_config.disk_block_count,
            "Leader config built"
        );

        // Step 5: Initialize all workers in parallel
        tracing::debug!("Step 5: Acquiring lock to create initialize futures");
        let initialize_futures = {
            tracing::debug!("Lock acquired for creating initialize futures");
            let state = self.state.lock();
            tracing::debug!(
                num_workers = state.worker_connector_clients.len(),
                "Creating initialize futures for all workers"
            );
            let mut futures = Vec::with_capacity(state.worker_connector_clients.len());
            for (idx, worker) in state.worker_connector_clients.iter().enumerate() {
                tracing::debug!(worker_idx = idx, "Creating initialize future for worker");
                futures.push(worker.initialize(leader_config.clone())?);
            }
            tracing::debug!(
                num_futures = futures.len(),
                "Created all initialize futures"
            );
            futures
        }; // Lock released here
        tracing::debug!("Lock released, starting to await worker initializations");

        // Step 6: Await all initializations and collect worker metadata
        tracing::debug!(
            num_futures = initialize_futures.len(),
            "Step 6: Awaiting all worker initializations"
        );
        let mut worker_layouts = HashMap::new();
        let mut collected_metadata = Vec::new();

        for (i, future) in initialize_futures.into_iter().enumerate() {
            tracing::debug!(worker_idx = i, "Awaiting initialization for worker");
            let worker_layout = future
                .await
                .with_context(|| format!("Failed to initialize worker {}", i))?;
            tracing::debug!(worker_idx = i, "Worker initialization completed");

            // Collect metadata for later storage
            collected_metadata.push(worker_layout.metadata.clone());
            worker_layouts.insert(i, worker_layout);
        }
        tracing::debug!(
            num_workers = collected_metadata.len(),
            "All worker initializations completed"
        );

        // Store all metadata and configure worker handles
        tracing::debug!("Acquiring lock to store metadata and configure handles");
        {
            tracing::debug!("Lock acquired for storing metadata");
            let mut state = self.state.lock();
            tracing::debug!(
                num_metadata = collected_metadata.len(),
                "Storing worker metadata"
            );
            state.worker_metadata = collected_metadata.clone();

            // Configure layout handles for each NovaWorkerClient from their metadata
            tracing::debug!("Configuring layout handles for all workers");
            for (i, (client, metadata)) in state
                .worker_transfer_clients
                .iter()
                .zip(collected_metadata.iter())
                .enumerate()
            {
                tracing::debug!(worker_idx = i, "Configuring layout handles for worker");
                client
                    .configure_layout_handles(metadata)
                    .with_context(|| format!("Failed to configure handles for worker {}", i))?;
                tracing::debug!(worker_idx = i, "Layout handles configured for worker");
            }
        }
        tracing::debug!("Lock released, configured layout handles for all workers");

        tracing::debug!("Creating block registry");
        let registry = BlockRegistry::with_frequency_tracker(
            FrequencyTrackingCapacity::Medium.create_tracker(),
        );
        tracing::debug!("Block registry created");

        tracing::debug!(
            host_block_count,
            page_size = reference_config.page_size,
            "Building G2 manager"
        );
        let g2_manager = BlockManager::<G2>::builder()
            .block_count(host_block_count)
            .block_size(reference_config.page_size)
            .registry(registry.clone())
            .with_lru_backend()
            .build()
            .expect("Should build G2 manager");
        tracing::debug!("G2 manager built");

        tracing::debug!("Building G3 manager");
        let g3_manager = disk_block_count.map(|count| {
            tracing::debug!(
                disk_block_count = count,
                page_size = reference_config.page_size,
                "Building G3 manager with disk cache"
            );
            BlockManager::<G3>::builder()
                .block_count(count)
                .block_size(reference_config.page_size)
                .registry(registry.clone())
                .with_lru_backend()
                .build()
                .expect("Should build G3 manager")
        });
        tracing::debug!("G3 manager built (if configured)");

        tracing::debug!("Acquiring lock to get worker clients and metadata");
        let (worker_clients, worker_metadata) = {
            tracing::debug!("Lock acquired for getting worker data");
            let state = self.state.lock();
            tracing::debug!(
                num_clients = state.worker_transfer_clients.len(),
                num_metadata = state.worker_metadata.len(),
                "Cloning worker clients and metadata"
            );
            (
                state.worker_transfer_clients.clone(),
                state.worker_metadata.clone(),
            )
        };
        tracing::debug!("Lock released, building InstanceLeader");

        tracing::debug!(
            num_workers = worker_clients.len(),
            "Building InstanceLeader"
        );
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
        tracing::debug!("InstanceLeader built");

        tracing::debug!("Registering handlers on InstanceLeader");
        leader.register_handlers()?;
        tracing::debug!("Handlers registered");

        tracing::debug!("Setting instance leader");
        self.set_instance_leader(leader)?;
        tracing::debug!("Instance leader set");

        tracing::info!("All workers initialized successfully");

        // Refresh handler lists for all workers since they registered new handlers during init
        // This clears the stale cache from the initial handshake (which only had connector handlers)
        tracing::debug!("Acquiring lock to get worker instance IDs for handler refresh");
        let worker_instance_ids = {
            tracing::debug!("Lock acquired for getting worker instance IDs");
            let state = self.state.lock();
            tracing::debug!(
                num_workers = state.worker_instance_ids.len(),
                "Cloning worker instance IDs"
            );
            state.worker_instance_ids.clone()
        };
        tracing::debug!("Lock released, starting handler refresh");

        tracing::debug!(
            num_workers = worker_instance_ids.len(),
            "Refreshing handler lists for all workers"
        );
        for (idx, instance_id) in worker_instance_ids.iter().enumerate() {
            tracing::debug!(
                worker_idx = idx,
                instance_id = ?instance_id,
                "Refreshing handlers for worker"
            );
            self.runtime
                .nova
                .refresh_handlers(*instance_id)
                .await
                .with_context(|| {
                    format!("Failed to refresh handlers for worker {}", instance_id)
                })?;
            tracing::debug!(
                worker_idx = idx,
                instance_id = ?instance_id,
                "Handler refresh completed for worker"
            );
        }

        tracing::debug!(
            num_workers = worker_instance_ids.len(),
            "Refreshed handler lists for all workers"
        );

        tracing::debug!("initialize_async completed successfully");
        Ok(())
    }
}
