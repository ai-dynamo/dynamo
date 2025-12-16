// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use super::ConnectorLeader;

use crate::distributed::leader::InstanceLeader;
use crate::distributed::worker::{LeaderLayoutConfig, NovaWorkerClient, Worker};
use crate::integrations::connector::worker::ConnectorWorkerClient;
use crate::logical::blocks::{BlockDuplicationPolicy, BlockRegistry};
use crate::logical::manager::{BlockManager, FrequencyTrackingCapacity};
use crate::v2::distributed::object::{ObjectLockManager, create_lock_manager};
use crate::v2::distributed::offload::{
    ObjectPipelineBuilder, ObjectPresenceFilter, OffloadEngine, PendingTracker, PipelineBuilder,
    S3PresenceChecker, create_policy_from_config,
};
use crate::{G1, G2, G3, InstanceId};

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
        let mut state = self.init.lock();

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
    pub(crate) async fn initialize_async(self: Arc<Self>) -> Result<()> {
        tracing::debug!("Starting initialize_async");

        // Step 1: Gather layout config futures while holding the lock
        tracing::debug!("Step 1: Acquiring lock to gather layout config futures");
        let layout_config_futures = {
            tracing::debug!("Lock acquired, checking worker count");
            let state = self.init.lock();

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

        tracing::debug!(
            host_block_count,
            disk_block_count,
            "Issuing leader config to workers"
        );

        // Step 4: Initialize all workers in parallel
        tracing::debug!("Step 5: Acquiring lock to create initialize futures");
        let initialize_futures = {
            tracing::debug!("Lock acquired for creating initialize futures");
            let state = self.init.lock();
            tracing::debug!(
                num_workers = state.worker_connector_clients.len(),
                "Creating initialize futures for all workers"
            );
            let mut futures = Vec::with_capacity(state.worker_connector_clients.len());
            let object_config = self.runtime.config().object.clone();
            for (idx, worker) in state.worker_connector_clients.iter().enumerate() {
                tracing::trace!(worker_idx = idx, "Creating initialize future for worker");
                let leader_config = LeaderLayoutConfig {
                    rank: idx,
                    host_block_count,
                    disk_block_count,
                    object: object_config.clone(),
                };
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
            tracing::trace!(worker_idx = i, "Awaiting initialization for worker");
            let worker_layout = future
                .await
                .with_context(|| format!("Failed to initialize worker {}", i))?;
            tracing::trace!(worker_idx = i, "Worker initialization completed");

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
            tracing::trace!(
                num_metadata = collected_metadata.len(),
                "Storing worker metadata"
            );
            tracing::trace!("Lock acquired for storing metadata");
            let mut state = self.init.lock();
            state.worker_metadata = collected_metadata.clone();

            // Configure layout handles for each NovaWorkerClient from their metadata
            tracing::debug!("Configuring layout handles for all workers");
            for (i, (client, metadata)) in state
                .worker_transfer_clients
                .iter()
                .zip(collected_metadata.iter())
                .enumerate()
            {
                tracing::trace!(worker_idx = i, "Configuring layout handles for worker");
                client
                    .configure_layout_handles(metadata)
                    .with_context(|| format!("Failed to configure handles for worker {}", i))?;
                tracing::trace!(worker_idx = i, "Layout handles configured for worker");
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
        let g2_manager = Arc::new(
            BlockManager::<G2>::builder()
                .block_count(host_block_count)
                .block_size(reference_config.page_size)
                .registry(registry.clone())
                .with_lru_backend()
                .duplication_policy(BlockDuplicationPolicy::Reject)
                .build()
                .expect("Should build G2 manager"),
        );
        tracing::debug!("G2 manager built");

        tracing::debug!("Building G3 manager");
        let g3_manager: Option<Arc<BlockManager<G3>>> = disk_block_count.map(|count| {
            tracing::debug!(
                disk_block_count = count,
                page_size = reference_config.page_size,
                "Building G3 manager with disk cache"
            );
            Arc::new(
                BlockManager::<G3>::builder()
                    .block_count(count)
                    .block_size(reference_config.page_size)
                    .registry(registry.clone())
                    .with_lru_backend()
                    .duplication_policy(BlockDuplicationPolicy::Reject)
                    .build()
                    .expect("Should build G3 manager"),
            )
        });
        tracing::debug!("G3 manager built (if configured)");

        tracing::debug!("Acquiring lock to get worker clients and metadata");
        let (worker_clients, worker_metadata) = {
            tracing::debug!("Lock acquired for getting worker data");
            let state = self.init.lock();
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
        // Clone registry and managers for OffloadEngine (they will share state via internal Arcs)
        let registry_for_offload = Arc::new(registry.clone());
        let g2_manager_for_offload = g2_manager.clone();
        let g3_manager_for_offload = g3_manager.clone();

        let mut leader_builder = InstanceLeader::builder()
            .nova(self.runtime.nova.clone())
            .registry(registry)
            .g2_manager(g2_manager)
            .workers(
                worker_clients
                    .into_iter()
                    .map(|client| Arc::new(client) as Arc<dyn Worker>)
                    .collect(),
            )
            .with_cached_worker_metadata(worker_metadata);

        // Conditionally add G3 manager
        if let Some(g3_mgr) = g3_manager {
            leader_builder = leader_builder.g3_manager(g3_mgr);
        }

        let leader = leader_builder.build()?;
        tracing::debug!("InstanceLeader built");

        tracing::debug!("Registering handlers on InstanceLeader");
        leader.register_handlers()?;
        tracing::debug!("Handlers registered");

        tracing::debug!("Setting instance leader");
        // Clone for the OnceLock storage, we'll wrap in Arc below for the engine
        self.set_instance_leader(leader.clone())?;
        tracing::debug!("Instance leader set");

        // Wrap in Arc for the engine builder and parallel_worker access
        let leader = Arc::new(leader);

        // Build OffloadEngine with config-driven policies
        tracing::debug!("Building OffloadEngine");
        let offload_config = &self.runtime.config().offload;
        let runtime_handle = self.runtime.tokio();

        // Build G1→G2 policy from config (or defaults if not configured)
        // G1 is externally owned by vLLM (GPU KV cache), accessed via ExternalBlock<G1>
        // Default: Presence filter to prevent duplicate transfers (pending auto-wired)
        let g1_to_g2_config = if offload_config.g1_to_g2.policies.is_empty() {
            tracing::debug!("No G1→G2 policies configured, using default: [Presence]");
            dynamo_kvbm_config::TierOffloadConfig {
                policies: vec![dynamo_kvbm_config::PolicyType::Presence],
                ..Default::default()
            }
        } else {
            offload_config.g1_to_g2.clone()
        };
        let g1_to_g2_pending = Arc::new(PendingTracker::new());
        let g1_to_g2_policy = create_policy_from_config::<G1, G2>(
            &g1_to_g2_config,
            registry_for_offload.clone(),
            Some(g1_to_g2_pending.clone()),
        );
        // Auto-chain G1→G2 completions to downstream tiers (G3 and/or G4)
        let has_downstream_tier =
            g3_manager_for_offload.is_some() || self.runtime.config().object.is_some();
        let g1_to_g2_pipeline = PipelineBuilder::<G1, G2>::new()
            .policy(g1_to_g2_policy)
            .pending_tracker(g1_to_g2_pending)
            .auto_chain(has_downstream_tier)
            .build();

        // Build G2→G3 policy from config (or defaults if not configured)
        // Default: PresenceLfu filter for presence + LFU frequency check (pending auto-wired)
        let g2_to_g3_config = if offload_config.g2_to_g3.policies.is_empty() {
            tracing::debug!("No G2→G3 policies configured, using default: [PresenceLfu]");
            dynamo_kvbm_config::TierOffloadConfig {
                policies: vec![dynamo_kvbm_config::PolicyType::PresenceLfu],
                ..Default::default()
            }
        } else {
            offload_config.g2_to_g3.clone()
        };
        let g2_to_g3_pending = Arc::new(PendingTracker::new());
        let g2_to_g3_policy = create_policy_from_config::<G2, G3>(
            &g2_to_g3_config,
            registry_for_offload.clone(),
            Some(g2_to_g3_pending.clone()),
        );
        let g2_to_g3_pipeline = PipelineBuilder::<G2, G3>::new()
            .policy(g2_to_g3_policy)
            .pending_tracker(g2_to_g3_pending)
            .build();

        let mut engine_builder = OffloadEngine::builder(leader.clone())
            .with_registry(registry_for_offload)
            .with_g2_manager(g2_manager_for_offload)
            .with_runtime(runtime_handle)
            .with_g1_to_g2_pipeline(g1_to_g2_pipeline);

        // Conditionally add G3 pipeline if G3 manager exists
        if let Some(g3_mgr) = g3_manager_for_offload {
            engine_builder = engine_builder
                .with_g3_manager(g3_mgr)
                .with_g2_to_g3_pipeline(g2_to_g3_pipeline);
        }

        // Build G2→G4 object storage pipeline if configured
        // Uses the parallel_worker from leader as ObjectBlockOps to fan out to all workers
        if let Some(object_config) = &self.runtime.config().object {
            tracing::debug!("Object storage configured, creating G2→G4 pipeline");

            // Create lock manager for distributed locking
            let instance_id = self.runtime.nova.instance_id().to_string();
            let lock_manager: Arc<dyn ObjectLockManager> =
                create_lock_manager(object_config, instance_id).await?;

            // Get parallel_worker from leader - it implements ObjectBlockOps and fans out to all workers
            if let Some(parallel_worker) = leader.parallel_worker() {
                // parallel_worker implements ObjectBlockOps, we can cast it to the trait object
                let object_ops: Arc<dyn crate::v2::distributed::object::ObjectBlockOps> =
                    parallel_worker;

                // Create S3 presence checker using the parallel worker
                // When has_blocks is called, it queries all workers who check S3 with their rank-prefixed keys
                let presence_checker = Arc::new(S3PresenceChecker::new(object_ops.clone()));

                // Create presence filter with pending tracker
                let g2_to_g4_pending = Arc::new(PendingTracker::new());
                let presence_filter = Arc::new(
                    ObjectPresenceFilter::<G2>::new(presence_checker)
                        .with_pending_tracker(g2_to_g4_pending.clone()),
                );

                // Build ObjectPipelineConfig
                let g2_to_g4_config = ObjectPipelineBuilder::<G2>::new()
                    .policy(presence_filter)
                    .pending_tracker(g2_to_g4_pending)
                    .lock_manager(lock_manager)
                    .build();

                // Add G2→G4 pipeline to engine
                engine_builder = engine_builder
                    .with_object_ops(object_ops)
                    .with_g2_to_g4_pipeline(g2_to_g4_config);

                tracing::info!("G2→G4 object storage pipeline configured with presence checking");
            } else {
                tracing::warn!(
                    "Object storage configured but no parallel_worker available - G2→G4 pipeline disabled"
                );
            }
        }

        match engine_builder.build() {
            Ok(offload_engine) => {
                tracing::debug!("OffloadEngine built successfully");
                let _ = self.offload_engine.set(offload_engine);
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to build OffloadEngine: {}. Continuing without offload.",
                    e
                );
            }
        }

        tracing::info!("All workers initialized successfully");

        // Refresh handler lists for all workers since they registered new handlers during init
        // This clears the stale cache from the initial handshake (which only had connector handlers)
        tracing::debug!("Acquiring lock to get worker instance IDs for handler refresh");
        let worker_instance_ids = {
            tracing::debug!("Lock acquired for getting worker instance IDs");
            let state = self.init.lock();
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
        let workers = self.init.lock().clone();
        let _ = self.workers.set(Arc::new(workers));

        // Start the control server
        tracing::debug!("Starting control server");
        match super::control::start_control_server(self.clone(), self.runtime.tokio()).await {
            Ok(shutdown_tx) => {
                let _ = self.control_server_shutdown.set(shutdown_tx);
                tracing::info!("Control server started successfully");
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to start control server: {}. Continuing without it.",
                    e
                );
            }
        }

        // Log the instance_id for distributed discovery
        // Operators can use this ID with the /register_leader endpoint on other instances
        tracing::info!(
            instance_id = %self.runtime.nova.instance_id(),
            "KVBM leader instance started - use this ID for register_leader on remote instances"
        );

        Ok(())
    }
}
