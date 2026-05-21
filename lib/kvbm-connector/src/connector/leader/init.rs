// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use super::ConnectorLeader;

use crate::connector::worker::ConnectorWorkerClient;
use crate::{G1, G2, G3, InstanceId};
use kvbm_engine::leader::ConsolidatorParams;
use kvbm_engine::leader::InstanceLeader;
use kvbm_engine::object::{ObjectLockManager, create_lock_manager, create_object_client};
use kvbm_engine::offload::{
    ObjectPipelineBuilder, ObjectPresenceFilter, OffloadEngine, PendingTracker, PipelineBuilder,
    S3PresenceChecker, create_policy_from_config,
};
use kvbm_engine::worker::{LeaderLayoutConfig, VeloWorkerClient, Worker};
use kvbm_logical::blocks::{BlockDuplicationPolicy, BlockRegistry};
use kvbm_logical::events::EventsManager;
use kvbm_logical::manager::{BlockManager, FrequencyTrackingCapacity};

use anyhow::{Context, Result, anyhow, bail};
use velo::{PeerInfo, WorkerAddress};

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
            .messenger()
            .register_peer(PeerInfo::new(instance_id, worker_address))?;

        state.worker_instance_ids.push(instance_id);
        state
            .worker_connector_clients
            .push(ConnectorWorkerClient::new(
                self.runtime.messenger().clone(),
                instance_id,
            ));
        state.worker_transfer_clients.push(VeloWorkerClient::new(
            self.runtime.messenger().clone(),
            instance_id,
        ));

        Ok(())
    }

    /// Lightweight KV-index-only hub registration: declare `Feature::Indexer`
    /// (+ the must-match runtime summary) so the hub reclaims this instance's
    /// index entries on unregister. Holds the [`HubClient`] alive on `self` so
    /// the RAII guard does not fire a premature `DELETE`. Used only when
    /// ConditionalDisagg is *not* effective (otherwise Indexer rides the CD
    /// registration).
    async fn register_indexer_only(
        &self,
        handshake: &super::hub_handshake::HubHandshake,
    ) -> Result<()> {
        let velo = self
            .runtime
            .velo()
            .ok_or_else(|| anyhow!("indexer hub registration requires a Velo runtime"))?;
        let hub = super::disagg::build_hub_client(&handshake.url)?;
        // Install hub velo handlers (heartbeat) so the hub's liveness probe
        // doesn't unregister us — which would prematurely sweep our index.
        hub.register_handlers_messenger(velo.messenger())
            .context("installing hub velo handlers for indexer registration")?;
        let max_seq_len = self.runtime.config().max_seq_len;
        hub.register_instance_with_features_and_runtime(
            velo.peer_info(),
            vec![kvbm_hub::Feature::Indexer(kvbm_hub::IndexerFeatureConfig {
                max_seq_len,
            })],
            handshake.runtime_summary.clone(),
        )
        .await
        .context("registering Feature::Indexer with kvbm-hub")?;
        self.set_indexer_hub_client(hub)?;
        tracing::info!(url = %handshake.url, ?max_seq_len, "indexer participation registered with hub");
        Ok(())
    }

    /// Build the mandatory P2P `layout_compat` payload from the leader's rank-0
    /// worker metadata. Shared by the CD and standalone-P2P registration paths.
    ///
    /// SPMD leaders carry identical per-worker shape (`validate_remote_metadata`
    /// enforces it), so a rank-0 sample suffices. The cached metadata is raw
    /// (`parallelism = None`); pass a freshly-built [`ParallelismTemplate`] so
    /// universal mode derives its canonical aggregate without pre-stamping.
    /// This must mirror the `leader_builder.parallelism_template(...)` site —
    /// both derive from the same `(reference_config, parallelism, num_workers)`.
    fn build_layout_compat_payload(
        &self,
        reference_config: &kvbm_physical::layout::LayoutConfig,
        num_workers: usize,
    ) -> Result<kvbm_hub::protocol::LayoutCompatPayload> {
        let block_layout_mode = self.runtime.config().block_layout;
        let template_for_payload = if reference_config.num_heads.is_some() {
            Some(
                kvbm_engine::leader::parallelism::ParallelismTemplate::from_layout_config(
                    reference_config,
                    self.runtime.config().cache.parallelism,
                    num_workers,
                )
                .context("building ParallelismTemplate for hub registration payload")?,
            )
        } else {
            None
        };
        let worker_metadata_for_payload = {
            let state = self.init.lock();
            state.worker_metadata.first().cloned()
        }
        .ok_or_else(|| {
            anyhow!(
                "cannot build layout_compat payload for hub registration: \
                 worker_metadata is empty (SPMD bring-up must populate \
                 state.worker_metadata before registration — call-ordering bug)."
            )
        })?;
        kvbm_engine::leader::layout_compat::build_layout_compat_payload_with_template(
            block_layout_mode,
            &worker_metadata_for_payload,
            template_for_payload.as_ref(),
        )
        .context("building layout_compat payload for hub registration")
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
            .compute_num_blocks(bytes_per_block);

        let disk_block_count = self
            .runtime
            .config()
            .cache
            .disk
            .as_ref()
            .and_then(|dc| dc.compute_num_blocks(bytes_per_block));

        // Mirror v1 sanity_check: at least one cache tier must produce a
        // non-zero block count, otherwise the leader has nothing to offload to.
        // Fail loudly rather than silently falling back to zero host blocks.
        let host_ok = host_block_count.is_some_and(|n| n > 0);
        let disk_ok = disk_block_count.is_some_and(|n| n > 0);
        if !host_ok && !disk_ok {
            bail!(
                "KVBM Configuration Error: At least one cache tier must be configured.\n\
                \n\
                Configure CPU cache (G2) for CPU memory offloading:\n\
                • DYN_KVBM_CPU_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_CPU_CACHE_GB=4)\n\
                • DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>  (e.g., DYN_KVBM_CPU_CACHE_OVERRIDE_NUM_BLOCKS=1000)\n\
                \n\
                OR configure disk cache (G3) for direct GPU->Disk offloading:\n\
                • DYN_KVBM_DISK_CACHE_GB=<size_in_gb>     (e.g., DYN_KVBM_DISK_CACHE_GB=8)\n\
                • DYN_KVBM_DISK_CACHE_OVERRIDE_NUM_BLOCKS=<num_blocks>\n\
                \n\
                Note: If only disk cache is configured, KVBM will offload directly from GPU (G1) to Disk (G3), bypassing CPU memory (G2)."
            );
        }

        let host_block_count = host_block_count.unwrap_or(0);

        // Host-bypass mode: when disk is configured and host is not, we serve
        // disk hits to GPU directly via GDS instead of staging through G2.
        // InstanceLeader still requires a G2 BlockManager, but no G1→G2 /
        // G2→G3 pipelines are wired and no onboarding routes through G2 —
        // so we build it with a sentinel block_count of 1 (BlockManager
        // validation rejects 0). It allocates no physical memory; it's a
        // logical-only placeholder that will never see registered blocks.
        let bypass_host = self.runtime.config().cache.bypass_host_cache();
        let g2_manager_block_count = if bypass_host {
            host_block_count.max(1)
        } else {
            host_block_count
        };

        tracing::info!(
            host_block_count,
            ?disk_block_count,
            bytes_per_block,
            bypass_host,
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
                    parallelism: self.runtime.config().cache.parallelism,
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
                .map_err(|e| {
                    tracing::error!(
                        worker_idx = i,
                        error = %e,
                        error_chain = ?e,
                        "Worker initialization failed"
                    );
                    e
                })
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

            // Configure layout handles for each VeloWorkerClient from their metadata
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

        // Hub handshake. `leader.hub` is the sole way the connector reaches a
        // hub; absent ⇒ no hub features (normal hub-less work). When present we
        // pull `GET /v1/config`, resolve the effective feature set, and learn
        // the KV-index ZMQ endpoint. (See `hub_handshake` for auto best-effort
        // vs explicit hard-fail semantics.)
        let cfg = self.runtime.config();
        let handshake: Option<super::hub_handshake::HubHandshake> = match cfg.hub.as_ref() {
            Some(hub) => Some(
                super::hub_handshake::resolve(
                    hub,
                    reference_config.page_size,
                    cfg.block_layout,
                    cfg.disagg.as_ref(),
                )
                .await
                .context("kvbm-hub handshake")?,
            ),
            None => None,
        };
        let indexer_endpoint: Option<String> = handshake
            .as_ref()
            .and_then(|h| h.indexer_zmq_endpoint.clone());

        // Create an EventsManager when either the consolidator or the KV-index
        // publisher needs block registration events. The same Arc is wired into
        // the BlockRegistry (so events are emitted on register/evict) and into
        // every subscriber (ConsolidatorParams, KV-index publisher).
        //
        // NOTE: the builder default policy is `AllEventsPolicy` (every block
        // position emits). We intentionally do not thread `EventsConfig.policy`
        // through here — the KV index wants full positional coverage, and the
        // consolidator path has always relied on the default.
        let events_manager: Option<std::sync::Arc<EventsManager>> =
            (self.consolidator_endpoints.is_some() || indexer_endpoint.is_some()).then(|| {
                tracing::debug!("Creating EventsManager");
                std::sync::Arc::new(EventsManager::builder().build())
            });

        // Wire the KV-index publisher onto the EventsManager subscription. The
        // publisher stamps this worker's velo instance id (as u128) so the
        // hub's query results map back to a discoverable peer.
        if let (Some(endpoint), Some(em)) = (&indexer_endpoint, events_manager.as_ref()) {
            match super::hub_indexer::ZmqHubPublisher::connect(endpoint) {
                Ok(zmq_pub) => {
                    let instance_id = self.runtime.messenger().instance_id().as_u128();
                    match kvbm_logical::events::KvbmCacheEventsPublisher::builder()
                        .instance_id(instance_id)
                        .event_stream(em.subscribe())
                        .publisher(std::sync::Arc::new(zmq_pub))
                        .subject(super::hub_indexer::SUBJECT)
                        .build()
                    {
                        Ok(publisher) => {
                            self.set_indexer_publisher(publisher)?;
                            tracing::info!(endpoint, instance_id, "indexer publisher wired");
                        }
                        Err(e) => {
                            tracing::warn!(error = %e, "indexer publisher build failed; skipping")
                        }
                    }
                }
                Err(e) => tracing::warn!(error = %e, "indexer PUB connect failed; skipping"),
            }
        }

        // KV-index-only hub registration — fires only when Indexer is the
        // *sole* effective hub feature. When P2P or ConditionalDisagg is also
        // effective, `wire_p2p` folds Indexer into that single
        // `POST /v1/instances` instead (avoiding a double registration).
        // Declaring `Feature::Indexer` lets the hub's `on_unregister` sweep
        // reclaim this instance's index entries.
        if let Some(h) = handshake.as_ref()
            && h.has(kvbm_hub::FeatureKey::Indexer)
            && !h.has(kvbm_hub::FeatureKey::P2P)
            && !h.has(kvbm_hub::FeatureKey::ConditionalDisagg)
        {
            self.register_indexer_only(h).await?;
        }

        tracing::debug!("Creating block registry");
        let mut registry_builder = BlockRegistry::builder()
            .frequency_tracker(FrequencyTrackingCapacity::Medium.create_tracker());
        if let Some(em) = events_manager.clone() {
            registry_builder = registry_builder.event_manager(em);
        }
        let registry = registry_builder.build();
        tracing::debug!("Block registry created");

        tracing::debug!(
            block_count = g2_manager_block_count,
            page_size = reference_config.page_size,
            bypass_host,
            "Building G2 manager"
        );
        let logical_metrics = self.runtime.observability().logical_aggregator();
        let g2_manager = Arc::new(
            BlockManager::<G2>::builder()
                .block_count(g2_manager_block_count)
                .block_size(reference_config.page_size)
                .registry(registry.clone())
                .with_lineage_backend()
                .aggregator(logical_metrics.clone())
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
                    .with_lineage_backend()
                    .aggregator(logical_metrics.clone())
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

        let num_workers = worker_clients.len();
        let mut leader_builder = InstanceLeader::builder()
            .messenger(self.runtime.messenger().clone())
            // Plumb the runtime's Prometheus registry through so the `metrics`
            // control module can answer `snapshot` from the same source the
            // production `start_metrics_server` exposes. Without this, a
            // leader built with `control.metrics = true` would silently log
            // a warning at register time and the module would never appear.
            .observability(self.runtime.observability().clone());
        // Plumb the full Velo handle when available — `core/register_leader`
        // uses `velo.discover_and_register_peer` (both messenger + streaming
        // registries). Without this the streaming registry stays empty and
        // `attach_anchor` fails with "TCP streaming: peer <id> not registered".
        if let Some(velo) = self.runtime.velo() {
            leader_builder = leader_builder.velo(velo.clone());
        }
        leader_builder = leader_builder
            // Cross-leader block-layout compat policy
            // (Operational by default; Universal opts-in to canonical-only
            // matching across permutations / TP / PP).
            .block_layout_mode(self.runtime.config().block_layout)
            .registry(registry)
            .g2_manager(g2_manager)
            .bypass_host(bypass_host)
            .workers(
                worker_clients
                    .into_iter()
                    .map(|client| Arc::new(client) as Arc<dyn Worker>)
                    .collect(),
            )
            .with_cached_worker_metadata(worker_metadata);

        // Stamp the disaggregation role onto the leader so `describe()` can
        // surface it. Standalone leaders (no disagg config) carry `None`.
        if let Some(disagg_cfg) = self.runtime.config().disagg.as_ref() {
            leader_builder = leader_builder.role(disagg_cfg.role);
        }

        // Stamp ParallelismDescriptors on per-worker metadata exported to
        // peer leaders, so cross-parallelism dispatch is informed (AB-1a).
        // Skip if num_heads is absent — leader falls back to the symmetric
        // path (descriptor remains None on the wire).
        if reference_config.num_heads.is_some() {
            let template =
                kvbm_engine::leader::parallelism::ParallelismTemplate::from_layout_config(
                    reference_config,
                    self.runtime.config().cache.parallelism,
                    num_workers,
                )?;
            leader_builder = leader_builder.parallelism_template(template);
        }

        // Conditionally add G3 manager
        if let Some(g3_mgr) = g3_manager {
            leader_builder = leader_builder.g3_manager(g3_mgr);
        }

        // Add object_client for G4 search (leader calls has_blocks on S3 directly)
        // Uses rank=None so keys are not prefixed - allows querying all worker-written blocks
        if let Some(object_config) = &self.runtime.config().object {
            tracing::debug!("Creating object client for G4 search (no rank prefix)");
            let object_client = create_object_client(object_config, None).await?;
            leader_builder = leader_builder.object_client(object_client);
        }

        let leader = leader_builder.build()?;
        tracing::debug!("InstanceLeader built");

        tracing::debug!("Registering handlers on InstanceLeader");
        leader.register_handlers()?;
        tracing::debug!("Handlers registered");

        // Start the in-process consolidator if endpoints were provided.
        // Hard-fail: a consolidator config error is a mis-configuration that
        // must surface immediately rather than silently degrade.
        if let Some(endpoints) = self.consolidator_endpoints.as_ref() {
            let em = events_manager
                .clone()
                .expect("events_manager must be Some when consolidator_endpoints is Some");
            let params = ConsolidatorParams {
                vllm_zmq_endpoint: endpoints.vllm_zmq_endpoint.clone(),
                egress_endpoint: endpoints.egress_endpoint.clone(),
                engine_source: endpoints.engine_source,
                events_manager: em,
            };
            tracing::info!(
                egress_endpoint = %endpoints.egress_endpoint,
                has_vllm_zmq = endpoints.vllm_zmq_endpoint.is_some(),
                "Starting in-process consolidator"
            );
            leader
                .with_consolidator(params)
                .await
                .context("failed to start in-process consolidator")?;
            tracing::info!("In-process consolidator started");
        }

        tracing::debug!("Setting instance leader");
        // Clone for the OnceLock storage, we'll wrap in Arc below for the engine
        self.set_instance_leader(leader.clone())?;
        tracing::debug!("Instance leader set");

        // Wrap in Arc for the engine builder and parallel_worker access
        let leader = Arc::new(leader);

        // Register the public leader control plane. `core` + `transfer` are
        // always on; `dev` / `test` are opt-in via `control.{dev,test}`. The
        // `transfer` module reads the disagg `SessionFactory` lazily from a
        // cell populated further below, once CD wiring builds the factory.
        let control_cfg = &self.runtime.config().control;
        let control_plane = leader
            .register_control_plane(control_cfg.dev, control_cfg.metrics)
            .context("registering leader control plane")?;
        tracing::debug!(
            dev = control_cfg.dev,
            metrics = control_cfg.metrics,
            "Leader control plane registered"
        );

        // Surface the enabled module set on the leader so `describe()` can
        // report it without having to re-traverse the control plane object.
        leader.set_modules(control_plane.enabled_modules().to_vec());

        // Build OffloadEngine with config-driven policies
        tracing::debug!("Building OffloadEngine");
        let offload_config = &self.runtime.config().offload;
        let runtime_handle = self.runtime.tokio();

        let mut engine_builder = OffloadEngine::builder(leader.clone())
            .with_registry(registry_for_offload.clone())
            .with_g2_manager(g2_manager_for_offload)
            .with_runtime(runtime_handle);

        if bypass_host {
            // Host-bypass mode: single G1→G3 pipeline, no G2 staging.
            // Default policy is Presence (symmetric with the non-bypass
            // G1→G2 default). G3 manager must exist — guaranteed by the
            // earlier disk_ok check.
            let g1_to_g3_config = if offload_config.g1_to_g3.policies.is_empty() {
                tracing::debug!("No G1→G3 policies configured, using default: [Presence]");
                kvbm_config::TierOffloadConfig {
                    policies: vec![kvbm_config::PolicyType::Presence],
                    ..Default::default()
                }
            } else {
                offload_config.g1_to_g3.clone()
            };
            let g1_to_g3_pending = Arc::new(PendingTracker::new());
            let g1_to_g3_policy = create_policy_from_config::<G1, G3>(
                &g1_to_g3_config,
                registry_for_offload.clone(),
                Some(g1_to_g3_pending.clone()),
            );
            let g1_to_g3_pipeline = PipelineBuilder::<G1, G3>::new()
                .policy(g1_to_g3_policy)
                .pending_tracker(g1_to_g3_pending)
                .build();

            let g3_mgr = g3_manager_for_offload.clone().ok_or_else(|| {
                anyhow::anyhow!("Host-bypass mode requires a configured G3 (disk) cache; got none")
            })?;
            engine_builder = engine_builder
                .with_g3_manager(g3_mgr)
                .with_g1_to_g3_pipeline(g1_to_g3_pipeline);
        } else {
            // Standard mode: G1→G2 (with auto-chain) and G2→G3.
            //
            // Build G1→G2 policy from config (or defaults if not configured).
            // G1 is externally owned by vLLM (GPU KV cache), accessed via ExternalBlock<G1>.
            // Default: Presence filter to prevent duplicate transfers (pending auto-wired).
            let g1_to_g2_config = if offload_config.g1_to_g2.policies.is_empty() {
                tracing::debug!("No G1→G2 policies configured, using default: [Presence]");
                kvbm_config::TierOffloadConfig {
                    policies: vec![kvbm_config::PolicyType::Presence],
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

            // Build G2→G3 policy from config (or defaults if not configured).
            // Default: Presence — symmetric with G1→G2. Offload any block not already
            // on disk and not already in flight; let G3's eviction backend handle
            // cold-block churn under pressure. Opt into LFU-on-admission for
            // workloads where disk write amplification matters by setting
            // KVBM_OFFLOAD_G2_TO_G3_POLICIES='["presence_lfu"]'.
            let g2_to_g3_config = if offload_config.g2_to_g3.policies.is_empty() {
                tracing::debug!("No G2→G3 policies configured, using default: [Presence]");
                kvbm_config::TierOffloadConfig {
                    policies: vec![kvbm_config::PolicyType::Presence],
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

            engine_builder = engine_builder.with_g1_to_g2_pipeline(g1_to_g2_pipeline);

            // Conditionally add G3 pipeline if G3 manager exists
            if let Some(g3_mgr) = g3_manager_for_offload {
                engine_builder = engine_builder
                    .with_g3_manager(g3_mgr)
                    .with_g2_to_g3_pipeline(g2_to_g3_pipeline);
            }
        }

        // Build G2→G4 object storage pipeline if configured
        // Uses the parallel_worker from leader as ObjectBlockOps to fan out to all workers
        if let Some(object_config) = &self.runtime.config().object {
            tracing::debug!("Object storage configured, creating G2→G4 pipeline");

            // Create lock manager for distributed locking
            let instance_id = self.runtime.messenger().instance_id().to_string();
            let lock_manager: Arc<dyn ObjectLockManager> =
                create_lock_manager(object_config, instance_id).await?;

            // Get parallel_worker from leader - it implements ObjectBlockOps and fans out to all workers
            if let Some(parallel_worker) = leader.parallel_worker() {
                // parallel_worker implements ObjectBlockOps, we can cast it to the trait object
                let object_ops: Arc<dyn kvbm_engine::object::ObjectBlockOps> = parallel_worker;

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
                .messenger()
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

        // The connector-leader control plane lives on the engine — it was
        // registered above via `leader.register_control_plane(...)`. The
        // hub's HTTP→velo proxy reaches it through the same handler-name
        // strings, so nothing on the hub side changes.

        // Log the instance_id for distributed discovery
        // Operators can use this ID with the /register_leader endpoint on other instances
        tracing::info!(
            instance_id = %self.runtime.messenger().instance_id(),
            "KVBM leader instance started - use this ID for register_leader on remote instances"
        );

        // Register with kvbm-hub for conditional disaggregation when the hub
        // handshake made ConditionalDisagg effective. Pre-flight guarantees a
        // `disagg` role is configured.
        if let Some(handshake) = handshake
            .as_ref()
            .filter(|h| h.has(kvbm_hub::FeatureKey::ConditionalDisagg))
        {
            let disagg_cfg = self
                .runtime
                .config()
                .disagg
                .clone()
                .expect("hub handshake pre-flight guarantees a disagg role for disagg");
            tracing::info!(
                role = ?disagg_cfg.role,
                hub_url = %handshake.url,
                "Registering leader with kvbm-hub for conditional disaggregation"
            );

            use super::disagg::{
                AlwaysRemote, ConditionalDisaggCoordinator, ConditionalDisaggLeader,
                ConditionalDisaggPolicy, ConnectorLeaderApi, ConnectorLeaderShim, CoordinatorParts,
                DecodeDisaggLeader, EngineP2pBlockTransport, HubRemotePrefillQueue, HubWiring,
                InnerLeaderShim, InnerLeaderWorkerHook, P2pBlockTransport, P2pWorkerHook,
                PeerResolver, PrefillDisaggLeader, RemotePrefillQueue,
            };
            use kvbm_config::DisaggregationRole;

            let layout_compat_payload =
                self.build_layout_compat_payload(reference_config, num_workers)?;

            // Register P2P (+ ConditionalDisagg, + Indexer when effective) and
            // build the shared P2P transfer foundation (peer resolver, session
            // factory, describe-push). CD layers its role-specific flow on top.
            let cd_role = match disagg_cfg.role {
                DisaggregationRole::Prefill => kvbm_hub::ConditionalDisaggRole::Prefill,
                DisaggregationRole::Decode => kvbm_hub::ConditionalDisaggRole::Decode,
            };
            let super::p2p::wire::P2pFoundation {
                hub,
                hub_velo_id,
                peer_resolver,
                session_factory,
                velo_runtime,
                tokio_handle,
            } = super::p2p::wire::wire_p2p(
                &self,
                &leader,
                handshake,
                layout_compat_payload,
                vec![kvbm_hub::Feature::ConditionalDisagg(
                    kvbm_hub::ConditionalDisaggConfig { role: cd_role },
                )],
            )
            .await
            .context("disagg P2P foundation wiring failed")?;

            // CD client around the shared hub (wire_p2p already registered);
            // seed the hub velo id for the prefill queue.
            let client = kvbm_hub::ConditionalDisaggClient::with_messenger(
                Arc::clone(&hub),
                velo_runtime.messenger().clone(),
                cd_role,
            );
            client.set_hub_velo_id(hub_velo_id);
            let _ = self.disagg_client.set(Arc::clone(&client));

            // Common building blocks used by both roles.
            let inner_shim: Arc<dyn InnerLeaderShim> = ConnectorLeaderShim::new(self.clone());
            let worker_hook: Arc<dyn P2pWorkerHook> = InnerLeaderWorkerHook::new(self.clone());
            let transport: Arc<dyn P2pBlockTransport> =
                EngineP2pBlockTransport::new(Arc::clone(&leader));

            // Construct the role-specific concrete leader (and, for
            // prefill, its coordinator).  We hold concrete `Arc<...>`
            // handles rather than `Arc<dyn ConnectorLeaderApi>` so
            // the `ConditionalDisaggLeader` dispatcher can wrap them.
            enum RoleSpecific {
                Decode(Arc<DecodeDisaggLeader>),
                Prefill {
                    leader: Arc<PrefillDisaggLeader>,
                    coordinator: Arc<ConditionalDisaggCoordinator>,
                },
            }

            let role_specific: RoleSpecific = match disagg_cfg.role {
                DisaggregationRole::Decode => {
                    // Production policy: AlwaysRemote (every CD-eligible
                    // request goes remote). Threshold-based policies are
                    // a Phase B.4 follow-up; the local-only path is
                    // exercised by running with `disagg = None`, which
                    // bypasses this whole block.
                    let policy: Arc<dyn ConditionalDisaggPolicy> = Arc::new(AlwaysRemote);
                    let queue: Arc<dyn RemotePrefillQueue> =
                        HubRemotePrefillQueue::new(Arc::clone(&client));
                    let coord = ConditionalDisaggCoordinator::new_with_decode(
                        CoordinatorParts {
                            inner: Arc::clone(&inner_shim),
                            transport: Arc::clone(&transport),
                            worker_hook: Arc::clone(&worker_hook),
                            session_factory: Arc::clone(&session_factory),
                            peer_resolver: Arc::clone(&peer_resolver) as Arc<dyn PeerResolver>,
                            runtime: tokio_handle.clone(),
                        },
                        policy,
                        queue,
                    );
                    let decode = DecodeDisaggLeader::from_parts(
                        Arc::clone(&inner_shim),
                        &disagg_cfg,
                        coord,
                        Arc::clone(&transport),
                        Arc::clone(&worker_hook),
                        tokio_handle.clone(),
                        HubWiring {
                            hub: Some(Arc::clone(&hub)),
                            client: Some(Arc::clone(&client)),
                            hub_velo_id,
                        },
                    );
                    RoleSpecific::Decode(decode)
                }
                DisaggregationRole::Prefill => {
                    let coord = ConditionalDisaggCoordinator::new(CoordinatorParts {
                        inner: Arc::clone(&inner_shim),
                        transport: Arc::clone(&transport),
                        worker_hook: Arc::clone(&worker_hook),
                        session_factory: Arc::clone(&session_factory),
                        peer_resolver: Arc::clone(&peer_resolver) as Arc<dyn PeerResolver>,
                        runtime: tokio_handle.clone(),
                    });

                    // Wire the offload-pipeline observer once — captures
                    // G1→G2 register events so output blocks flow back
                    // to the session via commit + make_available.
                    if let Some(engine) = self.offload_engine.get() {
                        engine
                            .add_g1_to_g2_register_observer(coord.observer_callback())
                            .context("registering CD output observer on G1→G2 pipeline")?;
                    } else {
                        tracing::warn!(
                            "CD prefill role configured but OffloadEngine missing — \
                             G2 output capture disabled"
                        );
                    }

                    let leader = PrefillDisaggLeader::from_parts(
                        Arc::clone(&inner_shim),
                        Arc::clone(&coord),
                        Arc::clone(&worker_hook),
                    );
                    RoleSpecific::Prefill {
                        leader,
                        coordinator: coord,
                    }
                }
            };

            // Build the per-request-dispatching ConditionalDisaggLeader
            // wrapping whichever role-specific flow was constructed above.
            // Today's hub plumbing registers each instance under exactly
            // one role, so only one flow is wired per instance — but the
            // leader is forward-compatible with both flows simultaneously.
            let mut builder = ConditionalDisaggLeader::builder(Arc::clone(&inner_shim))
                .with_hub(Arc::clone(&hub))
                .with_client(Arc::clone(&client));
            if let Some(id) = hub_velo_id {
                builder = builder.with_hub_velo_id(id);
            }
            let builder = match role_specific {
                RoleSpecific::Decode(decode) => builder.with_decode(decode),
                RoleSpecific::Prefill {
                    leader,
                    coordinator,
                } => builder.with_prefill(leader, coordinator),
            };
            let dispatcher: Arc<dyn ConnectorLeaderApi> = builder
                .build()
                .context("build ConditionalDisaggLeader from role-specific flow")?;
            tracing::info!(
                role = ?disagg_cfg.role,
                "Conditional-disagg dispatcher installed (ConditionalDisaggLeader)"
            );
            self.set_cd_api(dispatcher)
                .context("install CD dispatcher")?;
            // (describe-push is handled inside wire_p2p.)
        } else if let Some(handshake) = handshake
            .as_ref()
            .filter(|h| h.has(kvbm_hub::FeatureKey::P2P))
        {
            // Standalone P2P (no CD): register Feature::P2P + build the transfer
            // foundation so this instance is a hub-discoverable,
            // remote-controllable block-copy peer. Indexer (when effective) is
            // folded into the same registration by wire_p2p.
            let layout_compat_payload =
                self.build_layout_compat_payload(reference_config, num_workers)?;
            let foundation = super::p2p::wire::wire_p2p(
                &self,
                &leader,
                handshake,
                layout_compat_payload,
                vec![],
            )
            .await
            .context("standalone P2P foundation wiring failed")?;
            self.set_p2p_hub_client(foundation.hub)?;
            tracing::info!(url = %handshake.url, "standalone P2P participation registered with hub");
        }

        Ok(())
    }
}
