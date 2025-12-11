// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Leader implementation for coordinating distributed worker cohorts.
//!
//! The leader is responsible for:
//! - Accepting workers into the cohort with rank validation
//! - Broadcasting layout creation requests
//! - Coordinating transfer execution across workers
//! - Collecting layout metadata and descriptors
//! - Managing named synchronization barriers

use anyhow::Result;
use dynamo_am::api::client::{ActiveMessageClient, WorkerAddress};
use dynamo_am::runtime::{cohort::LeaderWorkerCohort, host::ActiveMessageServer};
use dynamo_am::zmq::ZmqServerConfig;
use dynamo_am::{
    ActiveMessageManager, handler_impls::typed_unary_handler_async_with_tracker,
    runtime::dispatcher::ControlMessage,
};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tracing::{debug, warn};

use super::messages::*;
use crate::v2::physical::layout::{LayoutConfig, LayoutDescriptor};
use crate::v2::physical::manager::{LayoutHandle, SerializedLayout};

/// Leader that coordinates a distributed cohort of workers.
///
/// The leader manages the lifecycle of the worker cohort, from initial
/// formation through layout creation, transfer coordination, and metadata collection.
pub struct Leader {
    server: Arc<ActiveMessageServer>,
    cohort: Arc<LeaderWorkerCohort>,
}

impl Clone for Leader {
    fn clone(&self) -> Self {
        Self {
            server: self.server.clone(),
            cohort: self.cohort.clone(),
        }
    }
}

impl Leader {
    /// Create a new leader with the given cohort configuration.
    ///
    /// # Arguments
    /// * `cohort_type` - The type of cohort (e.g., FixedSize(N))
    /// * `cancel_token` - Cancellation token for shutdown coordination
    ///
    /// # Returns
    /// A new Leader instance and its WorkerAddress
    pub async fn new(
        cohort_type: dynamo_am::runtime::cohort::CohortType,
        cancel_token: CancellationToken,
    ) -> Result<(Self, WorkerAddress)> {
        // Create the ActiveMessage server with ZMQ IPC transport (auto-generated unique path)
        let zmq_config = ZmqServerConfig::builder().ipc_endpoint("auto").build()?;

        let server = ActiveMessageServer::builder()
            .enable_zmq_with_config(zmq_config)
            .build(cancel_token)
            .await?;

        let client = server.client();
        let peer_info = server.peer_info().await;

        // Create the cohort
        let cohort = Arc::new(LeaderWorkerCohort::new(
            client.clone() as Arc<dyn ActiveMessageClient>,
            cohort_type,
        ));

        let leader = Self {
            server: Arc::new(server),
            cohort,
        };

        // Register the create_cohort handler
        leader.register_create_cohort_handler().await?;

        // Register the barrier handler
        leader.register_barrier_handler().await?;

        debug!(
            "Leader created at {}",
            peer_info.address.primary_endpoint().unwrap_or("unknown")
        );

        Ok((leader, peer_info.address))
    }

    /// Register the `kvbm.cohort.leader.create_cohort` handler.
    ///
    /// This handler processes join requests from workers.
    async fn register_create_cohort_handler(&self) -> Result<()> {
        let cohort = self.cohort.clone();
        let task_tracker = tokio_util::task::TaskTracker::new();

        let handler = typed_unary_handler_async_with_tracker(
            "kvbm.cohort.leader.create_cohort".to_string(),
            move |ctx: dynamo_am::runtime::handler_impls::TypedContext<CreateCohortRequest>| {
                let cohort = cohort.clone();
                async move {
                    debug!(
                        "Received join request from worker {} (rank: {:?}, world_size: {})",
                        ctx.sender_id, ctx.input.rank, ctx.input.world_size
                    );

                    // Validate and add worker to cohort
                    match cohort.add_worker(ctx.sender_id, ctx.input.rank).await {
                        Ok(position) => {
                            debug!(
                                "Worker {} joined cohort at position {} (rank: {:?})",
                                ctx.sender_id, position, ctx.input.rank
                            );
                            Ok(CreateCohortResponse {
                                accepted: true,
                                position: Some(position),
                                reason: None,
                            })
                        }
                        Err(e) => {
                            warn!("Worker {} failed to join cohort: {}", ctx.sender_id, e);
                            Ok(CreateCohortResponse {
                                accepted: false,
                                position: None,
                                reason: Some(e.to_string()),
                            })
                        }
                    }
                }
            },
            task_tracker,
        );

        // Register with dispatcher
        self.server
            .control_tx()
            .send(ControlMessage::Register {
                name: "kvbm.cohort.leader.create_cohort".to_string(),
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register create_cohort handler: {}", e))?;

        debug!("Registered kvbm.cohort.leader.create_cohort handler");
        Ok(())
    }

    /// Register the `kvbm.cohort.barrier` handler.
    ///
    /// This handler processes barrier arrival notifications from workers.
    /// Workers send this message to signal they've reached a synchronization point.
    async fn register_barrier_handler(&self) -> Result<()> {
        let cohort = self.cohort.clone();
        let task_tracker = tokio_util::task::TaskTracker::new();

        let handler = typed_unary_handler_async_with_tracker(
            "kvbm.cohort.barrier".to_string(),
            move |ctx: dynamo_am::runtime::handler_impls::TypedContext<BarrierReachedRequest>| {
                let cohort = cohort.clone();
                async move {
                    debug!(
                        "Worker {} reached barrier '{}'",
                        ctx.sender_id, ctx.input.barrier_name
                    );

                    // Record the worker's arrival at this barrier
                    cohort
                        .record_barrier_arrival(&ctx.input.barrier_name, ctx.sender_id)
                        .await;

                    // Send acknowledgment
                    Ok(BarrierReachedResponse { acknowledged: true })
                }
            },
            task_tracker,
        );

        // Register with dispatcher
        self.server
            .control_tx()
            .send(ControlMessage::Register {
                name: "kvbm.cohort.barrier".to_string(),
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register barrier handler: {}", e))?;

        debug!("Registered kvbm.cohort.barrier handler");
        Ok(())
    }

    /// Wait for the cohort to be complete (all expected workers joined).
    ///
    /// This method blocks until the cohort reaches its expected size and
    /// all rank validations pass (if ranks are used).
    pub async fn await_cohort_complete(&self) -> Result<()> {
        debug!("Waiting for cohort to complete...");

        // Poll until cohort is complete
        while !self.cohort.is_cohort_complete().await {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }

        let worker_count = self.cohort.worker_count().await;
        debug!("Cohort complete with {} workers", worker_count);

        Ok(())
    }

    /// Execute a transfer on a specific worker rank.
    ///
    /// This sends a transfer request to a single worker, instructing it to
    /// execute a local transfer using its TransportManager. The method validates
    /// that both handles belong to the specified rank before sending.
    ///
    /// This mirrors the TransportManager::execute_transfer API but operates on
    /// a specific worker rank in the cohort.
    ///
    /// # Arguments
    /// * `rank` - Worker rank to execute the transfer on
    /// * `src_handle` - Source layout handle (must belong to this rank)
    /// * `src_blocks` - Source block IDs to transfer
    /// * `dst_handle` - Destination layout handle (must belong to this rank)
    /// * `dst_blocks` - Destination block IDs to transfer
    /// * `options` - Transfer options (layer range, notifications, etc.)
    ///
    /// # Returns
    /// Ok(()) if the transfer completed successfully, Err otherwise
    ///
    /// # Errors
    /// Returns an error if:
    /// - Source handle doesn't belong to the specified rank
    /// - Destination handle doesn't belong to the specified rank
    /// - Worker fails to execute or complete the transfer
    pub async fn execute_transfer_on_rank(
        &self,
        rank: usize,
        src_handle: LayoutHandle,
        src_blocks: &[usize],
        dst_handle: LayoutHandle,
        dst_blocks: &[usize],
        options: TransferOptionsWire,
    ) -> Result<()> {
        // Validate handles belong to this rank
        let rank_u64 = rank as u64;
        if src_handle.worker_id() != rank_u64 {
            anyhow::bail!(
                "Source handle worker_id ({}) doesn't match rank ({})",
                src_handle.worker_id(),
                rank
            );
        }
        if dst_handle.worker_id() != rank_u64 {
            anyhow::bail!(
                "Destination handle worker_id ({}) doesn't match rank ({})",
                dst_handle.worker_id(),
                rank
            );
        }

        // Get worker instance ID for this rank
        let workers_by_rank = self.cohort.get_workers_by_rank().await;
        let (_worker_rank, instance_id) = workers_by_rank
            .get(rank)
            .ok_or_else(|| anyhow::anyhow!("No worker at rank {}", rank))?;

        debug!(
            "Executing transfer on rank {}: src={}, dst={}, {} blocks",
            rank,
            src_handle,
            dst_handle,
            src_blocks.len()
        );

        // Build request
        let request = ExecuteTransferRequest {
            src_handle,
            dst_handle,
            src_blocks: src_blocks.to_vec(),
            dst_blocks: dst_blocks.to_vec(),
            options,
        };

        // Send request and await response
        let client = self.server.client();
        let response = client
            .typed_unary::<ExecuteTransferResponse>("kvbm.cohort.worker.execute_transfer")?
            .payload(request)?
            .instance(*instance_id)
            .send()
            .await?;

        // Check result
        match response.result {
            Ok(()) => {
                debug!("Rank {} transfer completed successfully", rank);
                Ok(())
            }
            Err(error) => Err(anyhow::anyhow!("Rank {} transfer failed: {}", rank, error)),
        }
    }

    /// Broadcast execute transfer request to all workers.
    ///
    /// This instructs all workers to execute the same local transfer operation
    /// in parallel, using their worker-specific layout handles. Each worker
    /// performs the transfer locally and waits for completion before responding.
    ///
    /// This is built on top of execute_transfer_on_rank(), calling it in parallel
    /// for all ranks with their respective handles.
    ///
    /// # Arguments
    /// * `src_handles` - Source layout handles, one per worker in rank order
    /// * `dst_handles` - Destination layout handles, one per worker in rank order
    /// * `src_blocks` - Source block IDs to transfer (same for all workers)
    /// * `dst_blocks` - Destination block IDs to transfer (same for all workers)
    /// * `options` - Transfer options (layer range, notifications, etc.)
    ///
    /// # Returns
    /// Ok(()) if all workers successfully completed their transfers, Err otherwise
    ///
    /// # Errors
    /// Returns an error if:
    /// - Handle vectors don't match worker count
    /// - Any handle doesn't match its rank (via execute_transfer_on_rank validation)
    /// - Any worker fails to execute or complete the transfer
    pub async fn broadcast_execute_transfer(
        &self,
        src_handles: Vec<LayoutHandle>,
        dst_handles: Vec<LayoutHandle>,
        src_blocks: &[usize],
        dst_blocks: &[usize],
        options: TransferOptionsWire,
    ) -> Result<()> {
        let worker_count = self.cohort.worker_count().await;

        // Validate handle counts
        if src_handles.len() != worker_count {
            anyhow::bail!(
                "Source handle count ({}) doesn't match worker count ({})",
                src_handles.len(),
                worker_count
            );
        }
        if dst_handles.len() != worker_count {
            anyhow::bail!(
                "Destination handle count ({}) doesn't match worker count ({})",
                dst_handles.len(),
                worker_count
            );
        }

        debug!(
            "Broadcasting execute_transfer to {} workers: {} blocks",
            worker_count,
            src_blocks.len()
        );

        // Spawn parallel tasks, each calling execute_transfer_on_rank
        let mut tasks = Vec::new();

        for rank in 0..worker_count {
            let src_handle = src_handles[rank];
            let dst_handle = dst_handles[rank];
            let src_blocks = src_blocks.to_vec();
            let dst_blocks = dst_blocks.to_vec();
            let options = options.clone();

            // Clone self (Arc internally) for the task
            let leader = self.clone();

            let task = tokio::spawn(async move {
                leader
                    .execute_transfer_on_rank(
                        rank,
                        src_handle,
                        &src_blocks,
                        dst_handle,
                        &dst_blocks,
                        options,
                    )
                    .await
            });
            tasks.push(task);
        }

        // Wait for all tasks and collect results
        for (rank, task) in tasks.into_iter().enumerate() {
            let result = task
                .await
                .map_err(|e| anyhow::anyhow!("Task join error for rank {}: {}", rank, e))?;

            // Propagate any transfer errors
            result?;
        }

        debug!(
            "All {} workers successfully completed transfers",
            worker_count
        );
        Ok(())
    }

    /// Broadcast export layouts request to all workers.
    ///
    /// This collects serialized layout metadata from all workers in parallel,
    /// including NIXL registration data needed for remote memory access.
    ///
    /// # Returns
    /// Vector of SerializedLayout, one per worker in rank order
    pub async fn broadcast_export_layouts(&self) -> Result<Vec<SerializedLayout>> {
        debug!("Broadcasting export_layouts to all workers");

        let request = ExportLayoutsRequest {};

        // Use cohort's parallel broadcast with responses
        let results = self
            .cohort
            .par_broadcast_responses::<ExportLayoutsRequest, ExportLayoutsResponse>(
                "kvbm.cohort.worker.export_layouts",
                request,
                Duration::from_secs(30),
            )
            .await?;

        // Collect metadata from all workers
        let metadata_vec: Vec<SerializedLayout> = results.into_iter().map(|r| r.metadata).collect();

        debug!(
            "Collected layout metadata from {} workers",
            metadata_vec.len()
        );

        Ok(metadata_vec)
    }

    /// Broadcast layout creation request to all workers.
    ///
    /// This sends the layout configuration to all workers in parallel,
    /// instructing them to create their local physical layouts.
    ///
    /// # Arguments
    /// * `layout_config` - Configuration for the memory layout to create
    /// * `layout_type` - Type of layout structure (contiguous vs layer-separate)
    /// * `memory_type` - Type of memory storage (device, pinned, system, disk)
    /// * `name` - Name/identifier for this layout
    ///
    /// # Returns
    /// Vector of LayoutHandles, one per worker in rank order
    pub async fn broadcast_create_layout(
        &self,
        layout_config: LayoutConfig,
        layout_type: LayoutType,
        memory_type: MemoryType,
        name: String,
    ) -> Result<Vec<LayoutHandle>> {
        debug!(
            "Broadcasting create_layout '{}' to all workers (type: {:?}, memory: {:?})",
            name, layout_type, memory_type
        );

        let request = CreateLayoutRequest {
            layout_config,
            layout_type,
            memory_type,
            name: name.clone(),
        };

        // Use cohort's parallel broadcast with responses
        let results = self
            .cohort
            .par_broadcast_responses::<CreateLayoutRequest, CreateLayoutResponse>(
                "kvbm.cohort.worker.create_layout",
                request,
                Duration::from_secs(30),
            )
            .await?;

        // Collect handles from all workers, propagating errors
        let mut handles = Vec::new();
        for (idx, response) in results.into_iter().enumerate() {
            match response.result {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    return Err(anyhow::anyhow!(
                        "Worker {} failed to create layout '{}': {}",
                        idx,
                        name,
                        error
                    ));
                }
            }
        }

        debug!(
            "All {} workers successfully created layout '{}'",
            handles.len(),
            name
        );
        Ok(handles)
    }

    /// Wait for all workers to register their memory_descriptors handler.
    ///
    /// This acts as a fence, ensuring all workers have completed memory
    /// creation before we attempt to collect descriptors.
    pub async fn await_worker_memory_ready(&self) -> Result<()> {
        debug!("Waiting for workers to register memory_descriptors handler...");

        self.cohort
            .await_handler_on_all_workers(
                "kvbm.cohort.worker.memory_descriptors",
                Some(Duration::from_secs(30)),
            )
            .await?;

        debug!("All workers have memory_descriptors handler registered");
        Ok(())
    }

    /// Collect memory descriptors from all workers.
    ///
    /// This fetches the LayoutDescriptors from all workers in parallel,
    /// returning them in rank order.
    ///
    /// # Returns
    /// Vector of LayoutDescriptor vectors, one per worker in rank order
    pub async fn collect_memory_descriptors(&self) -> Result<Vec<Vec<LayoutDescriptor>>> {
        debug!("Collecting memory descriptors from all workers");

        let request = MemoryDescriptorsRequest {};

        // Use cohort's parallel broadcast with responses
        let results = self
            .cohort
            .par_broadcast_responses::<MemoryDescriptorsRequest, MemoryDescriptorsResponse>(
                "kvbm.cohort.worker.memory_descriptors",
                request,
                Duration::from_secs(30),
            )
            .await?;

        let descriptor_vecs: Vec<Vec<LayoutDescriptor>> =
            results.into_iter().map(|r| r.descriptors).collect();

        debug!(
            "Collected descriptors from {} workers",
            descriptor_vecs.len()
        );

        Ok(descriptor_vecs)
    }

    /// Get the cohort for direct access.
    pub fn cohort(&self) -> &Arc<LeaderWorkerCohort> {
        &self.cohort
    }

    /// Get the server for direct access.
    pub fn server(&self) -> &Arc<ActiveMessageServer> {
        &self.server
    }

    /// Shutdown the leader.
    pub async fn shutdown(&self) -> Result<()> {
        debug!("Shutting down Leader");
        self.server.shutdown().await?;
        Ok(())
    }
}
