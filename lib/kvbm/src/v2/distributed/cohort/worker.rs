// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker implementation for joining a leader-coordinated cohort.
//!
//! The worker is responsible for:
//! - Discovering the leader using a discovery mechanism
//! - Joining the leader's cohort with rank information
//! - Handling memory creation requests from the leader
//! - Registering handlers in phases to enable fencing/synchronization

use anyhow::Result;
use dynamo_am::ActiveMessageManager;
use dynamo_am::api::client::{ActiveMessageClient, WorkerAddress};
use dynamo_am::runtime::host::ActiveMessageServer;
use dynamo_am::zmq::ZmqServerConfig;
use dynamo_am::{
    handler_impls::typed_unary_handler_async_with_tracker, runtime::dispatcher::ControlMessage,
};
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio_util::sync::CancellationToken;
use tracing::{debug, info, warn};
use uuid::Uuid;

use super::discovery::LeaderDiscovery;
use super::messages::*;
use crate::v2::physical::layout::{LayoutDescriptor, PhysicalLayout};
use crate::v2::physical::manager::{LayoutHandle, TransportManager};

/// Worker that joins a leader-coordinated cohort.
///
/// The worker discovers the leader, joins the cohort, and participates
/// in coordinated memory setup. Handlers are registered in phases to
/// enable synchronization via `await_handler()`.
pub struct Worker {
    server: Arc<ActiveMessageServer>,
    discovery: Arc<dyn LeaderDiscovery>,
    /// Transport manager for layout registration and memory management
    transport_manager: Arc<TransportManager>,
    /// Stored memory descriptors, populated after create_memory
    memory_descriptors: Arc<RwLock<Vec<LayoutDescriptor>>>,
    /// Worker's rank in the distributed system (e.g., from torch.distributed)
    rank: Option<usize>,
    /// Total world size (number of workers)
    world_size: usize,
    /// Leader's instance ID (set after joining cohort)
    leader_id: Arc<RwLock<Option<dynamo_am::api::handler::InstanceId>>>,
}

impl Worker {
    /// Create a new worker with the given discovery mechanism.
    ///
    /// # Arguments
    /// * `discovery` - Discovery implementation to find the leader
    /// * `rank` - Optional rank for this worker (must be provided if leader expects ranks)
    /// * `world_size` - Total number of workers in the cohort
    /// * `cancel_token` - Cancellation token for shutdown coordination
    ///
    /// # Returns
    /// A new Worker instance and its WorkerAddress
    pub async fn new(
        discovery: Arc<dyn LeaderDiscovery>,
        rank: Option<usize>,
        world_size: usize,
        cancel_token: CancellationToken,
    ) -> Result<(Self, WorkerAddress)> {
        // Create the ActiveMessage server with ZMQ IPC transport (auto-generated unique path)
        let zmq_config = ZmqServerConfig::builder().ipc_endpoint("auto").build()?;

        let server = ActiveMessageServer::builder()
            .enable_zmq_with_config(zmq_config)
            .build(cancel_token)
            .await?;

        let peer_info = server.peer_info().await;

        debug!(
            "Worker created at {} (rank: {:?}, world_size: {})",
            peer_info.address.primary_endpoint().unwrap_or("unknown"),
            rank,
            world_size
        );

        // Create TransportManager with rank as worker_id (or 0 if no rank provided)
        let worker_id = rank.unwrap_or(0) as u64;

        let transport_manager = TransportManager::builder()
            .worker_id(worker_id)
            .nixl_agent_name(format!("worker-{}", worker_id))
            .cuda_device_id(0) // Always use device 0 for single-GPU testing
            .build()?;

        let worker = Self {
            server: Arc::new(server),
            discovery,
            transport_manager: Arc::new(transport_manager),
            memory_descriptors: Arc::new(RwLock::new(Vec::new())),
            rank,
            world_size,
            leader_id: Arc::new(RwLock::new(None)),
        };

        Ok((worker, peer_info.address))
    }

    /// Join the leader's cohort by discovering and connecting to the leader.
    ///
    /// This method:
    /// 1. Discovers the leader's address
    /// 2. Connects to the leader
    /// 3. Awaits the presence of the create_cohort handler
    /// 4. Sends a join request with rank/world_size information
    /// 5. Returns the position assigned by the leader
    ///
    /// # Returns
    /// The position assigned to this worker in the cohort
    pub async fn join_cohort(&self) -> Result<usize> {
        // Step 1: Discover the leader
        debug!("Discovering leader...");
        let leader_address = self.discovery.discover_leader().await?;
        info!(
            "Discovered leader at {}",
            leader_address.primary_endpoint().unwrap_or("unknown")
        );

        // Step 2: Connect to the leader
        let client = self.server.client();
        let leader_peer_info = dynamo_am::client::PeerInfo::new(
            Uuid::new_v4(), // Leader's instance ID (will be updated on first message)
            None,
            leader_address.primary_endpoint().unwrap_or(""),
        );

        debug!("Connecting to leader...");
        client.connect_to_peer(leader_peer_info.clone()).await?;

        // Step 3: Await the create_cohort handler on the leader
        debug!("Waiting for leader's create_cohort handler...");
        client
            .await_handler(
                leader_peer_info.instance_id,
                "kvbm.cohort.leader.create_cohort",
                None,
            )
            .await?;

        // Step 4: Send join request with response expectation using normal API
        debug!(
            "Sending join request (rank: {:?}, world_size: {})",
            self.rank, self.world_size
        );
        let request = CreateCohortRequest {
            rank: self.rank,
            world_size: self.world_size,
        };

        let response = client
            .typed_unary::<CreateCohortResponse>("kvbm.cohort.leader.create_cohort")?
            .payload(request)?
            .instance(leader_peer_info.instance_id)
            .send()
            .await?;

        if !response.accepted {
            let reason = response
                .reason
                .unwrap_or_else(|| "Unknown reason".to_string());
            anyhow::bail!("Leader rejected join request: {}", reason);
        }

        let position = response
            .position
            .ok_or_else(|| anyhow::anyhow!("Leader accepted but didn't provide position"))?;

        // Store leader's instance ID for later communication (e.g., barriers)
        *self.leader_id.write().await = Some(leader_peer_info.instance_id);

        info!("Successfully joined cohort at position {}", position);
        Ok(position)
    }

    /// Helper function to create a physical layout based on the configuration.
    ///
    /// # Arguments
    /// * `config` - Layout configuration
    /// * `layout_type` - Type of layout structure (contiguous vs layer-separate)
    /// * `memory_type` - Type of memory storage (device, pinned, system, disk)
    /// * `rank` - Worker rank (used for device_id)
    ///
    /// # Returns
    /// A PhysicalLayout ready for registration with the TransportManager
    fn create_physical_layout(
        transport_manager: &TransportManager,
        config: crate::v2::physical::layout::LayoutConfig,
        layout_type: LayoutType,
        memory_type: MemoryType,
        rank: usize,
    ) -> Result<PhysicalLayout> {
        use crate::v2::physical::layout::BlockDimension;

        // Get the NIXL agent from the transport manager context
        let nixl_agent = transport_manager.context().nixl_agent().clone();

        // Start building the layout with config
        let builder = PhysicalLayout::builder(nixl_agent).with_config(config);

        // Apply layout type and memory allocation
        match (layout_type, memory_type) {
            (LayoutType::FullyContiguous, MemoryType::System) => {
                builder.fully_contiguous().allocate_system().build()
            }
            (LayoutType::FullyContiguous, MemoryType::Pinned) => builder
                .fully_contiguous()
                .allocate_pinned(Some(rank as u32))
                .build(),
            (LayoutType::FullyContiguous, MemoryType::Device) => builder
                .fully_contiguous()
                .allocate_device(rank as u32)
                .build(),
            (LayoutType::FullyContiguous, MemoryType::Disk) => {
                use std::path::PathBuf;
                let disk_path = PathBuf::from(format!("/tmp/kvbm_layout_worker_{}", rank));
                builder
                    .fully_contiguous()
                    .allocate_disk(Some(disk_path))
                    .build()
            }
            (LayoutType::LayerSeparate, MemoryType::System) => builder
                .layer_separate(BlockDimension::BlockIsFirstDim)
                .allocate_system()
                .build(),
            (LayoutType::LayerSeparate, MemoryType::Pinned) => builder
                .layer_separate(BlockDimension::BlockIsFirstDim)
                .allocate_pinned(Some(rank as u32))
                .build(),
            (LayoutType::LayerSeparate, MemoryType::Device) => builder
                .layer_separate(BlockDimension::BlockIsFirstDim)
                .allocate_device(rank as u32)
                .build(),
            (LayoutType::LayerSeparate, MemoryType::Disk) => {
                use std::path::PathBuf;
                let disk_path = PathBuf::from(format!("/tmp/kvbm_layout_worker_{}", rank));
                builder
                    .layer_separate(BlockDimension::BlockIsFirstDim)
                    .allocate_disk(Some(disk_path))
                    .build()
            }
        }
    }

    /// Register the `kvbm.cohort.worker.execute_transfer` handler.
    ///
    /// This handler executes a local transfer between two layouts using the
    /// TransportManager. It receives transfer parameters from the leader,
    /// executes the transfer, awaits completion, and returns the result.
    ///
    /// # Returns
    /// Result indicating whether handler registration succeeded
    pub async fn register_execute_transfer_handler(&self) -> Result<()> {
        let transport_manager = self.transport_manager.clone();
        let task_tracker = tokio_util::task::TaskTracker::new();

        let handler = typed_unary_handler_async_with_tracker(
            "kvbm.cohort.worker.execute_transfer".to_string(),
            move |ctx: dynamo_am::runtime::handler_impls::TypedContext<ExecuteTransferRequest>| {
                let transport_manager = transport_manager.clone();

                async move {
                    debug!(
                        "Received execute_transfer request: src={}, dst={}, {} blocks",
                        ctx.input.src_handle,
                        ctx.input.dst_handle,
                        ctx.input.src_blocks.len()
                    );

                    // Execute the transfer
                    let result = async {
                        // Convert wire options to TransferOptions
                        let options = ctx.input.options.into();

                        // Execute transfer and get notification
                        let notification = transport_manager
                            .execute_transfer(
                                ctx.input.src_handle,
                                &ctx.input.src_blocks,
                                ctx.input.dst_handle,
                                &ctx.input.dst_blocks,
                                options,
                            )
                            .map_err(|e| format!("Failed to start transfer: {}", e))?;

                        // Await completion (notification implements Future<Output=Result<()>>)
                        notification
                            .await
                            .map_err(|e| format!("Transfer failed: {}", e))?;

                        debug!("Transfer completed successfully");
                        Ok::<(), String>(())
                    }
                    .await;

                    if let Err(ref e) = result {
                        warn!("Transfer error: {}", e);
                    }

                    Ok(ExecuteTransferResponse { result })
                }
            },
            task_tracker,
        );

        // Register with dispatcher
        self.server
            .control_tx()
            .send(ControlMessage::Register {
                name: "kvbm.cohort.worker.execute_transfer".to_string(),
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register execute_transfer handler: {}", e))?;

        debug!("Registered kvbm.cohort.worker.execute_transfer handler");
        Ok(())
    }

    /// Register the `kvbm.cohort.worker.export_layouts` handler.
    ///
    /// This handler exports all layouts registered with the TransportManager,
    /// returning the serialized metadata including NIXL registration data
    /// needed for remote memory access.
    ///
    /// # Returns
    /// Result indicating whether handler registration succeeded
    pub async fn register_export_layouts_handler(&self) -> Result<()> {
        let transport_manager = self.transport_manager.clone();
        let task_tracker = tokio_util::task::TaskTracker::new();

        let handler = typed_unary_handler_async_with_tracker(
            "kvbm.cohort.worker.export_layouts".to_string(),
            move |_ctx: dynamo_am::runtime::handler_impls::TypedContext<ExportLayoutsRequest>| {
                let transport_manager = transport_manager.clone();

                async move {
                    debug!("Received export_layouts request");

                    // Export metadata from the TransportManager
                    let metadata = match transport_manager.export_metadata() {
                        Ok(metadata) => metadata,
                        Err(e) => {
                            let error_msg = format!("Failed to export layout metadata: {}", e);
                            warn!("{}", error_msg);
                            return Err(error_msg);
                        }
                    };

                    debug!(
                        "Successfully exported layout metadata ({} bytes)",
                        metadata.len()
                    );

                    Ok(ExportLayoutsResponse { metadata })
                }
            },
            task_tracker,
        );

        // Register with dispatcher
        self.server
            .control_tx()
            .send(ControlMessage::Register {
                name: "kvbm.cohort.worker.export_layouts".to_string(),
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register export_layouts handler: {}", e))?;

        debug!("Registered kvbm.cohort.worker.export_layouts handler");
        Ok(())
    }

    /// Register the `kvbm.cohort.worker.create_layout` handler.
    ///
    /// This handler processes layout creation requests from the leader.
    /// Creates a physical layout, registers it with the TransportManager,
    /// and returns the handle.
    ///
    /// # Returns
    /// Result indicating whether handler registration succeeded
    pub async fn register_create_layout_handler(&self) -> Result<()> {
        let transport_manager = self.transport_manager.clone();
        let rank = self.rank.unwrap_or(0); // Use 0 as default if no rank provided
        let task_tracker = tokio_util::task::TaskTracker::new();

        let handler = typed_unary_handler_async_with_tracker(
            "kvbm.cohort.worker.create_layout".to_string(),
            move |ctx: dynamo_am::runtime::handler_impls::TypedContext<CreateLayoutRequest>| {
                let transport_manager = transport_manager.clone();

                async move {
                    debug!(
                        "Received create_layout request from {}: {:?}",
                        ctx.sender_id, ctx.input.name
                    );

                    // Create the physical layout
                    let result = (|| -> Result<LayoutHandle, String> {
                        let physical_layout = Self::create_physical_layout(
                            &transport_manager,
                            ctx.input.layout_config,
                            ctx.input.layout_type,
                            ctx.input.memory_type,
                            rank,
                        )
                        .map_err(|e| format!("Failed to create physical layout: {}", e))?;

                        let layout_handle = transport_manager
                            .register_layout(physical_layout)
                            .map_err(|e| format!("Failed to register layout: {}", e))?;

                        debug!(
                            "Layout '{}' created and registered with handle: {}",
                            ctx.input.name, layout_handle
                        );

                        Ok(layout_handle)
                    })();

                    if let Err(ref e) = result {
                        warn!("{}", e);
                    }

                    Ok(CreateLayoutResponse { result })
                }
            },
            task_tracker,
        );

        // Register with dispatcher
        self.server
            .control_tx()
            .send(ControlMessage::Register {
                name: "kvbm.cohort.worker.create_layout".to_string(),
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register create_layout handler: {}", e))?;

        debug!("Registered kvbm.cohort.worker.create_layout handler");
        Ok(())
    }

    /// Internal helper to register the memory_descriptors handler.
    ///
    /// This can be used in the future for explicit export/descriptor exchange operations.
    /// Currently unused but kept for future functionality.
    #[allow(dead_code)]
    async fn register_memory_descriptors_handler_internal(
        server: Arc<ActiveMessageServer>,
        memory_descriptors: Arc<RwLock<Vec<LayoutDescriptor>>>,
    ) -> Result<()> {
        let task_tracker = tokio_util::task::TaskTracker::new();

        let handler = typed_unary_handler_async_with_tracker(
            "kvbm.cohort.worker.memory_descriptors".to_string(),
            move |_ctx: dynamo_am::runtime::handler_impls::TypedContext<
                MemoryDescriptorsRequest,
            >| {
                let memory_descriptors = memory_descriptors.clone();

                async move {
                    debug!("Received memory_descriptors request");

                    let descriptors = memory_descriptors.read().await.clone();

                    Ok(MemoryDescriptorsResponse { descriptors })
                }
            },
            task_tracker,
        );

        // Register with dispatcher
        server
            .control_tx()
            .send(ControlMessage::Register {
                name: "kvbm.cohort.worker.memory_descriptors".to_string(),
                dispatcher: handler,
            })
            .await
            .map_err(|e| anyhow::anyhow!("Failed to register memory_descriptors handler: {}", e))?;

        debug!("Registered kvbm.cohort.worker.memory_descriptors handler");
        Ok(())
    }

    /// Get the server for direct access.
    pub fn server(&self) -> &Arc<ActiveMessageServer> {
        &self.server
    }

    /// Get the current memory descriptors.
    pub async fn memory_descriptors(&self) -> Vec<LayoutDescriptor> {
        self.memory_descriptors.read().await.clone()
    }

    /// Signal arrival at a named barrier to the leader.
    ///
    /// This sends a barrier_reached message to the leader, allowing the leader
    /// to coordinate synchronization points across all workers in the cohort.
    ///
    /// # Arguments
    /// * `barrier_name` - Name of the barrier to reach
    ///
    /// # Example
    /// ```rust,ignore
    /// // After completing memory setup, signal the leader
    /// worker.barrier("memory_ready").await?;
    /// ```
    pub async fn barrier(&self, barrier_name: &str) -> Result<()> {
        let leader_id = self.leader_id.read().await;
        let leader_id = leader_id.ok_or_else(|| {
            anyhow::anyhow!("Cannot reach barrier: worker has not joined cohort yet")
        })?;

        debug!("Reaching barrier '{}'", barrier_name);

        let request = BarrierReachedRequest {
            barrier_name: barrier_name.to_string(),
        };

        let client = self.server.client();
        let response = client
            .typed_unary::<BarrierReachedResponse>("kvbm.cohort.barrier")?
            .payload(request)?
            .instance(leader_id)
            .send()
            .await?;

        if !response.acknowledged {
            anyhow::bail!("Leader did not acknowledge barrier arrival");
        }

        debug!("Successfully reached barrier '{}'", barrier_name);
        Ok(())
    }

    /// Shutdown the worker.
    pub async fn shutdown(&self) -> Result<()> {
        debug!("Shutting down Worker");
        self.server.shutdown().await?;
        Ok(())
    }
}
