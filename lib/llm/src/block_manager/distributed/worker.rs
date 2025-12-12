// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use async_trait::async_trait;
use std::sync::OnceLock;
use transfer::*;
use super::g4_transfer::G4TransferHandler;
use utils::*;
use zmq::*;

use crate::block_manager::{
    BasicMetadata, BlockMetadata, LayoutConfigBuilder, NixlLayout, Storage,
    block::{
        Block, layout_to_blocks, locality,
        transfer::{PoolConfig, TransferContext},
    },
    config::ObjectStorageConfig,
    connector::scheduler::TransferSchedulerClient,
    layout::LayoutType,
    offload::{MAX_CONCURRENT_TRANSFERS, MAX_TRANSFER_BATCH_SIZE},
    storage::{DeviceAllocator, DeviceStorage, DiskAllocator, PinnedAllocator, torch::TorchTensor},
    v2::memory::DeviceStorage as DeviceStorageV2,
    v2::physical::{
        layout::{BlockDimension, LayoutConfig as LayoutConfigV2, builder::PhysicalLayoutBuilder},
        manager::TransportManager,
        transfer::{LayoutHandle, NixlAgent as NixlAgentV2, TransferCapabilities},
    },
};

use super::registry::{ObjectRegistry, create_registry_from_env};

use super::transfer_object::ObjectTransferHandler;

use derive_builder::Builder;
use nixl_sys::{Agent as NixlAgent, Params};
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use tokio::runtime::Handle;
use tokio_util::sync::CancellationToken;

use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use tokio::sync::{Mutex, RwLock, oneshot};

struct WorkerState {
    ready_for_ping: AtomicBool,
}

impl WorkerState {
    fn new() -> Self {
        Self {
            ready_for_ping: AtomicBool::new(false),
        }
    }
    fn mark_ready(&self) {
        self.ready_for_ping.store(true, Ordering::SeqCst);
    }
    fn is_ready(&self) -> bool {
        self.ready_for_ping.load(Ordering::SeqCst)
    }
}

pub fn load_and_validate_tensors(
    tensors: &[Arc<dyn TorchTensor>],
    device_id: usize,
) -> anyhow::Result<(Vec<DeviceStorage>, Vec<usize>)> {
    let mut shape = None;

    let mut device_tensors = Vec::with_capacity(tensors.len());
    let allocator = DeviceAllocator::new(device_id)?;

    for tensor in tensors {
        // Check the stride, and ensure our tensor is contiguous.
        // TODO: We eventually need to be able to handle this.
        let stride = tensor.stride();
        tracing::debug!("stride: {:?}", stride);
        tracing::debug!("stride is monotonically decreasing for NHD layout");
        tracing::debug!("stride is NOT monotonically decreasing for HND layout");

        // Check that all layer tensors have the same shape.
        // TODO: We eventually need to support the weirder models with heterogenous layers.
        if let Some(shape) = shape.as_ref() {
            if *shape != tensor.shape() {
                return Err(anyhow::anyhow!(
                    "All tensors must have the same shape! Got {:?} and {:?}",
                    *shape,
                    tensor.shape()
                ));
            }
        } else {
            shape = Some(tensor.shape());
        }

        // Build the storage object from the tensor.
        let device_tensor = DeviceStorage::new_from_torch(allocator.ctx(), tensor.clone())?;

        device_tensors.push(device_tensor);
    }

    Ok((device_tensors, shape.unwrap()))
}

fn build_agent(
    worker_id: usize,
    use_gds: bool,
    object_storage_config: Option<&ObjectStorageConfig>,
) -> anyhow::Result<NixlAgent> {
    let agent = NixlAgent::new(&format!("kvbm-worker-{}", worker_id))?;
    if use_gds {
        let (_, gds_params) = agent.get_plugin_params("GDS_MT")?;
        agent.create_backend("GDS_MT", &gds_params)?;
    }
    let (_, posix_params) = agent.get_plugin_params("POSIX")?;
    agent.create_backend("POSIX", &posix_params)?;

    // Load OBJ backend for G4 object storage with proper config
    if let Some(config) = object_storage_config {
        // Build OBJ params from config + credentials from env (same as V2 path)
        let resolved_bucket = config.resolve_bucket(worker_id as u32);
        tracing::info!("Resolved bucket for worker {}: {}", worker_id, resolved_bucket);

        let mut obj_param_pairs: Vec<(String, String)> = vec![
            ("bucket".to_string(), resolved_bucket),
        ];

        if let Some(endpoint) = &config.endpoint_override {
            obj_param_pairs.push(("endpoint".to_string(), endpoint.clone()));
        }
        if let Some(region) = &config.region {
            obj_param_pairs.push(("region".to_string(), region.clone()));
        }

        // Credentials from env vars (should not be in config for security)
        if let Ok(access_key) = std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY") {
            obj_param_pairs.push(("access_key".to_string(), access_key));
        }
        if let Ok(secret_key) = std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY") {
            obj_param_pairs.push(("secret_key".to_string(), secret_key));
        }

        match Params::from(obj_param_pairs.iter().map(|(k, v)| (k.as_str(), v.as_str()))) {
            Ok(obj_params) => {
                match agent.create_backend("OBJ", &obj_params) {
                    Ok(_) => {
                        tracing::info!("NIXL OBJ backend loaded for object storage (V1 path)");
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create NIXL OBJ backend: {}. G4 object storage will not be available.",
                            e
                        );
                    }
                }
            }
            Err(e) => {
                tracing::warn!(
                    "Failed to build OBJ backend params: {}. G4 object storage will not be available.",
                    e
                );
            }
        }
    }

    Ok(agent)
}

// Helper: perform allocation and build transfer handler (factored from previous code)
async fn perform_allocation_and_build_handler(
    device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
    mut layout_builder: LayoutConfigBuilder,
    worker_config: KvbmWorkerConfig,
    leader_meta: LeaderMetadata,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
) -> anyhow::Result<Arc<dyn BlockTransferHandler>> {
    let use_v2_transfer = std::env::var("DYN_KVBM_USE_V2_TRANSFER_EXPERIMENTAL")
        .unwrap_or("0".to_string())
        .parse::<usize>()
        .map(|v| v > 0)
        .unwrap_or(false);

    if use_v2_transfer {
        tracing::warn!("Using V2 transfer handler. This is experimental. Use at your own risk.");

        // Build backend params list based on configuration
        let mut backend_params: Vec<(&str, Params)> = Vec::new();

        // Helper to create empty params (signals "use defaults")
        let empty_params = || Params::from(std::iter::empty::<(&str, &str)>());

        // POSIX backend - get params from env or use defaults
        let posix_params = NixlAgentV2::create_backend_params_from_env("POSIX")?
            .map_or_else(|| empty_params(), Ok)?;
        backend_params.push(("POSIX", posix_params));

        // GDS_MT backend if disk blocks are configured
        if leader_meta.num_disk_blocks > 0 {
            let gds_params = NixlAgentV2::create_backend_params_from_env("GDS_MT")?
                .map_or_else(|| empty_params(), Ok)?;
            backend_params.push(("GDS_MT", gds_params));
        }

        // OBJ backend for object storage - build params with resolved bucket
        if let Some(config) = &leader_meta.object_storage_config {
            let resolved_bucket = config.resolve_bucket(worker_id as u32);
            tracing::info!("Resolved bucket for worker {}: {}", worker_id, resolved_bucket);

            // Build OBJ params from config + credentials from env
            let mut obj_param_pairs: Vec<(String, String)> = vec![
                ("bucket".to_string(), resolved_bucket),
            ];

            if let Some(endpoint) = &config.endpoint_override {
                obj_param_pairs.push(("endpoint".to_string(), endpoint.clone()));
            }
            if let Some(region) = &config.region {
                obj_param_pairs.push(("region".to_string(), region.clone()));
            }

            // Credentials still come from env vars (should not be in config for security)
            if let Ok(access_key) = std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_ACCESS_KEY") {
                obj_param_pairs.push(("access_key".to_string(), access_key));
            }
            if let Ok(secret_key) = std::env::var("DYN_KVBM_NIXL_BACKEND_OBJ_SECRET_KEY") {
                obj_param_pairs.push(("secret_key".to_string(), secret_key));
            }

            let obj_params = Params::from(
                obj_param_pairs.iter().map(|(k, v)| (k.as_str(), v.as_str()))
            )?;
            backend_params.push(("OBJ", obj_params));
        }

        let agent = NixlAgentV2::new_with_backends_and_params(
            worker_id.to_string().as_str(),
            &backend_params,
        )?;

        let mut layout_config = LayoutConfigV2::builder()
            .num_blocks(device_layout.config().num_blocks)
            .num_layers(device_layout.config().num_layers)
            .outer_dim(device_layout.config().outer_dim)
            .inner_dim(device_layout.config().inner_dim)
            .page_size(device_layout.config().page_size)
            .alignment(device_layout.config().alignment)
            .dtype_width_bytes(device_layout.config().dtype_width_bytes)
            .build()?;

        let layout_config_for_handler = layout_config.clone();
        let v2_device_layout =
            PhysicalLayoutBuilder::new(agent.clone()).with_config(layout_config.clone());

        let v2_device_layout =
            if let LayoutType::LayerSeparate { outer_contiguous } = device_layout.layout_type() {
                v2_device_layout.layer_separate(if outer_contiguous {
                    BlockDimension::BlockIsSecondDim
                } else {
                    BlockDimension::BlockIsFirstDim
                })
            } else {
                v2_device_layout.fully_contiguous()
            };

        let regions = device_layout
            .storage()
            .iter()
            .map(|s| DeviceStorageV2::from_v1(s).unwrap())
            .collect::<Vec<_>>();
        let v2_device_layout = v2_device_layout.with_memory_regions(regions)?.build()?;

        let host_layout = if leader_meta.num_host_blocks > 0 {
            layout_config.num_blocks = leader_meta.num_host_blocks;
            Some(
                PhysicalLayoutBuilder::new(agent.clone())
                    .with_config(layout_config.clone())
                    .fully_contiguous()
                    .allocate_pinned(true)
                    .build()?,
            )
        } else {
            None
        };

        let disk_layout = if leader_meta.num_disk_blocks > 0 {
            layout_config.num_blocks = leader_meta.num_disk_blocks;
            Some(
                PhysicalLayoutBuilder::new(agent.clone())
                    .with_config(layout_config)
                    .fully_contiguous()
                    .allocate_disk(None)
                    .build()?,
            )
        } else {
            None
        };

        let transport_manager = TransportManager::builder()
            .capabilities(TransferCapabilities::default().with_gds(true))
            .worker_id(worker_id as u64)
            .nixl_agent(agent.clone())
            .cuda_device_id(device_id)
            .build()?;

        let handler = BlockTransferHandlerV2::new(
            Some(v2_device_layout),
            host_layout,
            disk_layout,
            transport_manager,
            scheduler_client,
            Some(layout_config_for_handler),
            Some(agent),
        )?;

        Ok(Arc::new(handler) as Arc<dyn BlockTransferHandler>)
    } else {
        let agent = build_agent(
            worker_id,
            leader_meta.num_disk_blocks > 0,
            leader_meta.object_storage_config.as_ref(),
        )?;
        let pool_config = PoolConfig {
            enable_pool: true,
            max_concurrent_transfers: MAX_CONCURRENT_TRANSFERS,
            max_transfer_batch_size: MAX_TRANSFER_BATCH_SIZE,
            num_outer_components: device_layout.config().outer_dim,
            num_layers: device_layout.config().num_layers,
        };
        let transfer_context = Arc::new(TransferContext::new(
            Arc::new(Some(agent)),
            DeviceAllocator::new(device_id)?.ctx().new_stream()?,
            Handle::current(),
            Some(pool_config),
        ));

        // device
        let device_blocks = Some(KvbmWorker::make_layout::<_, BasicMetadata>(
            device_layout,
            transfer_context.nixl_agent().as_ref(),
            0,
            worker_id,
        )?);
        // host
        let host_blocks = if leader_meta.num_host_blocks > 0 {
            let host_allocator = Arc::new(PinnedAllocator::default());
            let host_layout = layout_builder
                .num_blocks(leader_meta.num_host_blocks)
                .build()?
                .allocate_layout(worker_config.host_layout_type, host_allocator)?;
            Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                host_layout,
                transfer_context.nixl_agent().as_ref(),
                1,
                worker_id,
            )?)
        } else {
            None
        };
        // disk
        let disk_blocks = if leader_meta.num_disk_blocks > 0 {
            let disk_allocator = Arc::new(DiskAllocator);
            let disk_layout = layout_builder
                .num_blocks(leader_meta.num_disk_blocks)
                .build()?
                .allocate_layout(worker_config.disk_layout_type, disk_allocator)?;
            Some(KvbmWorker::make_layout::<_, BasicMetadata>(
                disk_layout,
                transfer_context.nixl_agent().as_ref(),
                2,
                worker_id,
            )?)
        } else {
            None
        };

        let handler = BlockTransferHandlerV1::new(
            device_blocks,
            host_blocks,
            disk_blocks,
            transfer_context.clone(),
            scheduler_client,
        )?;

        // Initialize G4 object storage on V1 path if configured
        if let Some(config) = &leader_meta.object_storage_config {
            if leader_meta.num_object_blocks > 0 {
                // Get blocks from the handler (already converted to LocalBlockData)
                let host_list = handler.host_blocks().ok_or_else(|| {
                    anyhow::anyhow!("Host blocks required for G4 bounce buffers")
                })?;

                let device_list = handler.device_blocks().ok_or_else(|| {
                    anyhow::anyhow!("Device blocks required for G4")
                })?;

                // Create local registry
                let local_registry = Arc::new(ObjectRegistry::new());

                // Connect to distributed registry (if configured)
                let distributed_registry = create_registry_from_env().await;

                // Create the G4 handler
                match G4TransferHandler::new(
                    transfer_context.nixl_agent(),
                    local_registry,
                    distributed_registry,
                    config.clone(),
                    host_list.clone(),
                    device_list.clone(),
                    transfer_context.clone(),
                    worker_id as u64,
                ) {
                    Ok(obj_handler) => {
                        handler.set_object_handler(Arc::new(obj_handler));
                        tracing::info!("G4 object storage enabled on V1 path (write-through)");
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Failed to create G4 handler on V1 path: {}. Continuing without G4.",
                            e
                        );
                    }
                }
            }
        }

        Ok(Arc::new(handler) as Arc<dyn BlockTransferHandler>)
    }
}

struct WorkerMetadataHandler {
    num_device_blocks: usize,
    bytes_per_block: usize,
}

#[async_trait]
impl Handler for WorkerMetadataHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        let payload = bincode::serde::encode_to_vec(
            &WorkerMetadata {
                num_device_blocks: self.num_device_blocks,
                bytes_per_block: self.bytes_per_block,
            },
            bincode::config::standard(),
        )?;
        message
            .reply(ZMQ_WORKER_METADATA_MESSAGE, &[payload])
            .await?;
        Ok(())
    }
}

type TransferHandlerSender = Mutex<Option<oneshot::Sender<Arc<dyn BlockTransferHandler>>>>;

// Leader sends allocation config -> allocate -> publish handler -> mark ready -> ACK
struct LeaderMetadataHandler {
    state: Arc<WorkerState>,
    device_layout: Mutex<Option<Box<dyn NixlLayout<StorageType = DeviceStorage>>>>,
    layout_builder: LayoutConfigBuilder,
    worker_config: KvbmWorkerConfig,
    worker_id: usize,
    device_id: usize,
    scheduler_client: Option<TransferSchedulerClient>,
    handler_cell: Arc<RwLock<Option<Arc<dyn BlockTransferHandler>>>>,
    handler_tx: Arc<TransferHandlerSender>,
    started: AtomicBool,
}

#[async_trait]
impl Handler for LeaderMetadataHandler {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        // Always ACK ASAP so Drop can't panic and leader can finish the round.
        if let Err(e) = message.ack().await {
            tracing::error!("leader_metadata: failed to ACK: {e:#}");
        }

        // Validate payload; if bad, ignore.
        if message.data.len() != 1 {
            tracing::error!(
                "leader_metadata expects 1 payload frame (got {})",
                message.data.len()
            );
            return Ok(());
        }
        let leader_meta: LeaderMetadata = match bincode::serde::decode_from_slice(
            &message.data[0],
            bincode::config::standard(),
        ) {
            Ok((m, _)) => m,
            Err(e) => {
                tracing::error!("leader_metadata: bad payload: {e:#}");
                return Ok(());
            }
        };

        // Single-flight: only the first message triggers allocation.
        if self
            .started
            .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            tracing::debug!("leader_metadata: allocation already started; dropping duplicate");
            return Ok(());
        }

        // Take device_layout once.
        let dev_layout = {
            let mut guard = self.device_layout.lock().await;
            match guard.take() {
                Some(d) => d,
                None => {
                    tracing::warn!("leader_metadata: device_layout already consumed; dropping");
                    return Ok(());
                }
            }
        };

        // Capture what we need and run allocation in the background.
        let layout_builder = self.layout_builder.clone();
        let worker_config = self.worker_config.clone();
        let worker_id = self.worker_id;
        let device_id = self.device_id;
        let scheduler_client = self.scheduler_client.clone();
        let handler_cell = self.handler_cell.clone();
        let handler_tx = self.handler_tx.clone();
        let state = self.state.clone();

        // Capture G4 config before moving leader_meta
        let g4_config = leader_meta.object_storage_config.clone();
        let g4_num_blocks = leader_meta.num_object_blocks;

        tokio::spawn(async move {
            match perform_allocation_and_build_handler(
                dev_layout,
                layout_builder,
                worker_config,
                leader_meta,
                worker_id,
                device_id,
                scheduler_client,
            )
            .await
            {
                Ok(handler) => {
                    // Initialize G4 object storage if configured
                    if let Some(config) = &g4_config {
                        if g4_num_blocks > 0 {
                            // Try V2 handler first
                            if let Some(v2_handler) = handler
                                .as_any()
                                .downcast_ref::<BlockTransferHandlerV2>()
                            {
                                // Create object handler using V2's resources
                                let registry = Arc::new(ObjectRegistry::new());
                                let distributed_registry = create_registry_from_env().await;
                                if let Some(obj_handler) = v2_handler
                                    .create_object_handler(config.clone(), registry, distributed_registry)
                                {
                                    let obj_handler = Arc::new(obj_handler);
                                    v2_handler.set_object_handler(obj_handler);
                                    tracing::info!(
                                        "G4 object storage initialized on worker V2 (write-through enabled)"
                                    );
                                } else {
                                    tracing::warn!(
                                        "Failed to create G4 object handler V2 (host layout or resources missing)"
                                    );
                                }
                            }
                            // V1 G4 initialization is done inside perform_allocation_and_build_handler
                            // Check if V1 handler has G4 enabled
                            else if let Some(v1_handler) = handler
                                .as_any()
                                .downcast_ref::<BlockTransferHandlerV1>()
                            {
                                if v1_handler.has_g4() {
                                    tracing::info!(
                                        "G4 object storage initialized on worker V1 (write-through enabled)"
                                    );
                                } else {
                                    tracing::debug!(
                                        "G4 not initialized on V1 handler (may have failed during allocation)"
                                    );
                                }
                            } else {
                                tracing::debug!("Unknown handler type, G4 not initialized");
                            }
                        }
                    }

                    // Install transfer handler
                    {
                        let mut w = handler_cell.write().await;
                        *w = Some(handler.clone());
                    }
                    // Return handler to creator (once)
                    {
                        let mut g = handler_tx.lock().await;
                        if let Some(tx) = g.take() {
                            let _ = tx.send(handler);
                        }
                    }
                    // Now the worker can ACK pings
                    state.mark_ready();
                    tracing::info!("allocation finished; worker is ping-ACK-able");
                }
                Err(e) => {
                    tracing::error!("allocation failed: {e:#}");
                    // leave ready=false so pings keep being ignored
                }
            }
        });

        Ok(())
    }
}

// Gated ping: the worker can only response to ping after the state is ready
struct GatedPing {
    state: Arc<WorkerState>,
    // fired exactly once after the first successful ping ACK
    layout_ready_tx: Mutex<Option<oneshot::Sender<String>>>,
}

#[async_trait]
impl Handler for GatedPing {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        if !self.state.is_ready() {
            tracing::info!(
                "KVBM worker is under initialization. It could take a while if set with large CPU or DISK cache size. Please wait..."
            );
            tracing::debug!("Ping received but worker not ready; deferring ACK");
            // Prevent Drop panic; leader won't get an ACK for this round and will retry.
            message.mark_handled();
            return Ok(());
        }

        message.ack().await?;

        // After a successful ACK, flip the readiness oneshot exactly once
        let mut guard = self.layout_ready_tx.lock().await;
        if let Some(tx) = guard.take() {
            let _ = tx.send("ping-acked".to_string());
            tracing::info!("Reported ping-ready after first ACK");
        }

        Ok(())
    }
}

// Transfer dispatcher that waits until block transfer handler exists
struct BlockTransferDispatch {
    cell: Arc<RwLock<Option<Arc<dyn BlockTransferHandler>>>>,
}

#[async_trait]
impl Handler for BlockTransferDispatch {
    async fn handle(&self, message: MessageHandle) -> anyhow::Result<()> {
        let maybe = { self.cell.read().await.clone() };
        if let Some(inner) = maybe {
            inner.handle(message).await
        } else {
            Err(anyhow::anyhow!("transfer handler not ready yet"))
        }
    }
}

/// Handler for G4 onboard requests (G4->Host->Device transfers).
///
/// This handler is responsible for:
/// 1. Onboarding blocks from object storage to host bounce buffers
/// 2. Transferring from host bounce buffers to device
/// 3. Releasing bounce buffers
///
/// Supports both V1 and V2 transfer handlers.
struct G4OnboardDispatch {
    cell: Arc<RwLock<Option<Arc<dyn BlockTransferHandler>>>>,
}

#[async_trait]
impl Handler for G4OnboardDispatch {
    async fn handle(&self, mut message: MessageHandle) -> anyhow::Result<()> {
        // Deserialize request first (before any async work)
        if message.data.len() != 1 {
            message.ack().await?;
            return Err(anyhow::anyhow!(
                "G4 onboard request must have exactly one data element"
            ));
        }

        let request: G4OnboardRequest = match serde_json::from_slice(&message.data[0]) {
            Ok(r) => r,
            Err(e) => {
                message.ack().await?;
                return Err(anyhow::anyhow!("Failed to deserialize G4 onboard request: {}", e));
            }
        };

        // Run the actual handler logic, capturing any error
        let result = self.handle_inner(request).await;

        // ALWAYS ack the message before returning, even on error
        // This prevents the "Message was not acked!" panic
        if let Err(ref e) = result {
            tracing::error!("G4 onboard failed: {}", e);
        }
        message.ack().await?;

        result
    }
}

impl G4OnboardDispatch {
    async fn handle_inner(&self, request: G4OnboardRequest) -> anyhow::Result<()> {
        let maybe = { self.cell.read().await.clone() };
        let handler = match maybe {
            Some(h) => h,
            None => {
                tracing::warn!("G4 onboard handler not ready yet");
                return Err(anyhow::anyhow!("G4 onboard handler not ready yet"));
            }
        };

        tracing::info!(
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            num_blocks = request.sequence_hashes.len(),
            "Processing G4 onboard request"
        );

        // Try V1 handler first
        if let Some(v1_handler) = handler.as_any().downcast_ref::<BlockTransferHandlerV1>() {
            let object_handler = v1_handler
                .object_handler()
                .ok_or_else(|| anyhow::anyhow!("G4 not configured on V1 handler"))?;

            // If connector_req is present, schedule the transfer with the connector scheduler
            let completion_handle = if let Some(connector_req) = request.connector_req {
                let client = v1_handler
                    .scheduler_client()
                    .ok_or_else(|| anyhow::anyhow!(
                        "scheduler client is required for G4 onboard with connector_req"
                    ))?;

                let handle = client.schedule_transfer(connector_req).await?;
                assert_eq!(
                    handle.scheduler_decision(),
                    crate::block_manager::connector::scheduler::SchedulingDecision::Execute
                );
                Some(handle)
            } else {
                None
            };

            // Execute G4->H->D transfer (V1 path)
            let result = object_handler
                .onboard(
                    &request.sequence_hashes,
                    &request.host_block_ids,
                    &request.device_block_ids,
                )
                .await;

            // Mark the transfer as complete (success or failure)
            if let Some(handle) = completion_handle {
                match &result {
                    Ok(_) => handle.mark_complete(Ok(())).await,
                    Err(e) => handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await,
                }
            }

            let matched_hashes = result?;

            tracing::debug!(
                request_id = %request.request_id,
                operation_id = %request.operation_id,
                "G4 onboard complete: {} blocks",
                matched_hashes.len()
            );

            return Ok(());
        }

        // Fall back to V2 handler
        let v2_handler = handler
            .as_any()
            .downcast_ref::<BlockTransferHandlerV2>()
            .ok_or_else(|| anyhow::anyhow!("G4 requires V1 or V2 transfer handler"))?;

        // Get the object handler
        let object_handler = v2_handler
            .object_handler()
            .ok_or_else(|| anyhow::anyhow!("Object handler not configured"))?;

        let device_handle = v2_handler
            .device_handle()
            .ok_or_else(|| anyhow::anyhow!("Device handle not available"))?;

        // If connector_req is present, schedule the transfer with the connector scheduler
        // This allows the worker-side connector to track completion
        let completion_handle = if let Some(connector_req) = request.connector_req {
            let client = v2_handler
                .scheduler_client()
                .ok_or_else(|| anyhow::anyhow!(
                    "scheduler client is required for G4 onboard with connector_req"
                ))?;

            let handle = client.schedule_transfer(connector_req).await?;
            assert_eq!(
                handle.scheduler_decision(),
                crate::block_manager::connector::scheduler::SchedulingDecision::Execute
            );
            Some(handle)
        } else {
            None
        };

        // Execute G4->Host->Device transfer (Object->Host bounce->Device)
        // Host blocks for bounce buffers are provided by the leader
        let result = object_handler
            .onboard(
                &request.sequence_hashes,
                &request.host_block_ids,
                device_handle,
                &request.device_block_ids,
            )
            .await;

        // Note: The ObjectTransferHandler.onboard() already handles the H->D transfer
        // internally as part of its implementation, so we don't need a separate step.
        // The leader has already verified these hashes exist in the distributed registry.

        // Mark the transfer as complete (success or failure) to notify the connector scheduler
        if let Some(handle) = completion_handle {
            match &result {
                Ok(_) => handle.mark_complete(Ok(())).await,
                Err(e) => handle.mark_complete(Err(anyhow::anyhow!("{}", e))).await,
            }
        }

        let matched_hashes = result?;

        tracing::debug!(
            request_id = %request.request_id,
            operation_id = %request.operation_id,
            "G4 onboard complete: {} blocks",
            matched_hashes.len()
        );

        Ok(())
    }
}

#[derive(Builder, Clone)]
#[builder(pattern = "owned")]
pub struct KvbmWorkerConfig {
    cancel_token: CancellationToken,

    num_device_blocks: usize,

    #[builder(default = "32")]
    page_size: usize,

    #[builder(default = "Vec::new()")]
    tensors: Vec<Arc<dyn TorchTensor>>,

    #[builder(default = "0")]
    device_id: usize,

    #[builder(default = "2")]
    dtype_width_bytes: usize,

    #[builder(default = "LayoutType::FullyContiguous")]
    device_layout_type: LayoutType,

    #[builder(default = "LayoutType::FullyContiguous")]
    host_layout_type: LayoutType,

    #[builder(default = "LayoutType::FullyContiguous")]
    disk_layout_type: LayoutType,

    #[builder(default = "None")]
    scheduler_client: Option<TransferSchedulerClient>,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56001\")")]
    leader_pub_url: String,

    #[builder(default = "String::from(\"tcp://127.0.0.1:56002\")")]
    leader_ack_url: String,
}

impl KvbmWorkerConfig {
    pub fn builder() -> KvbmWorkerConfigBuilder {
        KvbmWorkerConfigBuilder::default()
    }
}

pub struct KvbmWorker {
    task: Option<CriticalTaskExecutionHandle>,
    block_transfer_handler_rx: Option<oneshot::Receiver<Arc<dyn BlockTransferHandler>>>,

    // G4 resources (created lazily when first needed)
    object_handler: OnceLock<Arc<ObjectTransferHandler>>,
    device_handle: OnceLock<LayoutHandle>,
    transfer_handler: OnceLock<Arc<dyn BlockTransferHandler>>,
}

impl KvbmWorker {
    pub async fn new(config: KvbmWorkerConfig, layout_blocking: bool) -> anyhow::Result<Self> {
        tracing::info!(
            "Initializing KvbmWorker with params: num_device_blocks={}, page_size={}, dtype_width_bytes={}",
            config.num_device_blocks,
            config.page_size,
            config.dtype_width_bytes
        );

        if config.num_device_blocks == 0 {
            return Err(anyhow::anyhow!("num_device_blocks must be greater than 0"));
        }

        let (device_tensors, shape) = load_and_validate_tensors(&config.tensors, config.device_id)?;

        if shape.len() < 3 {
            return Err(anyhow::anyhow!(format!(
                "Unsupported kv cache layout. Got shape: {:?}",
                shape
            )));
        }

        let (layout_type, num_layers, outer_dim, inner_dim) = match config.device_layout_type {
            LayoutType::FullyContiguous => {
                let num_layers = shape[1];
                let outer_dim = shape[2];
                let inner_dim = shape[3..].iter().product::<usize>() / config.page_size;
                tracing::info!(
                    "Inferred layout: num_layers={}, outer_dim={}, page_size={}, inner_dim={}",
                    num_layers,
                    outer_dim,
                    config.page_size,
                    inner_dim
                );

                (
                    LayoutType::FullyContiguous,
                    num_layers,
                    outer_dim,
                    inner_dim,
                )
            }
            LayoutType::LayerSeparate { outer_contiguous } => {
                // Use the already-detected layout type from config (no re-detection needed)
                let layout_type = config.device_layout_type;

                // Extract outer_dim based on the provided outer_contiguous value
                let outer_dim = if outer_contiguous {
                    shape[0] // Outer contiguous: [outer_dim, n_blocks, ...]
                } else {
                    shape[1] // Block contiguous: [n_blocks, outer_dim, ...]
                };

                let num_layers = device_tensors.len();
                let inner_dim = shape[2..].iter().product::<usize>() / config.page_size;

                tracing::info!(
                    "Inferred layout: num_layers={}, outer_dim={}, outer_contiguous={}, page_size={}, inner_dim={}",
                    num_layers,
                    outer_dim,
                    outer_contiguous,
                    config.page_size,
                    inner_dim
                );

                (layout_type, num_layers, outer_dim, inner_dim)
            }
        };

        let bytes_per_block =
            num_layers * outer_dim * config.page_size * inner_dim * config.dtype_width_bytes;

        let mut layout_builder_instance = LayoutConfigBuilder::default();
        let layout_builder = layout_builder_instance
            .num_layers(num_layers)
            .outer_dim(outer_dim)
            .page_size(config.page_size)
            .inner_dim(inner_dim)
            .dtype_width_bytes(config.dtype_width_bytes);

        let device_layout = layout_builder
            .num_blocks(config.num_device_blocks)
            .build()?
            .create_layout(layout_type, device_tensors)?;

        let layout_builder = layout_builder.clone();

        let (task, handler_rx) = if layout_blocking {
            Self::run_blocking_layout_initialization(
                config,
                bytes_per_block,
                device_layout,
                layout_builder,
                layout_type,
            )
            .await?
        } else {
            Self::run_non_blocking_layout_initialization(
                config,
                bytes_per_block,
                device_layout,
                layout_builder,
                layout_type,
            )
            .await?
        };

        Ok(Self {
            task: Some(task),
            block_transfer_handler_rx: Some(handler_rx),
            object_handler: OnceLock::new(),
            device_handle: OnceLock::new(),
            transfer_handler: OnceLock::new(),
        })
    }

    async fn run_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<Arc<dyn BlockTransferHandler>>,
    )> {
        let cancel_token = config.cancel_token.clone();

        // establish a oneshot channel to get back the raw BlockTransferHandler
        let (handler_tx, handler_rx) = oneshot::channel();
        let handler_tx_cell = Arc::new(Mutex::new(Some(handler_tx)));

        // establish a oneshot channel to block on the main routine to wait for layout allocation readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();
        let layout_ready_tx_cell = Mutex::new(Some(layout_ready_tx));

        let scheduler_client = config.scheduler_client.clone();

        let worker_config = config.clone();
        // start background worker task to do layout allocation for host or disk
        let task = CriticalTaskExecutionHandle::new(
            move |cancel_token| {
                KvbmWorker::worker_task(
                    device_layout,
                    layout_builder,
                    layout_type,
                    worker_config,
                    cancel_token,
                    handler_tx_cell,
                    layout_ready_tx_cell,
                    scheduler_client,
                    bytes_per_block,
                )
            },
            cancel_token.clone(),
            "kvbm-worker-task",
        )?;

        // waiting for the worker layout allocation ready
        match layout_ready_rx.await {
            Ok(_) => tracing::info!("worker layout allocation finished."),
            Err(_) => tracing::error!("Worker layout dropped without sending"),
        }

        Ok((task, handler_rx))
    }

    async fn run_non_blocking_layout_initialization(
        config: KvbmWorkerConfig,
        bytes_per_block: usize,
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage> + Send + 'static>,
        layout_builder: LayoutConfigBuilder,
        layout_type: LayoutType,
    ) -> anyhow::Result<(
        CriticalTaskExecutionHandle,
        oneshot::Receiver<Arc<dyn BlockTransferHandler>>,
    )> {
        let cancel_token = config.cancel_token.clone();
        let scheduler_client = config.scheduler_client.clone();

        // channel to get BlockTransferHandler back to the caller
        let (handler_tx, handler_rx) = oneshot::channel::<Arc<dyn BlockTransferHandler>>();
        let handler_tx_cell = Arc::new(Mutex::new(Some(handler_tx)));

        // channel that the worker will use to signal layout readiness
        let (layout_ready_tx, layout_ready_rx) = oneshot::channel::<String>();
        let layout_ready_tx_cell = Mutex::new(Some(layout_ready_tx));

        // clone what we need inside the orchestrator
        let worker_config = config.clone();
        let cancel_token_for_task = cancel_token.clone();

        // Single task that orchestrates everything in-order.
        let task = CriticalTaskExecutionHandle::new(
            move |ct| {
                let cfg = worker_config.clone();
                let scheduler = scheduler_client.clone();

                async move {
                    // Start the long-running worker.
                    let dev_layout = device_layout; // moved in
                    let lb = layout_builder; // moved in
                    let lt = layout_type; // moved in

                    let worker_fut = KvbmWorker::worker_task(
                        dev_layout,
                        lb,
                        lt,
                        cfg.clone(),
                        ct.clone(),
                        handler_tx_cell,
                        layout_ready_tx_cell,
                        scheduler,
                        bytes_per_block,
                    );

                    // If worker_task returns Result, handle/log it inside the spawned task.
                    tokio::spawn(async move {
                        if let Err(e) = worker_fut.await {
                            tracing::error!("worker_task exited with error: {e:#}");
                        }
                    });

                    // 3) wait for the worker’s layout allocation readiness
                    match layout_ready_rx.await {
                        Ok(_) => tracing::info!("worker layout allocation finished."),
                        Err(_) => tracing::warn!("worker layout readiness channel dropped"),
                    }

                    Ok::<(), anyhow::Error>(())
                }
            },
            cancel_token_for_task,
            "kvbm-worker-task",
        )?;

        Ok((task, handler_rx))
    }

    /// One-time use method to extract the block transfer handler from the worker.
    ///
    /// This is a bit of a hack. Improve the API design around this in the future.
    pub fn block_transfer_handler_rx(
        &mut self,
    ) -> Option<tokio::sync::oneshot::Receiver<Arc<dyn BlockTransferHandler>>> {
        self.block_transfer_handler_rx.take()
    }

    // ===== G4 Object Storage Methods =====

    /// Initialize G4 object storage support.
    ///
    /// Must be called after the transfer handler is ready.
    /// Returns Ok(true) if G4 was enabled, Ok(false) if not configured or not supported.
    pub async fn init_g4(
        &self,
        handler: Arc<dyn BlockTransferHandler>,
        config: ObjectStorageConfig,
    ) -> anyhow::Result<bool> {
        // Store the transfer handler
        let _ = self.transfer_handler.set(handler.clone());

        // Try to downcast to V2 handler
        let v2_handler = handler
            .as_any()
            .downcast_ref::<BlockTransferHandlerV2>()
            .ok_or_else(|| anyhow::anyhow!("G4 requires V2 transfer handler"))?;

        // Store device handle for offload/onboard
        if let Some(device_handle) = v2_handler.device_handle() {
            let _ = self.device_handle.set(device_handle);
        } else {
            return Err(anyhow::anyhow!("No device layout configured"));
        }

        // Create object handler using V2's resources
        let registry = Arc::new(ObjectRegistry::new());
        let distributed_registry = create_registry_from_env().await;
        let obj_handler = v2_handler
            .create_object_handler(config, registry, distributed_registry)
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "Failed to create object handler (host layout or V2 resources missing)"
                )
            })?;

        // Wrap in Arc to share between worker and V2 handler
        let obj_handler = Arc::new(obj_handler);

        // Wire up the object handler to V2 for write-through offloads
        v2_handler.set_object_handler(obj_handler.clone());

        self.object_handler
            .set(obj_handler)
            .map_err(|_| anyhow::anyhow!("Object handler already initialized"))?;

        tracing::info!("G4 object storage initialized on worker (write-through enabled)");
        Ok(true)
    }

    /// Check if G4 object storage is available.
    pub fn has_g4(&self) -> bool {
        self.object_handler.get().is_some()
    }

    /// Get the object transfer handler (if G4 is enabled).
    pub fn object_handler(&self) -> Option<&Arc<ObjectTransferHandler>> {
        self.object_handler.get()
    }

    /// Get the device handle (if G4 is enabled).
    pub fn g4_device_handle(&self) -> Option<LayoutHandle> {
        self.device_handle.get().copied()
    }

    /// Offload blocks to object storage.
    pub async fn offload_to_g4(
        &self,
        block_ids: &[usize],
        sequence_hashes: &[u64],
    ) -> anyhow::Result<usize> {
        let handler = self
            .object_handler
            .get()
            .ok_or_else(|| anyhow::anyhow!("G4 not initialized"))?;
        let device_handle = self
            .device_handle
            .get()
            .ok_or_else(|| anyhow::anyhow!("Device handle not available"))?;

        handler.offload(*device_handle, block_ids, sequence_hashes).await
    }

    /// Onboard blocks from object storage.
    ///
    /// # Arguments
    /// * `sequence_hashes` - Sequence hashes to onboard (object keys)
    /// * `host_block_ids` - Host block IDs to use as bounce buffers
    /// * `device_block_ids` - Destination device block IDs
    pub async fn onboard_from_g4(
        &self,
        sequence_hashes: &[u64],
        host_block_ids: &[usize],
        device_block_ids: &[usize],
    ) -> anyhow::Result<Vec<u64>> {
        let handler = self
            .object_handler
            .get()
            .ok_or_else(|| anyhow::anyhow!("G4 not initialized"))?;
        let device_handle = self
            .device_handle
            .get()
            .ok_or_else(|| anyhow::anyhow!("Device handle not available"))?;

        handler
            .onboard(sequence_hashes, host_block_ids, *device_handle, device_block_ids)
            .await
    }

    /// Lookup which hashes exist in object storage.
    pub fn lookup_g4(&self, sequence_hashes: &[u64]) -> Vec<u64> {
        self.object_handler
            .get()
            .map(|h| h.lookup(sequence_hashes))
            .unwrap_or_default()
    }

    fn make_layout<S: Storage, M: BlockMetadata>(
        mut layout: Box<dyn NixlLayout<StorageType = S>>,
        agent: &Option<NixlAgent>,
        block_set_idx: usize,
        worker_id: usize,
    ) -> anyhow::Result<Vec<Block<S, locality::Local, M>>> {
        // Register with NIXL, if applicable.
        if let Some(agent) = agent {
            layout.nixl_register(agent, None)?;
        }

        // Convert the layout into blocks.
        let layout: Arc<dyn NixlLayout<StorageType = S>> = Arc::from(layout);
        let blocks = layout_to_blocks::<_, M>(layout, block_set_idx, worker_id as u64)?;
        Ok(blocks)
    }

    #[allow(clippy::too_many_arguments)]
    async fn worker_task(
        device_layout: Box<dyn NixlLayout<StorageType = DeviceStorage>>,
        layout_builder: LayoutConfigBuilder,
        _device_layout_type: LayoutType,
        config: KvbmWorkerConfig,
        cancel_token: CancellationToken,
        handler_tx: Arc<TransferHandlerSender>,
        layout_ready_tx: tokio::sync::Mutex<Option<oneshot::Sender<String>>>,
        scheduler_client: Option<TransferSchedulerClient>,
        bytes_per_block: usize,
    ) -> anyhow::Result<()> {
        let worker_id = config.device_id;
        // Readiness gating for ping
        let state = Arc::new(WorkerState::new());

        // Cell to publish the transfer handler
        let transfer_handler_cell: Arc<RwLock<Option<Arc<dyn BlockTransferHandler>>>> =
            Arc::new(RwLock::new(None));

        // Build handlers map
        let mut handlers: HashMap<String, Arc<dyn Handler>> = HashMap::new();

        handlers.insert(
            ZMQ_PING_MESSAGE.to_string(),
            Arc::new(GatedPing {
                state: state.clone(),
                layout_ready_tx,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_WORKER_METADATA_MESSAGE.to_string(),
            Arc::new(WorkerMetadataHandler {
                num_device_blocks: config.num_device_blocks,
                bytes_per_block,
            }) as Arc<dyn Handler>,
        );

        handlers.insert(
            ZMQ_LEADER_METADATA_MESSAGE.to_string(),
            Arc::new(LeaderMetadataHandler {
                state: state.clone(),
                device_layout: tokio::sync::Mutex::new(Some(device_layout)), // moved in
                layout_builder,                                              // moved
                worker_config: config.clone(),
                worker_id,
                device_id: config.device_id,
                scheduler_client,
                handler_cell: transfer_handler_cell.clone(),
                handler_tx, // sends BlockTransferHandler to caller
                started: AtomicBool::new(false),
            }) as Arc<dyn Handler>,
        );

        // transfer requests get dispatched to built handler (after allocation)
        handlers.insert(
            ZMQ_TRANSFER_BLOCKS_MESSAGE.to_string(),
            Arc::new(BlockTransferDispatch {
                cell: transfer_handler_cell.clone(),
            }) as Arc<dyn Handler>,
        );

        // G4 onboard requests get dispatched to the same handler cell
        handlers.insert(
            ZMQ_G4_ONBOARD_MESSAGE.to_string(),
            Arc::new(G4OnboardDispatch {
                cell: transfer_handler_cell.clone(),
            }) as Arc<dyn Handler>,
        );

        let _zmq_worker = ZmqActiveMessageWorker::new(
            &config.leader_pub_url,
            &config.leader_ack_url,
            handlers,
            cancel_token.clone(),
        )?;

        // TODO: Some sort of fancy loop here.
        // For now, just wait for cancellation.
        cancel_token.cancelled().await;

        Ok(())
    }
}

impl Drop for KvbmWorker {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.cancel();
            task.detach();
        }
    }
}
