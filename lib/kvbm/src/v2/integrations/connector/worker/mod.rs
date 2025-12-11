// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod pending;

use bytes::Bytes;
pub use client::ConnectorWorkerClient;
pub use pending::PendingWorkerState;

use anyhow::{Result, bail};
use cudarc::driver::sys::{
    CUevent, CUevent_flags, CUresult, CUstream, cuEventCreate, cuEventQuery, cuEventRecord,
    cudaError_enum,
};
use dynamo_memory::TensorDescriptor;
use dynamo_nova::Nova;
use dynamo_nova::am::NovaHandler;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::ptr;
use std::sync::{Arc, Mutex, OnceLock};

use crate::KvbmRuntime;
use crate::v2::BlockId;
use crate::v2::distributed::worker::{
    DirectWorker, LeaderLayoutConfig, NovaWorkerService, WorkerLayoutResponse,
};
use crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata;
use crate::v2::integrations::vllm::layout::determine_kv_layout;

pub const ONBOARD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.onboard_complete";
pub const OFFLOAD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.offload_complete";
pub const FAILED_ONBOARD_HANDLER: &str = "kvbm.connector.worker.failed_onboard";
pub const GET_LAYOUT_CONFIG_HANDLER: &str = "kvbm.connector.worker.get_layout_config";
pub const INITIALIZE_HANDLER: &str = "kvbm.connector.worker.initialize";

pub trait ConnectorWorkerInterface: Send + Sync {
    /// Register KV cache tensors (deferred mode - caches state for later).
    fn register_kv_caches(
        &self,
        tensors: Vec<Arc<dyn TensorDescriptor>>,
        num_device_blocks: usize,
        page_size: usize,
        dtype_width_bytes: usize,
    ) -> Result<()>;

    /// Bind connector metadata from the leader.
    fn bind_connector_metadata(&self, metadata: KvConnectorMetadata) -> Result<()>;

    /// Clear connector metadata.
    fn clear_connector_metadata(&self) -> Result<()>;

    /// Check if we need a CUDA stream for event synchronization.
    ///
    /// Returns `true` if there's a pending forward pass event that needs
    /// to be synchronized via CUDA event before triggering.
    fn needs_cuda_stream(&self) -> bool;

    /// Save KV layer and trigger forward pass completion on last layer.
    ///
    /// This should be called on the last layer's save_kv_layer.
    /// It records the pre-created CUDA event on the provided stream,
    /// then spawns an async task that waits for the event and triggers
    /// the Nova forward pass event.
    ///
    /// # Arguments
    /// * `stream_handle` - Raw CUDA stream handle (u64) from Python's current stream
    fn save_kv_layer(&self, stream_handle: u64) -> Result<()>;

    /// Check if initialization has been completed.
    fn is_initialized(&self) -> bool;

    fn shutdown(&self) -> Result<()>;

    /// Get and drain all finished request IDs.
    fn get_finished(&self) -> (HashSet<String>, HashSet<String>);

    /// Get and drain all failed onboarding block IDs.
    fn get_failed_onboarding(&self) -> HashSet<usize>;
}

/// Tracks completed operations for worker-side reporting.
///
/// The leader populates this via `mark_onboarding_complete()` and
/// `mark_offloading_complete()` after detecting that transfers have finished.
/// The worker executor drains via `take_finished()` to report completed requests.
#[derive(Default, Debug)]
pub struct FinishedState {
    finished_onboarding: Mutex<HashSet<String>>,
    finished_offloading: Mutex<HashSet<String>>,
    failed_onboarding: Mutex<HashSet<usize>>,
}

impl FinishedState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn mark_onboarding_complete(&self, request_id: String) {
        self.finished_onboarding.lock().unwrap().insert(request_id);
    }

    pub fn mark_offloading_complete(&self, request_id: String) {
        self.finished_offloading.lock().unwrap().insert(request_id);
    }

    pub fn take_finished(&self) -> (HashSet<String>, HashSet<String>) {
        let offloading = std::mem::take(&mut *self.finished_offloading.lock().unwrap());
        let onboarding = std::mem::take(&mut *self.finished_onboarding.lock().unwrap());
        (offloading, onboarding)
    }

    pub fn mark_failed_onboarding(&self, block_ids: Vec<BlockId>) {
        self.failed_onboarding.lock().unwrap().extend(block_ids);
    }

    pub fn take_failed_onboarding(&self) -> HashSet<usize> {
        std::mem::take(&mut *self.failed_onboarding.lock().unwrap())
    }
}

/// Message sent by leader to workers when onboarding completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OnboardCompleteMessage {
    pub request_id: String,
}

/// Message sent by leader to workers when offloading completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OffloadCompleteMessage {
    pub request_id: String,
}

/// Message sent by leader to workers when onboarding fails for specific blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FailedOnboardMessage {
    pub request_id: String,
    pub block_ids: Vec<BlockId>,
}

/// GPU information for the worker, used for logging and debugging.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    pub device_index: usize,
    pub uuid: Option<String>,
}

impl GpuInfo {
    pub fn from_device_index(device_index: usize) -> Self {
        Self {
            device_index,
            uuid: None, // TODO: Implement UUID retrieval via NVML if needed
        }
    }
}

/// Paired forward pass events: Nova event handle + CUDA event for synchronization.
///
/// These are created together in `bind_connector_metadata` when a forward pass
/// event is present. The CUDA event is recorded on the stream in `save_kv_layer`,
/// then an async task waits for the CUDA event before triggering the Nova event.
struct ForwardPassEvents {
    nova_event: dynamo_nova::events::EventHandle,
    /// Raw CUDA event handle stored as u64 for Send/Sync safety
    cuda_event: u64,
}

/// Shared state for the connector worker, wrapped in Arc for handler sharing.
struct SharedWorkerState {
    runtime: Arc<KvbmRuntime>,
    pending_state: Mutex<Option<PendingWorkerState>>,
    service: OnceLock<NovaWorkerService>,
    gpu_info: OnceLock<GpuInfo>,
    finished_state: FinishedState,
    /// Current iteration's forward pass events (Nova + CUDA pair).
    /// Created in `bind_connector_metadata`, consumed in `save_kv_layer`.
    forward_pass_events: Mutex<Option<ForwardPassEvents>>,
}

/// Connector worker implementation that uses Nova for communication and
/// NIXL for RDMA transfers.
///
/// Uses deferred initialization by default:
/// - `register_kv_caches` only caches tensor state
/// - Leader calls `configure_layouts` RPC to complete initialization
pub struct ConnectorWorker {
    state: Arc<SharedWorkerState>,
}

impl ConnectorWorker {
    /// Create a new ConnectorWorker with deferred initialization.
    ///
    /// Registers the `kvbm.connector.configure_layouts` handler immediately
    /// so the leader can trigger initialization via RPC.
    pub fn new(runtime: Arc<KvbmRuntime>) -> Self {
        let state = Arc::new(SharedWorkerState {
            runtime,
            pending_state: Mutex::new(None),
            service: OnceLock::new(),
            gpu_info: OnceLock::new(),
            finished_state: FinishedState::new(),
            forward_pass_events: Mutex::new(None),
        });

        let nova = state.runtime.nova.clone();

        // Register handlers
        Self::register_initialize_handler(&nova, Arc::clone(&state));
        Self::register_completion_handlers(&nova, Arc::clone(&state));
        Self::register_get_layout_config_handler(&nova, Arc::clone(&state));

        Self { state }
    }

    /// Get the DirectWorker if initialized.
    pub fn worker(&self) -> Option<&Arc<DirectWorker>> {
        self.state.service.get().map(|s| s.worker())
    }

    /// Get serialized handshake metadata for sending to leader.
    /// Returns the layout_config JSON bytes.
    pub fn handshake_metadata(&self) -> Result<Vec<u8>> {
        let guard = self.state.pending_state.lock().unwrap();
        match guard.as_ref() {
            Some(pending) => Ok(serde_json::to_vec(&pending.layout_config)?),
            None => bail!("No pending state - call register_kv_caches first"),
        }
    }

    #[cfg(test)]
    #[expect(dead_code)]
    pub(crate) fn runtime(&self) -> &Arc<KvbmRuntime> {
        &self.state.runtime
    }

    /// Register the configure_layouts handler for leader-driven initialization.
    ///
    /// This handler is called by the leader after collecting handshake metadata.
    /// It completes NIXL registration and creates G1/G2/G3 layouts.
    fn register_initialize_handler(nova: &Arc<Nova>, state: Arc<SharedWorkerState>) {
        let handler = NovaHandler::typed_unary_async(INITIALIZE_HANDLER, move |ctx| {
            let state = Arc::clone(&state);

            async move {
                let config: LeaderLayoutConfig = ctx.input;
                state.initialize(config)
            }
        })
        .build();

        if let Err(e) = nova.register_handler(handler) {
            tracing::error!("Failed to register configure_layouts handler: {}", e);
        }
    }

    /// Register completion handlers for leader notifications.
    fn register_completion_handlers(nova: &Arc<Nova>, state: Arc<SharedWorkerState>) {
        // Handler: "kvbm.connector.worker.onboard_complete"
        let onboard_state = Arc::clone(&state);
        let onboard_handler =
            NovaHandler::typed_unary_async(ONBOARD_COMPLETE_HANDLER, move |ctx| {
                let state = Arc::clone(&onboard_state);
                async move {
                    let msg: OnboardCompleteMessage = ctx.input;
                    tracing::debug!(request_id = %msg.request_id, "Worker received onboard complete");
                    state
                        .finished_state
                        .mark_onboarding_complete(msg.request_id);
                    Ok(())
                }
            })
            .build();

        if let Err(e) = nova.register_handler(onboard_handler) {
            tracing::error!("Failed to register onboard_complete handler: {}", e);
        }

        // Handler: "kvbm.connector.worker.offload_complete"
        let offload_state = Arc::clone(&state);
        let offload_handler =
            NovaHandler::typed_unary_async(OFFLOAD_COMPLETE_HANDLER, move |ctx| {
                let state = Arc::clone(&offload_state);
                async move {
                    let msg: OffloadCompleteMessage = ctx.input;
                    tracing::debug!(request_id = %msg.request_id, "Worker received offload complete");
                    state
                        .finished_state
                        .mark_offloading_complete(msg.request_id);
                    Ok(())
                }
            })
            .build();

        if let Err(e) = nova.register_handler(offload_handler) {
            tracing::error!("Failed to register offload_complete handler: {}", e);
        }

        // Handler: "kvbm.connector.worker.failed_onboard"
        let failed_state = state;
        let failed_handler = NovaHandler::typed_unary_async(FAILED_ONBOARD_HANDLER, move |ctx| {
            let state = Arc::clone(&failed_state);
            async move {
                let msg: FailedOnboardMessage = ctx.input;
                tracing::debug!(
                    request_id = %msg.request_id,
                    block_ids = ?msg.block_ids,
                    "Worker received failed onboard notification"
                );
                state.finished_state.mark_failed_onboarding(msg.block_ids);
                Ok(())
            }
        })
        .build();

        if let Err(e) = nova.register_handler(failed_handler) {
            tracing::error!("Failed to register failed_onboard handler: {}", e);
        }
    }

    fn register_get_layout_config_handler(nova: &Arc<Nova>, state: Arc<SharedWorkerState>) {
        let handler = NovaHandler::unary_handler_async(GET_LAYOUT_CONFIG_HANDLER, move |_ctx| {
            let state = Arc::clone(&state);
            async move {
                let guard = state.pending_state.lock().unwrap();
                match guard.as_ref() {
                    Some(pending) => {
                        let config = pending.layout_config.clone();
                        Ok(Some(Bytes::from(serde_json::to_vec(&config)?)))
                    }
                    None => bail!("No pending state - call register_kv_caches first"),
                }
            }
        })
        .build();

        if let Err(e) = nova.register_handler(handler) {
            tracing::error!("Failed to register get_layout_config handler: {}", e);
        }
    }
}

impl ConnectorWorkerInterface for ConnectorWorker {
    #[tracing::instrument(level = "debug", skip_all, fields(instance_id = ?self.state.runtime.nova.instance_id()))]
    fn register_kv_caches(
        &self,
        tensors: Vec<Arc<dyn TensorDescriptor>>,
        num_device_blocks: usize,
        page_size: usize,
        dtype_width_bytes: usize,
    ) -> Result<()> {
        // Prevent double registration
        if self.state.service.get().is_some() {
            bail!("KV caches already registered");
        }
        if self.state.pending_state.lock().unwrap().is_some() {
            bail!("KV caches already pending registration");
        }

        // Determine layout from tensor shapes
        let (layout_config, block_dim) =
            determine_kv_layout(num_device_blocks, page_size, dtype_width_bytes, &tensors)?;

        tracing::debug!(
            ?layout_config,
            ?block_dim,
            "Determined KV layout configuration"
        );

        // Create pending state (validates tensors internally)
        let pending = PendingWorkerState::new(
            tensors,
            num_device_blocks,
            page_size,
            dtype_width_bytes,
            layout_config,
            block_dim,
        )?;

        // Store GPU info for logging
        let _ = self.state.gpu_info.set(pending.gpu_info.clone());

        tracing::info!(
            cuda_device = pending.cuda_device_id,
            gpu_uuid = ?pending.gpu_info.uuid,
            num_tensors = pending.tensors.len(),
            num_device_blocks,
            page_size,
            dtype_width_bytes,
            "KV caches registered (deferred mode - waiting for leader RPC)"
        );

        *self.state.pending_state.lock().unwrap() = Some(pending);

        Ok(())
    }

    fn bind_connector_metadata(&self, metadata: KvConnectorMetadata) -> Result<()> {
        tracing::debug!(iteration = metadata.iteration, "Binding connector metadata");

        // Load forward pass event if present and create paired CUDA event
        if let Some(event_map) = metadata.forward_pass_events {
            let my_instance_id = self.state.runtime.nova().instance_id();

            if let Some(&nova_event) = event_map.get(&my_instance_id) {
                // Create a CUDA event for synchronization (disabled timing for performance)
                let cuda_event: u64 = unsafe {
                    let mut event: CUevent = ptr::null_mut();
                    let status =
                        cuEventCreate(&mut event, CUevent_flags::CU_EVENT_DISABLE_TIMING as u32);
                    if status != cudaError_enum::CUDA_SUCCESS {
                        bail!("cuEventCreate failed with status: {:?}", status);
                    }
                    event as u64
                };

                tracing::debug!(
                    ?nova_event,
                    cuda_event,
                    "Created paired forward pass events"
                );

                *self.state.forward_pass_events.lock().unwrap() = Some(ForwardPassEvents {
                    nova_event,
                    cuda_event,
                });
            }
        }

        Ok(())
    }

    fn clear_connector_metadata(&self) -> Result<()> {
        tracing::debug!("Clearing connector metadata");

        // Verify that forward pass events have been consumed by save_kv_layer
        let events = self.state.forward_pass_events.lock().unwrap();
        if events.is_some() {
            tracing::warn!(
                "Forward pass events not consumed - save_kv_layer may not have been called on last layer"
            );
            // Don't bail here - this could happen if there was an error during forward pass
            // The events will be cleaned up on next bind_connector_metadata
        }

        Ok(())
    }

    fn needs_cuda_stream(&self) -> bool {
        self.state.forward_pass_events.lock().unwrap().is_some()
    }

    fn save_kv_layer(&self, stream_handle: u64) -> Result<()> {
        // Take the paired events
        let events = self
            .state
            .forward_pass_events
            .lock()
            .unwrap()
            .take()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "No forward pass events - needs_cuda_stream() should have returned false"
                )
            })?;

        tracing::debug!(
            ?events.nova_event,
            cuda_event = events.cuda_event,
            stream_handle,
            "Recording CUDA event on stream and spawning completion task"
        );

        // Record the CUDA event on the provided stream
        unsafe {
            let status = cuEventRecord(events.cuda_event as CUevent, stream_handle as CUstream);
            if status != cudaError_enum::CUDA_SUCCESS {
                bail!("cuEventRecord failed with status: {:?}", status);
            }
        }

        // Spawn async task to wait for CUDA event then trigger Nova event
        let nova = self.state.runtime.nova().clone();
        let nova_event = events.nova_event;
        let cuda_event = events.cuda_event;

        self.state.runtime.nova().tracker().spawn_on(
            async move {
                // Poll the CUDA event until complete
                loop {
                    let status = unsafe { cuEventQuery(cuda_event as CUevent) };
                    match status {
                        CUresult::CUDA_SUCCESS => break,
                        CUresult::CUDA_ERROR_NOT_READY => {
                            // Yield to other tasks
                            tokio::task::yield_now().await;
                        }
                        _ => {
                            tracing::error!("CUDA event query failed: {:?}", status);
                            break;
                        }
                    }
                }

                // Trigger the Nova forward pass event
                tracing::debug!(?nova_event, "CUDA event complete, triggering Nova event");
                if let Err(e) = nova.events().trigger(nova_event).await {
                    tracing::error!("Failed to trigger forward pass event: {}", e);
                }

                // Note: We don't call cuEventDestroy here because it will be handled
                // when the task completes. The event was created in bind_connector_metadata
                // and ownership was transferred to this task.
                // TODO: Consider explicit cleanup with cuEventDestroy
            },
            &self.state.runtime.tokio(),
        );

        Ok(())
    }

    fn is_initialized(&self) -> bool {
        self.state.service.get().is_some()
    }

    fn shutdown(&self) -> Result<()> {
        tracing::info!("Connector worker shutdown");
        Ok(())
    }

    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn get_finished(&self) -> (HashSet<String>, HashSet<String>) {
        self.state.finished_state.take_finished()
    }

    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn get_failed_onboarding(&self) -> HashSet<usize> {
        self.state.finished_state.take_failed_onboarding()
    }
}

impl SharedWorkerState {
    fn initialize(&self, config: LeaderLayoutConfig) -> Result<WorkerLayoutResponse> {
        // Check if already initialized
        if self.service.get().is_some() {
            bail!("Worker already initialized");
        }

        // Take pending state
        let pending =
            self.pending_state.lock().unwrap().take().ok_or_else(|| {
                anyhow::anyhow!("No pending state - call register_kv_caches first")
            })?;

        tracing::info!(
            cuda_device = pending.cuda_device_id,
            host_block_count = config.host_block_count,
            disk_block_count = ?config.disk_block_count,
            "Completing deferred NIXL initialization"
        );

        // Complete initialization
        let (worker, response) = pending.complete_initialization(&self.runtime, config)?;

        // Build NovaWorkerService
        let service = NovaWorkerService::new(self.runtime.nova.clone(), worker)?;

        self.service
            .set(service)
            .map_err(|_| anyhow::anyhow!("service already initialized (race condition)"))?;

        tracing::info!(
            created_layouts = ?response.created_layouts,
            "Deferred initialization complete - NIXL registered"
        );

        Ok(response)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_failed_onboarding_adds_block_ids() {
        let state = FinishedState::new();

        state.mark_failed_onboarding(vec![1, 2, 3]);

        let failed = state.take_failed_onboarding();
        assert_eq!(failed.len(), 3);
        assert!(failed.contains(&1));
        assert!(failed.contains(&2));
        assert!(failed.contains(&3));
    }

    #[test]
    fn test_take_failed_onboarding_drains_set() {
        let state = FinishedState::new();

        state.mark_failed_onboarding(vec![10, 20]);

        // First take should return the block IDs
        let first_take = state.take_failed_onboarding();
        assert_eq!(first_take.len(), 2);
        assert!(first_take.contains(&10));
        assert!(first_take.contains(&20));

        // Second take should return empty set
        let second_take = state.take_failed_onboarding();
        assert!(second_take.is_empty());
    }

    #[test]
    fn test_failed_onboarding_before_complete() {
        // Verifies that failed blocks can be marked before marking completion
        // This matches the ordering guarantee in the implementation
        let state = FinishedState::new();

        // First mark some blocks as failed
        state.mark_failed_onboarding(vec![5, 6, 7]);

        // Then mark onboarding as complete for a request
        state.mark_onboarding_complete("req-123".to_string());

        // Both should be retrievable
        let failed = state.take_failed_onboarding();
        assert_eq!(failed.len(), 3);

        let (offloading, onboarding) = state.take_finished();
        assert!(offloading.is_empty());
        assert!(onboarding.contains("req-123"));
    }

    #[test]
    fn test_mark_failed_onboarding_accumulates() {
        let state = FinishedState::new();

        // Mark multiple batches of failed blocks
        state.mark_failed_onboarding(vec![1, 2]);
        state.mark_failed_onboarding(vec![3, 4, 5]);

        let failed = state.take_failed_onboarding();
        assert_eq!(failed.len(), 5);
        for id in 1..=5 {
            assert!(failed.contains(&id));
        }
    }
}
