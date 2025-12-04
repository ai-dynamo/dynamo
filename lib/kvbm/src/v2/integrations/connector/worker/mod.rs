// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod client;
mod pending;

use bytes::Bytes;
pub use client::ConnectorWorkerClient;
pub use pending::PendingWorkerState;

use anyhow::{Result, bail};
use dynamo_memory::TensorDescriptor;
use dynamo_nova::Nova;
use dynamo_nova::am::NovaHandler;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::{Arc, Mutex, OnceLock};

use crate::KvbmRuntime;
use crate::v2::distributed::worker::{
    DirectWorker, LeaderLayoutConfig, NovaWorkerService, WorkerLayoutResponse,
};
use crate::v2::integrations::vllm::layout::determine_kv_layout;

pub const ONBOARD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.onboard_complete";
pub const OFFLOAD_COMPLETE_HANDLER: &str = "kvbm.connector.worker.offload_complete";
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

    // /// Complete NIXL initialization with leader-provided config.
    // fn complete_initialization(&self, config: LeaderLayoutConfig) -> Result<WorkerLayoutResponse>;

    /// Check if initialization has been completed.
    fn is_initialized(&self) -> bool;

    fn shutdown(&self) -> Result<()>;

    /// Get and drain all finished request IDs.
    fn get_finished(&self) -> (HashSet<String>, HashSet<String>);
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

/// Shared state for the connector worker, wrapped in Arc for handler sharing.
struct SharedWorkerState {
    runtime: Arc<KvbmRuntime>,
    pending_state: Mutex<Option<PendingWorkerState>>,
    service: OnceLock<NovaWorkerService>,
    gpu_info: OnceLock<GpuInfo>,
    finished_state: FinishedState,
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
        let offload_state = state;
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

    fn is_initialized(&self) -> bool {
        self.state.service.get().is_some()
    }

    fn shutdown(&self) -> Result<()> {
        tracing::info!("Connector worker shutdown");
        Ok(())
    }

    fn get_finished(&self) -> (HashSet<String>, HashSet<String>) {
        self.state.finished_state.take_finished()
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
