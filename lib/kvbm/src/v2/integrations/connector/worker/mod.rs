// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! There are four action that the [`ConnectorWorker`] is allowed to perform:
//!
//! 1. Intra-Pass Onboarding
//!    - This action is triggered by the presence of an onboarding message from the leader and is detected
//!      while binding the [`KvConnectorMetadata`].
//!    - The presence of this action evaluated during the call to start_load_kv
//!    - The metadata will contain the G2 source block_ids and G1 destination block_ids.
//!    - Using a TransferManager CudaStream, each layer will be triggered with pre-defined set of events
//!      being recorded between each layer.
//!    - On wait_for_layer_load, the worker will inject a cuda stream wait event on the torch.cuda.current_stream()
//!      corresponding to the specific event recorded for the specific layer's onboard in start_load_kv.
//!    - CUDA will ensure that the the next attention layer will not start until the the onboarding for that layer
//!      is complete.
//! 2. Inter-Pass Onboarding
//!    - The NovaWorkerService performs this action via the wrapped DirectWorker.
//!    - This is performed out-of-band from the forward pass execution and is driven by the leader.
//!    - The completion of this action is another active message from the leader that updates the
//!      finished_state of the worker which which be observed via calls to get_finished.
//! 3. Issue Forward Pass Completion Notificaiton back to the leader
//!    - As part of the [`KvConnectorMetadata`], an optional ForwardPassCompletionEvent will be provided
//!      from the leader.
//!    - On binding the metadata, the action will be armed and triggered on the last call to save_kv_layer.
//!      - The arming of the action is the creation of CudaEvent on bindings
//!      - The triggering is to record the CudaEvent on the Torch CUDA stream on the last call to save_kv_layer;
//!        the event immediately pass to an async task which await on the completion, then triggers the Nova
//!        active message to trigger the EventHandle specific to the worker's rank back to the leader.
//!    - The ForwardPassCompletionEvent is used as a precondition for leader initiated action that require the
//!      forward pass completion event to be triggered.
//! 4. Perform direct layer-wise offloads
//!    - Future optimization for P/D offloading from prefill to decode.

mod init;
mod nova;
mod state;

pub use nova::client::ConnectorWorkerClient;

use init::PendingWorkerState;
use state::{WorkerDetails, WorkerState};

use anyhow::{Result, bail};
use cudarc::driver::sys::{
    CUevent, CUresult, CUstream, cuEventQuery, cuEventRecord, cuStreamWaitEvent, cudaError_enum,
};
use derive_getters::Dissolve;
use dynamo_memory::TensorDescriptor;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::KvbmRuntime;
use crate::v2::distributed::worker::DirectWorker;
use crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata;
use crate::v2::integrations::vllm::layout::determine_kv_layout;

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

    /// Start loading KV cache.
    ///
    /// If the bound metadata dictates that we should start loading KV cache,
    /// this function will trigger the loading of the KV cache.
    fn start_load_kv(&self) -> Result<()>;

    /// Save KV layer and trigger forward pass completion on last layer.
    ///
    /// Always callable - returns immediately if no action is needed for this layer.
    /// On the last layer, records a CUDA event and spawns a task to trigger
    /// the Nova forward pass completion event.
    ///
    /// # Arguments
    /// * `layer_index` - The layer index being saved
    /// * `stream_handle` - Raw CUDA stream handle (u64) from Python's current stream
    fn save_kv_layer(&self, layer_index: usize, stream_handle: u64) -> Result<()>;

    /// Wait for a specific layer's KV cache load to complete.
    ///
    /// If intra-pass onboarding was triggered in `start_load_kv`, this method
    /// inserts a `cudaStreamWaitEvent` on the provided torch stream to synchronize
    /// with the layer's onboard completion. This ensures the attention computation
    /// for this layer doesn't start until its KV cache data is available.
    ///
    /// # Arguments
    /// * `layer_index` - The layer index to wait for
    /// * `stream_handle` - Raw CUDA stream handle (u64) from Python's current torch stream
    fn wait_for_layer_load(&self, layer_index: usize, stream_handle: u64) -> Result<()>;

    /// Check if initialization has been completed.
    fn is_initialized(&self) -> bool;

    fn shutdown(&self) -> Result<()>;

    /// Get and drain all finished request IDs.
    fn get_finished(&self) -> FinishedRequests;

    /// Get and drain all failed onboarding block IDs.
    fn get_failed_onboarding(&self) -> HashSet<usize>;
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

/// Connector worker implementation that uses Nova for communication and
/// NIXL for RDMA transfers.
///
/// Uses deferred initialization by default:
/// - `register_kv_caches` only caches tensor state
/// - Leader calls `configure_layouts` RPC to complete initialization
pub struct ConnectorWorker {
    runtime: Arc<KvbmRuntime>,
    state: Arc<WorkerState>,
    metadata: Mutex<Option<KvConnectorMetadata>>,
    /// Flag indicating whether intra-pass onboarding is active for this iteration.
    /// Set to true in start_load_kv when metadata has intra_pass_load,
    /// used by wait_for_layer_load to decide whether to insert cudaStreamWaitEvent.
    intra_pass_onboard_active: Arc<AtomicBool>,
    /// Flag indicating whether forward pass completion notification is active.
    /// Set to true in bind_connector_metadata when a Nova event is present,
    /// used by save_kv_layer to decide whether to record events.
    forward_pass_completion_active: Arc<AtomicBool>,
    /// Flag for direct offloading (stub for future use, always false for now).
    is_direct_offloading_active: Arc<AtomicBool>,

    /// Start Forward Pass
    forward_pass_start: Mutex<Option<Instant>>,
}

impl ConnectorWorker {
    /// Create a new ConnectorWorker with deferred initialization.
    ///
    /// Registers the `kvbm.connector.configure_layouts` handler immediately
    /// so the leader can trigger initialization via RPC.
    pub fn new(runtime: Arc<KvbmRuntime>) -> Self {
        let nova = runtime.nova.clone();
        let state = Arc::new(WorkerState::new(Arc::clone(&runtime)));

        // Register handlers
        nova::service::init(&nova, Arc::clone(&state));

        Self {
            runtime,
            state,
            metadata: Mutex::new(None),
            intra_pass_onboard_active: Arc::new(AtomicBool::new(false)),
            forward_pass_completion_active: Arc::new(AtomicBool::new(false)),
            is_direct_offloading_active: Arc::new(AtomicBool::new(false)),
            forward_pass_start: Mutex::new(None),
        }
    }

    /// Get the DirectWorker if initialized.
    pub fn worker(&self) -> Option<&Arc<DirectWorker>> {
        self.state.service.get().map(|s| s.worker())
    }

    /// Get serialized handshake metadata for sending to leader.
    /// Returns the layout_config JSON bytes.
    pub fn handshake_metadata(&self) -> Result<Vec<u8>> {
        self.state.pending_layout_config()
    }

    #[cfg(test)]
    #[expect(dead_code)]
    pub(crate) fn runtime(&self) -> &Arc<KvbmRuntime> {
        self.state.runtime()
    }

    fn num_layers(&self) -> usize {
        self.state
            .details
            .get()
            .expect("details not set")
            .num_layers
    }

    /// Check if we need to perform any offload action for this layer.
    ///
    /// Returns true if:
    /// - Direct offloading is active (future - currently stubbed as false), OR
    /// - Forward pass completion is active AND this is the last layer
    fn needs_offload_action(&self, layer_index: usize) -> bool {
        // Stub for future direct offloading support
        let is_direct_offloading = self.is_direct_offloading_active.load(Ordering::Relaxed);

        // Forward pass completion triggers on last layer only
        let is_last_layer = layer_index == self.num_layers() - 1;
        let forward_pass_on_last =
            self.forward_pass_completion_active.load(Ordering::Relaxed) && is_last_layer;

        is_direct_offloading || forward_pass_on_last
    }

    /// Perform offload actions for the given layer.
    ///
    /// This is called from `save_kv_layer` when `needs_offload_action` returns true.
    /// Actions performed:
    /// - Record CUDA event on the stream for this layer
    /// - On last layer with forward pass completion: spawn task to trigger Nova event
    fn perform_offload_action(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        let is_last_layer = layer_index == self.num_layers() - 1;
        let forward_pass_active = self.forward_pass_completion_active.load(Ordering::Relaxed);

        // Get pre-allocated save layer events
        let layer_events = self.state.save_layer_events()?;

        // Validate layer_index
        if layer_index >= layer_events.len() {
            return Err(anyhow::anyhow!(
                "layer_index {} out of range (num_layers={})",
                layer_index,
                layer_events.len()
            ));
        }

        let event = &layer_events[layer_index];

        // Record CUDA event on the provided stream
        unsafe {
            let status = cuEventRecord(event.cu_event(), stream_handle as CUstream);
            if status != cudaError_enum::CUDA_SUCCESS {
                bail!("cuEventRecord failed with status: {:?}", status);
            }
        }

        tracing::trace!(layer_index, "Recorded save layer CUDA event");

        // If direct offloading is active, perform enqueue into a kvbm stream the event and the offload action
        // to take once the event is complete.
        // todo: add method to DirectWorker for this operation. - this will be a local operation from g1 -> g2
        // note: the operation might be a permute kernel if we are gathering or scattering kv to remote workers
        // with different tensor parallel world sizes.

        // On last layer with forward pass completion: spawn task to trigger Nova
        if is_last_layer && forward_pass_active {
            self.trigger_forward_pass_completion(event.clone())?;
        }

        Ok(())
    }

    /// Spawn async task to wait for CUDA event then trigger Nova forward pass event.
    fn trigger_forward_pass_completion(
        &self,
        cuda_event: Arc<cudarc::driver::CudaEvent>,
    ) -> Result<()> {
        // Take the Nova event handle
        let nova_event = self.state.take_forward_pass_nova_event().ok_or_else(|| {
            anyhow::anyhow!(
                "No Nova event handle - forward_pass_completion_active was true but no event set"
            )
        })?;

        let nova = self.runtime.nova().clone();
        let cuda_event_handle = cuda_event.cu_event() as u64;

        tracing::debug!(
            ?nova_event,
            cuda_event = cuda_event_handle,
            "Spawning forward pass completion task"
        );

        self.runtime.nova().tracker().spawn_on(
            async move {
                // Poll the CUDA event until complete
                loop {
                    let status = unsafe { cuEventQuery(cuda_event_handle as CUevent) };
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
            },
            &self.runtime.tokio(),
        );

        Ok(())
    }
}

impl ConnectorWorkerInterface for ConnectorWorker {
    #[tracing::instrument(level = "debug", skip_all, fields(instance_id = ?self.runtime.nova.instance_id()))]
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
        if self.state.has_pending_state() {
            bail!("KV caches already pending registration");
        }
        if self.state.details.get().is_some() {
            bail!("Worker details already set");
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
        let pending = PendingWorkerState::builder()
            .tensors(tensors)
            .num_device_blocks(num_device_blocks)
            .page_size(page_size)
            .dtype_width_bytes(dtype_width_bytes)
            .layout_config(layout_config)
            .block_dim(block_dim)
            .build()?;

        let details = WorkerDetails {
            num_layers: pending.tensors.len(),
        };

        tracing::info!(
            cuda_device = pending.cuda_device_id,
            num_tensors = pending.tensors.len(),
            num_device_blocks,
            page_size,
            dtype_width_bytes,
            "KV caches registered (deferred mode - waiting for leader RPC)"
        );

        self.state.set_pending_state(pending)?;

        self.state
            .details
            .set(details)
            .map_err(|_| anyhow::anyhow!("details already set"))?;

        Ok(())
    }

    fn bind_connector_metadata(&self, metadata: KvConnectorMetadata) -> Result<()> {
        tracing::debug!(iteration = metadata.iteration, "Binding connector metadata");

        // Store Nova event handle if present (we use pre-allocated CUDA events now)
        if let Some(event_map) = &metadata.foward_pass_completion_events {
            let my_instance_id = self.state.runtime().nova().instance_id();

            if let Some(&nova_event) = event_map.get(&my_instance_id) {
                tracing::debug!(?nova_event, "Storing forward pass Nova event");
                self.state.set_forward_pass_nova_event(nova_event);
                self.forward_pass_completion_active
                    .store(true, Ordering::Relaxed);
            }

            tracing::info!("Binding connector metadata: {:?}", metadata.summary());
        }

        // Store the metadata for use by start_load_kv
        *self.metadata.lock() = Some(metadata);

        Ok(())
    }

    fn clear_connector_metadata(&self) -> Result<()> {
        tracing::debug!("Clearing connector metadata");

        // Verify that Nova event has been consumed by save_kv_layer (on last layer)
        if self.state.take_forward_pass_nova_event().is_some() {
            // This could happen if there was an error during forward pass
            // or if no layers were processed. Log but don't fail.
            tracing::trace!(
                "Forward pass Nova event not consumed - save_kv_layer may not have been called on last layer"
            );
        }

        // Clear metadata and reset atomic flags
        *self.metadata.lock() = None;
        self.intra_pass_onboard_active
            .store(false, Ordering::Relaxed);
        self.forward_pass_completion_active
            .store(false, Ordering::Relaxed);
        self.is_direct_offloading_active
            .store(false, Ordering::Relaxed);

        Ok(())
    }

    fn start_load_kv(&self) -> Result<()> {
        // Check if metadata has intra-pass load request
        let intra_pass_load = {
            let mut metadata_guard = self.metadata.lock();
            metadata_guard
                .as_mut()
                .and_then(|m| m.intra_pass_load.take())
        };

        *self.forward_pass_start.lock() = Some(Instant::now());

        if let Some(load) = intra_pass_load {
            tracing::debug!(
                g2_blocks = load.g2_src_block_ids.len(),
                g1_blocks = load.g1_dst_block_ids.len(),
                "Starting intra-pass layer-wise onboard from G2 to G1"
            );

            // Get the DirectWorker
            let worker = self
                .worker()
                .ok_or_else(|| anyhow::anyhow!("Worker not initialized"))?;

            // Get pre-allocated layer events
            let layer_events = self.state.onboard_layer_events()?;

            // Execute layer-wise onboard
            worker.execute_local_layerwise_onboard(
                &load.g2_src_block_ids,
                &load.g1_dst_block_ids,
                layer_events,
            )?;

            // Set flag so wait_for_layer_load knows to sync
            self.intra_pass_onboard_active
                .store(true, Ordering::Relaxed);

            tracing::debug!("Intra-pass onboard initiated - events recorded on transfer stream");
        }

        Ok(())
    }

    fn wait_for_layer_load(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        // Only insert wait if intra-pass onboarding is active
        if !self.intra_pass_onboard_active.load(Ordering::Relaxed) {
            return Ok(());
        }

        // Get the pre-allocated layer events
        let layer_events = self.state.onboard_layer_events()?;

        // Validate layer_index
        if layer_index >= layer_events.len() {
            return Err(anyhow::anyhow!(
                "layer_index {} out of range (num_layers={})",
                layer_index,
                layer_events.len()
            ));
        }

        let event = &layer_events[layer_index];

        // Insert cudaStreamWaitEvent to make torch stream wait for this layer's onboard
        unsafe {
            let status = cuStreamWaitEvent(
                stream_handle as CUstream,
                event.cu_event(),
                0, // flags = 0
            );
            if status != cudaError_enum::CUDA_SUCCESS {
                bail!("cuStreamWaitEvent failed with status: {:?}", status);
            }
        }

        tracing::trace!(
            layer_index,
            "Inserted cudaStreamWaitEvent for layer onboard sync"
        );

        Ok(())
    }

    fn save_kv_layer(&self, layer_index: usize, stream_handle: u64) -> Result<()> {
        // Early return if no action needed for this layer
        if !self.needs_offload_action(layer_index) {
            return Ok(());
        }

        // Perform the offload action(s)
        self.perform_offload_action(layer_index, stream_handle)
    }

    fn is_initialized(&self) -> bool {
        self.state.service.get().is_some()
    }

    fn shutdown(&self) -> Result<()> {
        tracing::info!("Connector worker shutdown");
        Ok(())
    }

    /// Get and drain all finished request IDs.
    ///
    /// When [`FinishedRequests::dissolve`] is called, the returned tuple will be (offloading, onboarding).
    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn get_finished(&self) -> FinishedRequests {
        self.state.finished_state.take_finished()
    }

    #[tracing::instrument(level = "debug", skip(self), ret)]
    fn get_failed_onboarding(&self) -> HashSet<usize> {
        self.state.finished_state.take_failed_onboarding()
    }
}

#[derive(Default, Debug, Clone, Dissolve)]
pub struct FinishedRequests {
    pub offloading: HashSet<String>,
    pub onboarding: HashSet<String>,
}
