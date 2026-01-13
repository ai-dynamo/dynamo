// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! State module for the [`ConnectorWorker`]
//!
//! This module implements the state of the [`ConnectorWorker`] and the associated
//! state transitions.
//!
//! # Architecture
//!
//! The state is organized using interior mutability pattern with fine-grained locks:
//!
//! - `OnceLock` for one-time initialization (details, layout_config, service)
//! - `Mutex` for transient state (pending, forward_pass_events)
//! - `FinishedState` encapsulates finished tracking with its own internal lock

use anyhow::{Result, bail};
use cudarc::driver::CudaEvent;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::sync::{Arc, OnceLock};

use super::{FinishedRequests, init::PendingWorkerState};

use crate::{
    KvbmRuntime,
    distributed::worker::{LeaderLayoutConfig, NovaWorkerService, WorkerLayoutResponse},
    physical::layout::LayoutConfig,
};

/// Nova event handle for forward pass completion notification.
/// Stored separately from CUDA events since we now use pre-allocated per-layer events.
pub(crate) type ForwardPassNovaEvent = dynamo_nova::events::EventHandle;

/// Worker details set during KV cache registration.
pub(crate) struct WorkerDetails {
    pub num_layers: usize,
}

/// Encapsulates finished request tracking with interior mutability.
///
/// This struct provides thread-safe access to finished/failed request tracking
/// without requiring external locking.
pub(crate) struct FinishedState {
    inner: Mutex<FinishedStateInner>,
}

struct FinishedStateInner {
    finished_onboarding: HashSet<String>,
    finished_offloading: HashSet<String>,
    failed_onboarding: HashSet<usize>,
}

impl Default for FinishedState {
    fn default() -> Self {
        Self {
            inner: Mutex::new(FinishedStateInner {
                finished_onboarding: HashSet::new(),
                finished_offloading: HashSet::new(),
                failed_onboarding: HashSet::new(),
            }),
        }
    }
}

impl FinishedState {
    /// Mark a request as having completed onboarding.
    pub fn mark_onboarding_complete(&self, request_id: String) {
        self.inner.lock().finished_onboarding.insert(request_id);
    }

    /// Mark a request as having completed offloading.
    pub fn mark_offloading_complete(&self, request_id: String) {
        self.inner.lock().finished_offloading.insert(request_id);
    }

    /// Mark block IDs as having failed onboarding.
    pub fn mark_failed_onboarding(&self, block_ids: Vec<usize>) {
        self.inner.lock().failed_onboarding.extend(block_ids);
    }

    /// Take and drain all finished request IDs.
    ///
    /// Returns (finished_offloading, finished_onboarding) to match vLLM's API
    /// which expects (sending/saving ids, recving/loading ids).
    pub fn take_finished(&self) -> FinishedRequests {
        let mut inner = self.inner.lock();
        let finished_onboarding = std::mem::take(&mut inner.finished_onboarding);
        let finished_offloading = std::mem::take(&mut inner.finished_offloading);
        FinishedRequests {
            offloading: finished_offloading,
            onboarding: finished_onboarding,
        }
    }

    /// Take and drain all failed onboarding block IDs.
    pub fn take_failed_onboarding(&self) -> HashSet<usize> {
        std::mem::take(&mut self.inner.lock().failed_onboarding)
    }
}

/// Shared state for the connector worker.
///
/// Uses interior mutability pattern with fine-grained locks:
/// - `OnceLock` fields for one-time initialization
/// - `Mutex` fields for mutable state
/// - `FinishedState` for finished tracking (has its own internal lock)
///
/// This design allows the state to be shared via `Arc<WorkerState>` without
/// an outer Mutex, avoiding nested locking issues.
pub struct WorkerState {
    /// Reference to the runtime.
    runtime: Arc<KvbmRuntime>,

    // --- One-time initialization fields (OnceLock) ---
    /// Worker details (num_layers), set during KV cache registration.
    pub(crate) details: OnceLock<WorkerDetails>,

    /// Layout configuration, set during KV cache registration.
    layout_config: OnceLock<LayoutConfig>,

    /// Nova worker service, set when initialization completes.
    pub(crate) service: OnceLock<NovaWorkerService>,

    // --- Mutable state fields (Mutex) ---
    /// Pending state for deferred initialization.
    pending: Mutex<Option<PendingWorkerState>>,

    /// Nova event handle for forward pass completion notification.
    /// Set in `bind_connector_metadata`, consumed in `save_kv_layer` on last layer.
    pub(crate) forward_pass_nova_event: Mutex<Option<ForwardPassNovaEvent>>,

    // --- Pre-allocated CUDA events for layer-wise operations ---
    /// CUDA events for intra-pass G2→G1 onboarding, one per layer.
    /// Created during initialization and reused every iteration.
    /// Recorded on the transfer stream during start_load_kv,
    /// then consumed via cudaStreamWaitEvent in wait_for_layer_load.
    pub(crate) onboard_layer_events: OnceLock<Vec<Arc<CudaEvent>>>,

    /// CUDA events for layer-wise offloading, one per layer.
    /// Created during initialization and reused every iteration.
    /// Recorded on the torch stream during save_kv_layer,
    /// and represents the moment in time when the layer has been computed
    /// and is ready to be offloaded.
    /// The last layer event triggers Nova forward pass completion notification.
    pub(crate) compute_layer_events: OnceLock<Vec<Arc<CudaEvent>>>,

    /// Recorded on the offload stream when the last layer is complete.
    /// This event is then synchronously awaited by the workers in wait_for_save.
    pub(crate) offload_complete_event: OnceLock<Arc<CudaEvent>>,

    // --- Finished tracking (encapsulated with own lock) ---
    /// Tracks finished onboarding/offloading requests and failed blocks.
    pub(crate) finished_state: FinishedState,
}

impl WorkerState {
    /// Create a new WorkerState with the given runtime.
    pub fn new(runtime: Arc<KvbmRuntime>) -> Self {
        Self {
            runtime,
            details: OnceLock::new(),
            layout_config: OnceLock::new(),
            service: OnceLock::new(),
            pending: Mutex::new(None),
            forward_pass_nova_event: Mutex::new(None),
            onboard_layer_events: OnceLock::new(),
            compute_layer_events: OnceLock::new(),
            offload_complete_event: OnceLock::new(),
            finished_state: FinishedState::default(),
        }
    }

    /// Get a reference to the runtime.
    pub fn runtime(&self) -> &Arc<KvbmRuntime> {
        &self.runtime
    }

    /// Complete the deferred initialization.
    ///
    /// Worker initialization happens in multiple parts, on the worker side the following steps are taken:
    /// 1. Register the KV caches
    /// 2. Export the handshake metadata
    /// 3. Wait for the leader to call set xfer handshake metadata
    /// 4. During the set_xfer_handshake_metadata call, the leader will
    ///    - Call each worker and acquire a layout config
    ///    - The leader will determine how many blocks should be allocated for each tier
    ///    - The leader then triggers this `initialize` handler on each worker.
    pub(crate) fn initialize(&self, config: LeaderLayoutConfig) -> Result<WorkerLayoutResponse> {
        // Check if already initialized
        if self.service.get().is_some() {
            bail!("Worker already initialized");
        }

        // Take pending state
        let pending =
            self.pending.lock().take().ok_or_else(|| {
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

        // Pre-allocate CUDA events for layer-wise operations (onboarding and offloading)
        let num_layers = self
            .details
            .get()
            .ok_or_else(|| anyhow::anyhow!("Worker details not set"))?
            .num_layers;

        let transfer_manager = worker.transfer_manager();

        // Pre-allocate onboard events (H2D stream)
        let h2d_stream = transfer_manager.context().acquire_h2d_stream();
        let mut onboard_events = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let event = h2d_stream.record_event(None)?;
            onboard_events.push(Arc::new(event));
        }

        self.onboard_layer_events
            .set(onboard_events)
            .map_err(|_| anyhow::anyhow!("onboard_layer_events already set (race condition)"))?;

        // Pre-allocate save/offload events (D2H stream for consistency)
        let d2h_stream = transfer_manager.context().acquire_d2h_stream();
        let mut save_events = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            let event = d2h_stream.record_event(None)?;
            save_events.push(Arc::new(event));
        }

        self.compute_layer_events
            .set(save_events)
            .map_err(|_| anyhow::anyhow!("compute_layer_events already set (race condition)"))?;

        // Create the offload complete event to be awaited by the workers in wait_for_save.
        self.offload_complete_event
            .set(Arc::new(d2h_stream.record_event(None)?))
            .map_err(|_| anyhow::anyhow!("offload_complete_event already set (race condition)"))?;

        tracing::debug!(
            num_layers,
            "Pre-allocated layer events for intra-pass onboarding and offloading"
        );

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

    /// Set pending state during KV cache registration.
    pub(crate) fn set_pending_state(&self, pending_state: PendingWorkerState) -> Result<()> {
        let layout_config = pending_state.layout_config.clone();
        self.layout_config
            .set(layout_config)
            .map_err(|_| anyhow::anyhow!("layout config already set"))?;
        *self.pending.lock() = Some(pending_state);
        Ok(())
    }

    /// Check if pending state is already set.
    pub(crate) fn has_pending_state(&self) -> bool {
        self.pending.lock().is_some()
    }

    /// Get the pending state's layout config for handshake metadata.
    pub(crate) fn pending_layout_config(&self) -> Result<Vec<u8>> {
        let guard = self.pending.lock();
        match guard.as_ref() {
            Some(pending) => Ok(serde_json::to_vec(&pending.layout_config)?),
            None => bail!("No pending state - call register_kv_caches first"),
        }
    }

    /// Get the layout configuration.
    pub(crate) fn layout_config(&self) -> Result<LayoutConfig> {
        Ok(self
            .layout_config
            .get()
            .ok_or_else(|| anyhow::anyhow!("layout config not set"))?
            .clone())
    }

    /// Get the pre-allocated layer events for intra-pass onboarding.
    ///
    /// Returns a reference to the events if they have been allocated (during initialize),
    /// or an error if initialization hasn't completed yet.
    pub(crate) fn onboard_layer_events(&self) -> Result<&[Arc<CudaEvent>]> {
        self.onboard_layer_events
            .get()
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow::anyhow!("onboard_layer_events not initialized"))
    }

    /// Get the pre-allocated layer events for save/offload operations.
    ///
    /// Returns a reference to the events if they have been allocated (during initialize),
    /// or an error if initialization hasn't completed yet.
    pub(crate) fn compute_layer_events(&self) -> Result<&[Arc<CudaEvent>]> {
        self.compute_layer_events
            .get()
            .map(|v| v.as_slice())
            .ok_or_else(|| anyhow::anyhow!("compute_layer_events not initialized"))
    }

    /// Store the Nova event handle for forward pass completion.
    pub(crate) fn set_forward_pass_nova_event(&self, event: ForwardPassNovaEvent) {
        *self.forward_pass_nova_event.lock() = Some(event);
    }

    /// Take the Nova event handle for forward pass completion.
    /// Returns None if no event was set.
    pub(crate) fn take_forward_pass_nova_event(&self) -> Option<ForwardPassNovaEvent> {
        self.forward_pass_nova_event.lock().take()
    }

    /// Clear the Nova event handle without consuming it.
    #[expect(dead_code)]
    pub(crate) fn clear_forward_pass_nova_event(&self) {
        *self.forward_pass_nova_event.lock() = None;
    }

    // --- Delegate methods to FinishedState ---

    pub(crate) fn mark_onboarding_complete(&self, request_id: String) {
        self.finished_state.mark_onboarding_complete(request_id);
    }

    pub(crate) fn mark_offloading_complete(&self, request_id: String) {
        self.finished_state.mark_offloading_complete(request_id);
    }

    pub(crate) fn mark_failed_onboarding(&self, block_ids: Vec<usize>) {
        self.finished_state.mark_failed_onboarding(block_ids);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mark_failed_onboarding_adds_block_ids() {
        let state = FinishedState::default();

        state.mark_failed_onboarding(vec![1, 2, 3]);

        let failed = state.take_failed_onboarding();
        assert_eq!(failed.len(), 3);
        assert!(failed.contains(&1));
        assert!(failed.contains(&2));
        assert!(failed.contains(&3));
    }

    #[test]
    fn test_take_failed_onboarding_drains_set() {
        let state = FinishedState::default();

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
        let state = FinishedState::default();

        // First mark some blocks as failed
        state.mark_failed_onboarding(vec![5, 6, 7]);

        // Then mark onboarding as complete for a request
        state.mark_onboarding_complete("req-123".to_string());

        // Both should be retrievable
        let failed = state.take_failed_onboarding();
        assert_eq!(failed.len(), 3);

        // take_finished returns (onboarding, offloading)
        let (offloading, onboarding) = state.take_finished().dissolve();
        assert!(onboarding.contains("req-123"));
        assert!(offloading.is_empty());
    }

    #[test]
    fn test_mark_failed_onboarding_accumulates() {
        let state = FinishedState::default();

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
