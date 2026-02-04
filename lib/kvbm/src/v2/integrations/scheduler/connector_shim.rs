// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Connector shim for Rust scheduler integration.
//!
//! This module provides a thin wrapper around [`ConnectorLeader`] that manages
//! slot lifecycle automatically, mirroring the pattern used by Python's
//! `SchedulerConnectorLeader` in `lib/bindings/kvbm/python/kvbm/v2/vllm/schedulers/leader.py`.
//!
//! # Slot Lifecycle
//!
//! The shim handles three key operations:
//!
//! 1. **Slot Creation**: Automatically creates a connector slot on the first
//!    call to `get_num_new_matched_tokens` for a request.
//!
//! 2. **Token Sync**: Provides `extend_slot_tokens` to sync new tokens before
//!    `build_connector_meta` (mirrors Python's `update_slot()` hack).
//!
//! 3. **Slot Deletion**: Deletes the connector slot when `request_finished`
//!    is called.
//!
//! # Usage
//!
//! ```ignore
//! use dynamo_kvbm::v2::integrations::scheduler::SchedulerConnectorShim;
//!
//! let connector = Arc::new(ConnectorLeader::new(runtime, block_size));
//! let shim = SchedulerConnectorShim::new(connector);
//!
//! // In schedule_new_request:
//! let matched = shim.get_num_new_matched_tokens(&request, num_computed)?;
//!
//! // After block allocation:
//! shim.update_state_after_alloc(&request.req_id, &block_ids, num_external)?;
//!
//! // On request finish:
//! let hold_blocks = shim.request_finished(&request.req_id)?;
//! ```

use super::request::SchedulerRequest;
use crate::v2::integrations::connector::leader::{
    BlockBoundaryInfo, ConnectorLeader, EvictionScore, FinishedStatus, SchedulerOutput,
};
use crate::v2::BlockId;

use anyhow::Result;
use parking_lot::RwLock;
use std::collections::HashSet;
use std::sync::Arc;

/// Shim that wraps ConnectorLeader for Rust scheduler use.
///
/// Mirrors Python's `SchedulerConnectorLeader` pattern:
/// - Auto-creates slots on first `get_num_new_matched_tokens` call
/// - Tracks inflight requests by ID
/// - Deletes slots on `request_finished`
///
/// The shim takes references to [`SchedulerRequest`] to extract request data
/// for slot creation, similar to how Python passes vLLM Request objects.
pub struct SchedulerConnectorShim {
    /// The underlying ConnectorLeader.
    leader: Arc<ConnectorLeader>,

    /// Tracked inflight request IDs that have slots.
    /// We only track which requests have slots, not the request data itself.
    inflight: RwLock<HashSet<String>>,
}

impl SchedulerConnectorShim {
    /// Create a new connector shim wrapping the given ConnectorLeader.
    pub fn new(leader: Arc<ConnectorLeader>) -> Self {
        Self {
            leader,
            inflight: RwLock::new(HashSet::new()),
        }
    }

    /// Get a reference to the underlying ConnectorLeader.
    pub fn leader(&self) -> &Arc<ConnectorLeader> {
        &self.leader
    }

    /// Check if a slot exists for the given request ID.
    pub fn has_slot(&self, request_id: &str) -> bool {
        self.inflight.read().contains(request_id)
    }

    /// Create slot for request if it doesn't exist.
    ///
    /// Called automatically by `get_num_new_matched_tokens`.
    fn ensure_slot(&self, request: &SchedulerRequest) -> Result<()> {
        let request_id = &request.request.request_id;

        // Fast path: already have slot
        if self.inflight.read().contains(request_id) {
            return Ok(());
        }

        // Create slot using the Request from SchedulerRequest
        // Clone the request data for the connector slot
        let connector_request = request.request.clone_without_metadata();

        self.leader.create_slot(connector_request)?;
        self.inflight.write().insert(request_id.clone());

        tracing::debug!(
            request_id = %request_id,
            "Created connector slot for request"
        );

        Ok(())
    }

    // =========================================================================
    // Connector API (delegates to leader)
    // =========================================================================

    /// Check for external KV cache matches.
    ///
    /// Auto-creates a connector slot on first call for this request.
    ///
    /// # Returns
    ///
    /// A tuple of:
    /// - `Option<usize>`: Number of matched tokens, or `None` if still searching
    /// - `bool`: Whether async loading is in progress (inter-pass mode)
    pub fn get_num_new_matched_tokens(
        &self,
        request: &SchedulerRequest,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        self.ensure_slot(request)?;
        self.leader
            .get_num_new_matched_tokens(&request.request.request_id, num_computed_tokens)
    }

    /// Notify connector after block allocation.
    ///
    /// Called after the scheduler allocates blocks to a request.
    /// The connector tracks block mappings for offload operations.
    pub fn update_state_after_alloc(
        &self,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        self.leader
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
    }

    /// Sync new tokens to slot before `build_connector_meta`.
    ///
    /// Mirrors Python's `update_slot()` hack for vLLM. Called when new tokens
    /// have been generated and need to be synchronized to the connector slot.
    pub fn extend_slot_tokens(&self, request_id: &str, tokens: Vec<u32>) -> Result<()> {
        self.leader.extend_slot_tokens(request_id, tokens)
    }

    /// Build connector metadata for this scheduling iteration.
    ///
    /// Called at the end of `schedule()` to produce metadata that workers
    /// need for model execution (forward pass events, intra-pass loads).
    pub fn build_connector_meta(
        &self,
        output: SchedulerOutput,
    ) -> Result<crate::v2::integrations::connector::leader::scheduler::KvConnectorMetadata> {
        self.leader.build_connector_meta(output)
    }

    /// Mark request as finished, delete slot.
    ///
    /// # Returns
    ///
    /// A [`FinishedStatus`] indicating:
    /// - `Finished`: Blocks can be freed immediately
    /// - `Pending`: Blocks must be held until `finished_sending` signal
    /// - `UntrackedRequest`: No slot existed for this request
    pub fn request_finished(&self, request_id: &str) -> FinishedStatus {
        // Remove from our tracking regardless of connector state
        self.inflight.write().remove(request_id);

        // Delegate to leader (which handles slot cleanup)
        let status = self.leader.request_finished(request_id);

        tracing::debug!(
            request_id = %request_id,
            ?status,
            "Request finished, connector slot cleaned up"
        );

        status
    }

    /// Process connector output signals.
    ///
    /// Called with signals from the model execution phase:
    /// - `finished_sending`: Offloads complete, blocks safe to free
    /// - `finished_recving`: Async loads complete, requests can continue
    ///
    /// Delegates to [`ConnectorLeader::update_connector_output`] which:
    /// - For `finished_recving`: Releases onboarding sessions
    /// - For `finished_sending`: Verifies offload handles are complete
    pub fn update_connector_output(
        &self,
        finished_sending: HashSet<String>,
        finished_recving: HashSet<String>,
    ) -> Result<()> {
        if !finished_sending.is_empty() || !finished_recving.is_empty() {
            tracing::debug!(
                finished_sending = finished_sending.len(),
                finished_recving = finished_recving.len(),
                "Processing connector output signals"
            );
        }
        self.leader
            .update_connector_output(finished_sending, finished_recving)
    }

    // =========================================================================
    // Eviction Support
    // =========================================================================

    /// Check if a request can be safely evicted.
    ///
    /// Returns `false` if request has inflight offloads (RDMA transfers).
    pub fn can_evict(&self, request_id: &str) -> bool {
        self.leader.can_evict(request_id)
    }

    /// Get eviction score for ranking candidates.
    ///
    /// Higher score = better eviction candidate (more G2 coverage).
    pub fn get_eviction_score(&self, request_id: &str) -> Result<EvictionScore> {
        self.leader.get_eviction_score(request_id)
    }

    /// Get block boundary alignment information for a request.
    pub fn get_block_boundary_info(&self, request_id: &str) -> Result<BlockBoundaryInfo> {
        self.leader.get_block_boundary_info(request_id)
    }

    // =========================================================================
    // Projection System Support
    // =========================================================================

    /// Request priority offload for blocks planned for eviction.
    pub fn request_priority_offload(
        &self,
        request_id: &str,
        block_ids: &[BlockId],
    ) -> Result<usize> {
        self.leader.request_priority_offload(request_id, block_ids)
    }

    /// Get per-block G2 status for a request.
    pub fn get_block_g2_status(
        &self,
        request_id: &str,
    ) -> Result<std::collections::HashMap<BlockId, bool>> {
        self.leader.get_block_g2_status(request_id)
    }
}

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;

    // Note: Full integration tests would require a ConnectorLeader,
    // which needs KvbmRuntime. For now, we just document the expected behavior.
    //
    // The key behaviors to test:
    // 1. ensure_slot creates slot on first call, no-op on subsequent calls
    // 2. request_finished removes from inflight tracking
    // 3. All methods properly delegate to ConnectorLeader
}
