// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Request state and lifecycle management.

use super::kv_cache::RequestBlockState;
use crate::v2::integrations::common::Request;
use crate::v2::logical::blocks::{ImmutableBlock, MutableBlock};
use crate::v2::BlockId;
use crate::G1;

/// Status of a request in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    /// Request is waiting to be scheduled.
    Waiting,
    /// Request is currently running (scheduled for this iteration).
    Running,
    /// Request was preempted due to memory pressure.
    Preempted,
    /// Request finished normally (hit stop token or max tokens).
    FinishedStopped,
    /// Request was aborted (cancelled by user or error).
    FinishedAborted,
    /// Request finished due to reaching length limit.
    FinishedLengthCapped,
}

impl RequestStatus {
    /// Returns true if the request is in a finished state.
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            RequestStatus::FinishedStopped
                | RequestStatus::FinishedAborted
                | RequestStatus::FinishedLengthCapped
        )
    }

    /// Returns true if the request can be scheduled.
    pub fn can_schedule(&self) -> bool {
        matches!(self, RequestStatus::Waiting | RequestStatus::Preempted)
    }
}

/// Internal scheduler representation of a request.
///
/// This struct tracks the block allocations for a request using RAII guards.
/// The `block_state` holds both pending (mutable) and registered (immutable)
/// blocks, managing their lifecycle automatically.
pub struct SchedulerRequest {
    /// The original request data.
    pub request: Request,

    /// Current status of the request.
    pub status: RequestStatus,

    /// RAII block state for this request.
    ///
    /// Contains both pending blocks (allocated but not yet filled with token data)
    /// and registered blocks (completed and in the cache).
    pub block_state: RequestBlockState,

    /// Number of tokens that have been computed (KV cache filled).
    pub num_computed_tokens: usize,

    /// Number of output tokens generated so far.
    pub num_output_tokens: usize,

    /// Whether this request was just resumed from preemption.
    /// Reset to false after being scheduled once.
    pub resumed_from_preemption: bool,
}

impl SchedulerRequest {
    /// Create a new scheduler request from a request.
    pub fn new(request: Request) -> Self {
        Self {
            request,
            status: RequestStatus::Waiting,
            block_state: RequestBlockState::new(),
            num_computed_tokens: 0,
            num_output_tokens: 0,
            resumed_from_preemption: false,
        }
    }

    /// Get the request ID.
    pub fn request_id(&self) -> &str {
        &self.request.request_id
    }

    /// Get the total number of tokens in the prompt.
    pub fn prompt_len(&self) -> usize {
        self.request.tokens.len()
    }

    /// Get the total number of tokens (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_len() + self.num_output_tokens
    }

    /// Get the number of tokens that still need to be computed.
    pub fn num_tokens_to_compute(&self) -> usize {
        self.total_tokens().saturating_sub(self.num_computed_tokens)
    }

    /// Get the number of blocks required for the current token count.
    pub fn num_blocks_required(&self, block_size: usize) -> usize {
        (self.total_tokens() + block_size - 1) / block_size
    }

    /// Get the number of new blocks needed (beyond what's already allocated).
    pub fn num_new_blocks_needed(&self, block_size: usize) -> usize {
        self.num_blocks_required(block_size)
            .saturating_sub(self.block_state.total_blocks())
    }

    /// Check if the request has reached its maximum token limit.
    pub fn is_at_max_tokens(&self) -> bool {
        if let Some(max) = self.request.max_tokens {
            self.num_output_tokens >= max
        } else {
            false
        }
    }

    /// Get the block IDs allocated to this request.
    ///
    /// Returns all block IDs (both pending and registered).
    pub fn block_ids(&self) -> Vec<BlockId> {
        self.block_state.all_block_ids()
    }

    /// Add pending (mutable) blocks to this request.
    pub fn add_pending_blocks(&mut self, blocks: Vec<MutableBlock<G1>>) {
        self.block_state.add_pending(blocks);
    }

    /// Add registered (immutable) blocks to this request.
    pub fn add_registered_blocks(&mut self, blocks: Vec<ImmutableBlock<G1>>) {
        self.block_state.add_registered(blocks);
    }

    /// Take pending blocks out of this request.
    ///
    /// Used when transitioning blocks to registered after a forward pass.
    pub fn take_pending_blocks(&mut self) -> Vec<MutableBlock<G1>> {
        self.block_state.take_pending()
    }

    /// Transition the request to running state.
    pub fn start_running(&mut self) {
        self.status = RequestStatus::Running;
    }

    /// Preempt the request, releasing all blocks.
    ///
    /// All RAII blocks are dropped, returning them to the appropriate pools.
    pub fn preempt(&mut self) {
        self.status = RequestStatus::Preempted;
        // Clear blocks - RAII returns them to pools
        self.block_state.clear();
        // Reset computed tokens since blocks are freed
        self.num_computed_tokens = 0;
    }

    /// Resume the request from preemption.
    pub fn resume(&mut self) {
        debug_assert_eq!(self.status, RequestStatus::Preempted);
        self.status = RequestStatus::Waiting;
        self.resumed_from_preemption = true;
    }

    /// Finish the request with the given status.
    ///
    /// All RAII blocks are dropped, returning them to the appropriate pools.
    pub fn finish(&mut self, status: RequestStatus) {
        debug_assert!(status.is_finished());
        self.status = status;
        // Clear blocks - RAII returns them to pools
        self.block_state.clear();
    }

    /// Add output tokens after a forward pass.
    pub fn add_output_tokens(&mut self, num_tokens: usize) {
        self.num_output_tokens += num_tokens;
    }

    /// Update the number of computed tokens after a forward pass.
    pub fn update_computed_tokens(&mut self, num_computed: usize) {
        self.num_computed_tokens = num_computed;
    }

    /// Clear the resumed flag (called after scheduling).
    pub fn clear_resumed_flag(&mut self) {
        self.resumed_from_preemption = false;
    }
}

impl std::fmt::Debug for SchedulerRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SchedulerRequest")
            .field("request_id", &self.request.request_id)
            .field("status", &self.status)
            .field("block_state", &self.block_state)
            .field("num_computed_tokens", &self.num_computed_tokens)
            .field("num_output_tokens", &self.num_output_tokens)
            .field("resumed_from_preemption", &self.resumed_from_preemption)
            .finish()
    }
}
