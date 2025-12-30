// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler output types shared between scheduler and connector.

use crate::v2::BlockId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data for a newly scheduled request that hasn't been seen before.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub req_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
}

/// Data for a cached request that was previously scheduled.
///
/// This represents a request that has been scheduled before and may have been
/// preempted. The `resumed` field indicates if it resumed from preemption,
/// and `all_token_ids` contains the full token sequence if resumed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub req_id: String,
    /// Whether this request resumed from preemption (derived from resumed_req_ids membership).
    pub resumed: bool,
    /// New token IDs added in this scheduling step.
    pub new_token_ids: Vec<u32>,
    /// All token IDs for the request (present only if resumed from preemption).
    pub all_token_ids: Option<Vec<u32>>,
    /// New block IDs allocated in this scheduling step.
    pub new_block_ids: Vec<BlockId>,
    /// Number of computed tokens for this request.
    pub num_computed_tokens: usize,
    /// Number of output tokens generated for this request.
    pub num_output_tokens: usize,
}

/// Scheduler output containing all requests scheduled in a single iteration.
///
/// This mirrors vLLM's `SchedulerOutput` structure with the updated API that uses
/// `resumed_req_ids` and `all_token_ids` instead of deprecated per-item fields.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SchedulerOutput {
    /// Iteration number
    pub iteration: usize,
    /// Requests scheduled for the first time.
    pub scheduled_new_reqs: Vec<NewRequestData>,
    /// Requests that have been scheduled before (may have been preempted).
    pub scheduled_cached_reqs: Vec<CachedRequestData>,
    /// Number of tokens scheduled for each request ID.
    pub num_scheduled_tokens: HashMap<String, usize>,
    /// Total number of tokens scheduled across all requests.
    pub total_num_scheduled_tokens: usize,
}

impl SchedulerOutput {
    /// Create a new empty SchedulerOutput.
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            ..Default::default()
        }
    }

    /// Add a new request to the output.
    pub fn add_new_request(
        &mut self,
        req_id: String,
        prompt_token_ids: Vec<u32>,
        block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
    ) {
        self.scheduled_new_reqs.push(NewRequestData {
            req_id,
            prompt_token_ids,
            block_ids,
            num_computed_tokens,
        });
    }

    /// Add a cached request to the output.
    ///
    /// # Arguments
    /// * `req_id` - The request ID
    /// * `resumed` - Whether this request resumed from preemption
    /// * `new_token_ids` - New token IDs added in this step
    /// * `all_token_ids` - All token IDs (if resumed, otherwise None)
    /// * `new_block_ids` - New block IDs allocated in this step
    /// * `num_computed_tokens` - Number of computed tokens
    /// * `num_output_tokens` - Number of output tokens generated
    #[allow(clippy::too_many_arguments)]
    pub fn add_cached_request(
        &mut self,
        req_id: String,
        resumed: bool,
        new_token_ids: Vec<u32>,
        all_token_ids: Option<Vec<u32>>,
        new_block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
        num_output_tokens: usize,
    ) {
        self.scheduled_cached_reqs.push(CachedRequestData {
            req_id,
            resumed,
            new_token_ids,
            all_token_ids,
            new_block_ids,
            num_computed_tokens,
            num_output_tokens,
        });
    }

    /// Set the number of scheduled tokens for each request.
    ///
    /// This also updates `total_num_scheduled_tokens` to be the sum of all values.
    pub fn set_num_scheduled_tokens(&mut self, num_scheduled_tokens: HashMap<String, usize>) {
        self.num_scheduled_tokens = num_scheduled_tokens;
        self.total_num_scheduled_tokens = self.num_scheduled_tokens.values().sum();
    }

    /// Get the total number of scheduled tokens.
    pub fn total_num_scheduled_tokens(&self) -> usize {
        self.total_num_scheduled_tokens
    }

    /// Get the number of scheduled tokens for a specific request.
    pub fn num_scheduled_tokens(&self, req_id: &str) -> Option<usize> {
        self.num_scheduled_tokens.get(req_id).copied()
    }

    /// Get an iterator over new requests.
    pub fn new_requests(&self) -> impl Iterator<Item = &NewRequestData> {
        self.scheduled_new_reqs.iter()
    }

    /// Get an iterator over cached requests.
    pub fn cached_requests(&self) -> impl Iterator<Item = &CachedRequestData> {
        self.scheduled_cached_reqs.iter()
    }
}
