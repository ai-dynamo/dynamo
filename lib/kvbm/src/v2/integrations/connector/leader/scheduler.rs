// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::ConnectorLeader;
use crate::{
    G1,
    distributed::offload::{ExternalBlock, SourceBlock},
    v2::BlockId,
};

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Instant};

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

pub struct IterationSession {
    pub iteration: usize,
    pub created: Instant,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvConnectorMetadata {
    pub iteration: usize,
}

pub struct ForwardPassBuilder {
    pub iteration: usize,
}

impl ConnectorLeader {
    /// Process the scheduler output and return the connector metadata.
    ///
    /// On each iteration that has a at least one total_num_scheduled_tokens, we will create a IterationSession
    /// and before returning, register the IterationSession with the [`ConnectorLeader`] and spawn a tracking task
    /// to wait for the IterationSession to be completed.
    pub fn process_scheduler_output(
        &self,
        scheduler_output: &SchedulerOutput,
    ) -> Result<KvConnectorMetadata> {
        // todo: early exit if the number of scheduled tokens is 0
        // - we do not register the session
        if scheduler_output.total_num_scheduled_tokens == 0 {
            tracing::debug!("no scheduled tokens, early exiting");
            return Ok(KvConnectorMetadata::new(scheduler_output.iteration));
        }

        // create a forward pass builder, this will be used to apply actions to the forward pass
        // - we will generate from this object a session and a metadata object
        // - the session we will register with the leader so the worker can interact with it via nova
        // - the metadata object will be serialized and sent to the workers so they know what actions to perform
        let _builder = ForwardPassBuilder::new(scheduler_output.iteration);

        for req in &scheduler_output.scheduled_new_reqs {
            match self.get_slot(&req.req_id) {
                Ok(shared_slot) => {
                    // thompson sampling to determine if we should offload any blocks for this request
                    // - todo: need to consider the what variables we should sample on
                    // determine if we should offload any blocks for this request
                    // - if the request max_tokens is 1 or very small, we don't have a lot of time to offload
                    //   and not put memory pressure on the device memory pool (request may finish before offload finishes)
                    // - if the request is nearing max_seq_lenght, the chances of it being used against being do diminish
                    //   as agents will generally need to re-write the context and start over - we may get more aggressive
                    //   and demote blocks that achieve this threshold to a lower priority offload / free-list
                    // match self.oracle.evaluate_new_request(req) {
                    //     Ok(_) => {
                    //         tracing::debug!("new_request_data: {:#?}", req);
                    //         // todo!("update fp builder")
                    //     }
                    //     Err(_) => {
                    //         tracing::warn!("failed to evaluate new request_data: {:?}", req);
                    //         continue;
                    //     }
                    // }

                    // we should dump all the blocks for this request into the offload engine and let the offload engine
                    // and it's respective policies decide whether or not to offload them.
                    let slot = shared_slot.lock();

                    if slot.is_marked_for_deletion() {
                        tracing::warn!("slot is not marked for deletion, skipping offload");
                        continue;
                    }

                    let block_count = req.block_ids.len();
                    assert_eq!(req.prompt_token_ids.len(), slot.sequence.total_tokens());

                    let token_blocks = slot
                        .sequence
                        .blocks()
                        .get(0..block_count).expect("block count must be less than or equal to the total number of blocks in the sequence");

                    let sequence_hashes = token_blocks
                        .iter()
                        .map(|b| b.positional_sequence_hash())
                        .collect::<Vec<_>>();

                    let source_blocks = req
                        .block_ids
                        .iter()
                        .zip(sequence_hashes)
                        .map(|(block_id, sequence_hash)| {
                            ExternalBlock::<G1>::new(*block_id, sequence_hash)
                        })
                        .collect::<Vec<_>>();

                    let tranfer_handle = self
                        .offload_engine
                        .get()
                        .unwrap()
                        .enqueue_g1_to_g2(source_blocks);

                    //slot.offload_blocks(&sequence_hashes, transfer_handle);
                }
                Err(_) => {
                    tracing::warn!(
                        "unexpected event: slot not found for request id: {}",
                        req.req_id
                    );
                    continue;
                }
            }
        }

        for req in &scheduler_output.scheduled_cached_reqs {
            tracing::debug!("cached_request_data: {:#?}", req);
        }

        Ok(KvConnectorMetadata::new(scheduler_output.iteration))
    }
}

impl IterationSession {
    pub fn new(iteration: usize) -> Self {
        Self {
            iteration,
            created: Instant::now(),
        }
    }
}

impl KvConnectorMetadata {
    pub fn new(iteration: usize) -> Self {
        Self { iteration }
    }
}

impl ForwardPassBuilder {
    pub fn new(iteration: usize) -> Self {
        Self { iteration }
    }
}

pub trait Oracle: Send + Sync {
    // Evaluate the new request and determine if we should offload any blocks
    // This must be a fast local decision to avoid slowing down the scheduler
    // A positive result here means we put the returne blocks into an offload
    // engine, which is out-of-band from the scheduler and can take into account
    // more global information.
    fn evaluate_new_request(&self, req: &NewRequestData) -> Result<()>;
}

#[derive(Default, Debug)]
pub struct DefaultOracle {}

impl Oracle for DefaultOracle {
    fn evaluate_new_request(&self, _req: &NewRequestData) -> Result<()> {
        Ok(())
    }
}
