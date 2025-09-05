// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::block_manager::{
    block::BlockId, connector::protocol::WorkerTransferRequest, distributed::BlockTransferRequest,
    pool::BlockPoolError,
};

pub mod leader;
pub mod trtllm_leader;
pub mod trtllm_worker;
pub mod worker;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::to_pyerr;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SchedulerOutput {
    // new requests - requests which have not been seen before
    pub new_requests: Vec<NewRequestData>,

    // cached requests - previously seen requests which could have been preempted
    pub cached_requests: Vec<CachedRequestData>,

    // scheduled tokens per request
    pub num_scheduled_tokens: HashMap<String, usize>,
}

#[pymethods]
impl SchedulerOutput {
    #[new]
    fn new() -> Self {
        Self {
            new_requests: Vec::new(),
            cached_requests: Vec::new(),
            num_scheduled_tokens: HashMap::new(),
        }
    }

    // I am surprised that vLLM's NewRequestData does not include the salt hash.
    // It has almost everything else to compute the block hashes worker side.
    pub fn add_new_request(
        &mut self,
        request_id: String,
        prompt_token_ids: Vec<u32>,
        block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
    ) {
        self.new_requests.push(NewRequestData {
            request_id,
            prompt_token_ids,
            block_ids,
            num_computed_tokens,
        });
    }

    /// This is called by the leader to update the cached requests
    pub fn add_cached_request(
        &mut self,
        request_id: String,
        resumed_from_preemption: bool,
        new_token_ids: Vec<u32>,
        new_block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
    ) {
        self.cached_requests.push(CachedRequestData {
            request_id,
            resumed_from_preemption,
            new_token_ids,
            new_block_ids,
            num_computed_tokens,
        });
    }

    /// This is called by the leader to update the number of scheduled tokens for a request
    pub fn add_num_scheduled_tokens(&mut self, num_scheduled_tokens: HashMap<String, usize>) {
        self.num_scheduled_tokens.clear();
        self.num_scheduled_tokens.extend(num_scheduled_tokens)
    }

    /// Use this to assert that the total number of scheduled tokens is correct
    /// Compare this to the value in in the vLLM SchedulerOutput
    pub fn get_num_scheduled_tokens(&self) -> usize {
        self.num_scheduled_tokens.values().sum()
    }

    pub fn serialize(&self) -> PyResult<Vec<u8>> {
        let bytes = serde_json::to_vec(self).map_err(to_pyerr)?;
        Ok(bytes)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
}

impl std::fmt::Debug for NewRequestData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NewRequestData")
            .field("request_id", &self.request_id)
            .field("num_tokens", &self.prompt_token_ids.len())
            .field("num_blocks", &self.block_ids.len())
            .field("num_computed_tokens", &self.num_computed_tokens)
            .finish()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub request_id: String,
    pub resumed_from_preemption: bool,
    pub new_token_ids: Vec<u32>,
    pub new_block_ids: Vec<BlockId>,
    pub num_computed_tokens: usize,
}

impl std::fmt::Debug for CachedRequestData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedRequestData")
            .field("request_id", &self.request_id)
            .field("resumed_from_preemption", &self.resumed_from_preemption)
            .field("num_new_tokens", &self.new_token_ids.len())
            .field("num_new_blocks", &self.new_block_ids.len())
            .field("num_computed_tokens", &self.num_computed_tokens)
            .finish()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorMetadata {
    /// The iteration at which the metadata was built.
    pub iteration: u64,

    /// The new slots that were created in this iteration.
    pub new_slots: Vec<String>,

    /// The operations that were initialized in this iteration.
    pub operations: Vec<WorkerTransferRequest>,
}

impl ConnectorMetadata {
    pub fn new(iteration: u64) -> Self {
        Self {
            iteration,
            new_slots: Vec::new(),
            operations: Vec::new(),
        }
    }

    pub fn create_slot(&mut self, request_id: String) {
        self.new_slots.push(request_id);
    }

    pub fn add_operations(&mut self, xfer_reqs: Vec<WorkerTransferRequest>) {
        self.operations.extend(xfer_reqs);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectorOperation {
    pub req_id: String,
    pub iteration: u64,
    pub uuid: uuid::Uuid,
    pub xfer_req: BlockTransferRequest,
}
