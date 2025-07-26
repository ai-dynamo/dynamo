// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod leader;
pub mod worker;

use std::collections::HashMap;

use dynamo_llm::block_manager::block::BlockId;
use pyo3::{prelude::*, wrap_pymodule};
use serde::{Deserialize, Serialize};

use crate::to_pyerr;

pub struct KvConnectorMetadata {}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[pyclass]
pub struct SchedulerOutput {
    // new requests - requests which have not been seen before
    pub new_requests: Vec<NewRequestData>,

    // cached requests - previously seen requests which could have been preempted
    pub cached_requests: Vec<CachedRequestData>,

    // scheduled tokens per request
    pub num_scheduled_tokens: HashMap<String, u64>,
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
        num_computed_tokens: u32,
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
        num_computed_tokens: u32,
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
    pub fn add_num_scheduled_tokens(&mut self, num_scheduled_tokens: HashMap<String, u64>) {
        self.num_scheduled_tokens.extend(num_scheduled_tokens)
    }

    /// Use this to assert that the total number of scheduled tokens is correct
    /// Compare this to the value in in the vLLM SchedulerOutput
    pub fn get_num_scheduled_tokens(&self) -> u64 {
        self.num_scheduled_tokens.values().sum()
    }

    pub fn serialize(&self) -> PyResult<Vec<u8>> {
        let bytes = serde_json::to_vec(self).map_err(to_pyerr)?;
        Ok(bytes)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewRequestData {
    pub request_id: String,
    pub prompt_token_ids: Vec<u32>,
    pub block_ids: Vec<BlockId>,
    pub num_computed_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CachedRequestData {
    pub request_id: String,
    pub resumed_from_preemption: bool,
    pub new_token_ids: Vec<u32>,
    pub new_block_ids: Vec<BlockId>,
    pub num_computed_tokens: u32,
}
