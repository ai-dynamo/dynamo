// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_kvbm::integrations::connector::leader::scheduler::SchedulerOutput as RustSchedulerOutput;
use dynamo_kvbm::v2::BlockId;

use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::to_pyerr;

/// Python wrapper for SchedulerOutput.
///
/// This provides a PyO3 interface to the Rust SchedulerOutput struct,
/// enabling Python code to build scheduler outputs that can be passed
/// to the connector leader's build_connector_meta method.
#[pyclass(name = "SchedulerOutput")]
#[derive(Clone, Serialize, Deserialize)]
pub struct PySchedulerOutput {
    inner: RustSchedulerOutput,
}

#[pymethods]
impl PySchedulerOutput {
    #[new]
    fn new(iteration: usize) -> Self {
        Self {
            inner: RustSchedulerOutput::new(iteration),
        }
    }

    /// Add a new request to the scheduler output.
    ///
    /// Args:
    ///     req_id: The request ID
    ///     prompt_token_ids: Optional prompt token IDs
    ///     block_ids: Block IDs allocated for this request
    ///     num_computed_tokens: Number of computed tokens
    #[pyo3(signature = (req_id, *, prompt_token_ids, block_ids, num_computed_tokens))]
    pub fn add_new_request(
        &mut self,
        req_id: String,
        prompt_token_ids: Vec<u32>,
        block_ids: Vec<BlockId>,
        num_computed_tokens: usize,
    ) {
        self.inner
            .add_new_request(req_id, prompt_token_ids, block_ids, num_computed_tokens);
    }

    /// Add a cached request to the scheduler output.
    ///
    /// Args:
    ///     req_id: The request ID
    ///     resumed: Whether this request resumed from preemption
    ///     new_token_ids: New token IDs added in this step
    ///     all_token_ids: All token IDs (if resumed, otherwise None)
    ///     new_block_ids: New block IDs allocated in this step
    ///     num_computed_tokens: Number of computed tokens
    ///     num_output_tokens: Number of output tokens generated
    #[pyo3(signature = (req_id, resumed, new_token_ids, *, all_token_ids = None, new_block_ids, num_computed_tokens, num_output_tokens))]
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
        self.inner.add_cached_request(
            req_id,
            resumed,
            new_token_ids,
            all_token_ids,
            new_block_ids,
            num_computed_tokens,
            num_output_tokens,
        );
    }

    /// Set the number of scheduled tokens for each request.
    ///
    /// Args:
    ///     num_scheduled_tokens: Dictionary mapping request ID to number of scheduled tokens
    pub fn set_num_scheduled_tokens(&mut self, num_scheduled_tokens: HashMap<String, usize>) {
        self.inner.set_num_scheduled_tokens(num_scheduled_tokens);
    }

    /// Get the total number of scheduled tokens.
    pub fn get_total_num_scheduled_tokens(&self) -> usize {
        self.inner.total_num_scheduled_tokens()
    }

    /// Serialize the scheduler output to JSON bytes.
    pub fn serialize(&self) -> PyResult<Vec<u8>> {
        let bytes = serde_json::to_vec(&self.inner).map_err(to_pyerr)?;
        Ok(bytes)
    }
}

impl PySchedulerOutput {
    /// Get a reference to the inner Rust SchedulerOutput.
    pub fn inner(&self) -> RustSchedulerOutput {
        self.inner.clone()
    }
}
