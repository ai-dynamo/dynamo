// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for the Rust scheduler.
//!
//! This module provides PyO3 wrappers around the real Rust scheduler from
//! `dynamo_kvbm::v2::integrations::scheduler`. The bindings are thin wrappers
//! that delegate all logic to the real implementation.
//!
//! # Architecture
//!
//! ```text
//! PyScheduler (thin wrapper)
//!     └── inner: Scheduler (real implementation from kvbm)
//!             └── kv_cache: KVCacheManager
//!                     └── block_manager: BlockManager<G1>
//!                             └── RAII MutableBlock/ImmutableBlock with real block_ids
//! ```
//!
//! # Block Management
//!
//! Block IDs come from the real `BlockManager<G1>` - they are NOT made up.
//! RAII guards (`MutableBlock`, `ImmutableBlock`) manage block lifecycle automatically.

pub mod config;
pub mod status;

pub use config::PySchedulerConfig;
pub use status::PyRequestStatus;

use dynamo_kvbm::G1;
use dynamo_kvbm::v2::integrations::common::{Request, SchedulerOutput};
use dynamo_kvbm::v2::integrations::scheduler::{KVCacheManager, Scheduler};
use dynamo_kvbm::v2::logical::BlockRegistry;
use dynamo_kvbm::v2::logical::manager::BlockManager;
use dynamo_kvbm::v2::logical::pools::BlockDuplicationPolicy;
use dynamo_kvbm::v2::utils::tinylfu::TinyLFUTracker;
use pyo3::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

use crate::v2::connector::leader::PyConnectorLeader;

/// Python wrapper for the Rust Scheduler.
///
/// This wraps the real `Scheduler` from `dynamo_kvbm::v2::integrations::scheduler`.
/// All scheduling logic, block allocation, and request lifecycle management is
/// delegated to the real implementation.
///
/// Example:
///     config = SchedulerConfig(max_num_batched_tokens=8192, max_num_seqs=256, block_size=16, total_blocks=10132)
///     scheduler = RustScheduler(config)
///     scheduler.add_request(request_id="req-1", prompt_token_ids=[1, 2, 3])
///     output = scheduler.schedule()
#[pyclass(name = "RustScheduler")]
pub struct PyScheduler {
    /// The real Rust scheduler from kvbm.
    inner: Scheduler,

    /// Total blocks available (stored for query methods).
    total_blocks: usize,
}

#[pymethods]
impl PyScheduler {
    /// Create a new RustScheduler with the given configuration.
    ///
    /// This creates a real `BlockManager<G1>` and `KVCacheManager` to manage
    /// KV cache blocks with RAII semantics.
    ///
    /// Args:
    ///     config: Scheduler configuration (including total_blocks for KV cache)
    ///     connector: Optional ConnectorLeader for KV cache offloading and intelligent eviction
    #[new]
    #[pyo3(signature = (config, connector=None))]
    pub fn new(
        config: &PySchedulerConfig,
        connector: Option<&PyConnectorLeader>,
    ) -> PyResult<Self> {
        // Calculate total blocks: use configured value or conservative default
        let total_blocks = config.total_blocks.unwrap_or_else(|| {
            // Default: enough blocks for max_num_seqs requests with average 512 tokens each
            let avg_tokens_per_request = 512;
            let blocks_per_request =
                (avg_tokens_per_request + config.inner.block_size - 1) / config.inner.block_size;
            config.inner.max_num_seqs * blocks_per_request
        });

        let has_connector = connector.is_some();
        tracing::info!(
            max_num_batched_tokens = config.inner.max_num_batched_tokens,
            max_num_seqs = config.inner.max_num_seqs,
            block_size = config.inner.block_size,
            total_blocks = total_blocks,
            has_connector = has_connector,
            "Creating Dynamo Scheduler"
        );

        // Create frequency tracker for MultiLRU backend
        let frequency_tracker = Arc::new(TinyLFUTracker::<u128>::new(total_blocks));

        // Create BlockRegistry with frequency tracking for MultiLRU backend
        let registry = BlockRegistry::with_frequency_tracker(frequency_tracker);

        // Create BlockManager<G1> with real blocks
        let block_manager = BlockManager::<G1>::builder()
            .block_count(total_blocks)
            .block_size(config.inner.block_size)
            .registry(registry)
            .with_lineage_backend()
            .duplication_policy(BlockDuplicationPolicy::Allow)
            .build()
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to create BlockManager: {}",
                    e
                ))
            })?;

        // Create KVCacheManager wrapping the BlockManager
        let kv_cache = KVCacheManager::with_prefix_caching(
            block_manager,
            config.inner.block_size,
            config.inner.enable_prefix_caching,
        )
        .map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to create KVCacheManager: {}",
                e
            ))
        })?;

        // Create the Scheduler using builder pattern
        let mut builder = Scheduler::builder()
            .config(config.inner.clone())
            .kv_cache(kv_cache);

        if let Some(conn) = connector {
            builder = builder.connector(conn.inner());
        }

        let inner = builder.build().map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!("Failed to build scheduler: {}", e))
        })?;

        Ok(Self {
            inner,
            total_blocks,
        })
    }

    /// Add a new request to the scheduler.
    ///
    /// Args:
    ///     request_id: Unique identifier for the request
    ///     prompt_token_ids: List of prompt token IDs
    #[pyo3(signature = (request_id, prompt_token_ids))]
    pub fn add_request(&mut self, request_id: String, prompt_token_ids: Vec<u32>) -> PyResult<()> {
        tracing::info!(
            request_id = %request_id,
            prompt_len = prompt_token_ids.len(),
            "RustScheduler: Adding request"
        );

        // Create Request from common module
        let request = Request::new(
            request_id,
            prompt_token_ids, // Converts to Tokens
            None,             // lora_name
            None,             // salt
            None,             // max_tokens
        );

        self.inner.add_request(request);
        Ok(())
    }

    /// Run the scheduler to produce a scheduling decision.
    ///
    /// Returns:
    ///     dict: Scheduling output containing scheduled requests with REAL block IDs
    pub fn schedule(&mut self, py: Python<'_>) -> PyResult<PyObject> {
        // Call the real scheduler
        let output = self.inner.schedule();

        tracing::info!(
            iteration = output.iteration,
            total_num_scheduled_tokens = output.total_num_scheduled_tokens,
            num_new_reqs = output.scheduled_new_reqs.len(),
            num_cached_reqs = output.scheduled_cached_reqs.len(),
            num_running = self.inner.num_running(),
            num_waiting = self.inner.num_waiting(),
            cache_usage = self.inner.cache_usage(),
            "RustScheduler: schedule() complete"
        );

        // Convert SchedulerOutput to Python dict
        convert_scheduler_output_to_python(py, &output)
    }

    /// Abort a request by ID.
    ///
    /// Args:
    ///     request_id: ID of the request to abort
    pub fn abort_request(&mut self, request_id: &str) -> PyResult<()> {
        tracing::info!(request_id = %request_id, "RustScheduler: Aborting request");
        self.inner.abort_request(request_id);
        Ok(())
    }

    /// Finish requests by ID.
    ///
    /// Args:
    ///     request_ids: List of request IDs to finish
    ///     status: Finish status
    #[pyo3(signature = (request_ids, status))]
    pub fn finish_requests(
        &mut self,
        request_ids: Vec<String>,
        status: &PyRequestStatus,
    ) -> PyResult<()> {
        tracing::info!(
            request_ids = ?request_ids,
            status = ?status.inner,
            "RustScheduler: Finishing requests"
        );
        self.inner.finish_requests(&request_ids, status.inner);
        Ok(())
    }

    /// Update state after model output.
    ///
    /// Args:
    ///     finished_ids: List of request IDs that finished
    ///     output_tokens: Dict mapping request_id -> list of output tokens
    #[pyo3(signature = (finished_ids, output_tokens))]
    pub fn update_from_output(
        &mut self,
        finished_ids: Vec<String>,
        output_tokens: HashMap<String, Vec<u32>>,
    ) -> PyResult<()> {
        tracing::debug!(
            finished_ids = ?finished_ids,
            num_output_requests = output_tokens.len(),
            "RustScheduler: update_from_output()"
        );
        self.inner.update_from_output(&finished_ids, &output_tokens);
        Ok(())
    }

    /// Get the number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.inner.num_waiting()
    }

    /// Get the number of running requests.
    pub fn num_running(&self) -> usize {
        self.inner.num_running()
    }

    /// Get the current iteration number.
    pub fn iteration(&self) -> usize {
        self.inner.iteration()
    }

    /// Get the cache usage as a fraction (0.0 to 1.0).
    pub fn cache_usage(&self) -> f32 {
        self.inner.cache_usage()
    }

    /// Get the number of used blocks.
    pub fn used_blocks(&self) -> usize {
        // Calculate from cache_usage and total_blocks
        (self.inner.cache_usage() * self.total_blocks as f32) as usize
    }

    /// Get the total number of blocks.
    pub fn total_blocks(&self) -> usize {
        self.total_blocks
    }

    /// Check if there are unfinished requests.
    pub fn has_unfinished_requests(&self) -> bool {
        self.inner.num_waiting() > 0 || self.inner.num_running() > 0
    }

    /// Get the number of unfinished requests.
    pub fn get_num_unfinished_requests(&self) -> usize {
        self.inner.num_waiting() + self.inner.num_running()
    }
}

/// Convert SchedulerOutput to Python dict matching vLLM's expected format.
///
/// The block IDs in the output are REAL block IDs from BlockManager<G1>.
fn convert_scheduler_output_to_python(
    py: Python<'_>,
    output: &SchedulerOutput,
) -> PyResult<PyObject> {
    let result = pyo3::types::PyDict::new(py);
    result.set_item("iteration", output.iteration)?;

    // Convert scheduled_new_reqs - block_ids are REAL from BlockManager<G1>
    let scheduled_new_reqs = pyo3::types::PyList::empty(py);
    for req in &output.scheduled_new_reqs {
        let new_req = pyo3::types::PyDict::new(py);
        new_req.set_item("req_id", &req.req_id)?;
        new_req.set_item("prompt_token_ids", &req.prompt_token_ids)?;
        // Wrap block_ids in a vec for vLLM format: [[block_ids]]
        new_req.set_item("block_ids", vec![req.block_ids.clone()])?;
        new_req.set_item("num_computed_tokens", req.num_computed_tokens)?;
        scheduled_new_reqs.append(new_req)?;
    }
    result.set_item("scheduled_new_reqs", scheduled_new_reqs)?;

    // Convert scheduled_cached_reqs
    let scheduled_cached_reqs = pyo3::types::PyDict::new(py);
    let req_ids = pyo3::types::PyList::empty(py);
    let resumed_from_preemption = pyo3::types::PyList::empty(py);
    let new_token_ids = pyo3::types::PyList::empty(py);
    let new_block_ids_list = pyo3::types::PyList::empty(py);
    let num_computed_tokens_list = pyo3::types::PyList::empty(py);

    for req in &output.scheduled_cached_reqs {
        req_ids.append(&req.req_id)?;
        resumed_from_preemption.append(req.resumed)?;
        new_token_ids.append(pyo3::types::PyList::new(py, &req.new_token_ids)?)?;

        // new_block_ids are REAL from BlockManager<G1>
        if !req.new_block_ids.is_empty() {
            new_block_ids_list.append(vec![req.new_block_ids.clone()])?;
        } else {
            new_block_ids_list.append(py.None())?;
        }

        num_computed_tokens_list.append(req.num_computed_tokens)?;
    }

    scheduled_cached_reqs.set_item("req_ids", req_ids)?;
    scheduled_cached_reqs.set_item("resumed_from_preemption", resumed_from_preemption)?;
    scheduled_cached_reqs.set_item("new_token_ids", new_token_ids)?;
    scheduled_cached_reqs.set_item("new_block_ids", new_block_ids_list)?;
    scheduled_cached_reqs.set_item("num_computed_tokens", num_computed_tokens_list)?;
    result.set_item("scheduled_cached_reqs", scheduled_cached_reqs)?;

    // Convert num_scheduled_tokens
    let num_scheduled_tokens_dict = pyo3::types::PyDict::new(py);
    for (req_id, tokens) in &output.num_scheduled_tokens {
        num_scheduled_tokens_dict.set_item(req_id, *tokens)?;
    }
    result.set_item("num_scheduled_tokens", num_scheduled_tokens_dict)?;
    result.set_item(
        "total_num_scheduled_tokens",
        output.total_num_scheduled_tokens,
    )?;

    // vLLM-expected empty fields
    result.set_item("scheduled_spec_decode_tokens", pyo3::types::PyDict::new(py))?;
    result.set_item("scheduled_encoder_inputs", pyo3::types::PyDict::new(py))?;
    result.set_item("num_common_prefix_blocks", pyo3::types::PyList::empty(py))?;
    result.set_item("finished_req_ids", pyo3::types::PyList::empty(py))?;
    result.set_item("free_encoder_mm_hashes", pyo3::types::PyList::empty(py))?;

    // Add kv_connector_metadata if present (serialized as JSON bytes)
    if let Some(ref metadata) = output.kv_connector_metadata {
        let metadata_bytes = serde_json::to_vec(metadata).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Failed to serialize kv_connector_metadata: {}",
                e
            ))
        })?;
        result.set_item("kv_connector_metadata", metadata_bytes)?;
    } else {
        result.set_item("kv_connector_metadata", py.None())?;
    }

    Ok(result.into())
}
