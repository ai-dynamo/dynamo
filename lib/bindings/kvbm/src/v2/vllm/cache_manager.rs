// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! PyO3 binding for vLLM's kv-cache-manager shim.
//!
//! [`PyRustKvCacheManager`] is the `#[pyclass]` that powers the
//! `kvbm.v2.vllm.kv_cache_manager.RustKvCacheManager` Python shim. It
//! is a thin wrapper over
//! [`kvbm_connector::vllm::cache_manager::RustKvCacheManager`] — the
//! real implementation lives there; this file only translates
//! PyO3 types and error conventions.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyList, PyType};

use kvbm_connector::vllm::cache_manager::RustKvCacheManager;
use kvbm_logical::BlockId;

use super::handle::PyG1BlockManagerHandle;
use crate::to_pyerr;

/// vLLM kv-cache-manager binding exposed to Python as
/// ``kvbm._core.v2.RustKvCacheManager``.
#[pyclass(name = "RustKvCacheManager")]
pub struct PyRustKvCacheManager {
    core: RustKvCacheManager,
}

#[pymethods]
impl PyRustKvCacheManager {
    /// Build a manager with a private `BlockManager<G1>`.
    #[new]
    #[pyo3(signature = (
        total_blocks,
        block_size,
        enable_caching = true,
        log_stats = false,
    ))]
    pub fn new(
        total_blocks: usize,
        block_size: usize,
        enable_caching: bool,
        log_stats: bool,
    ) -> PyResult<Self> {
        let core = RustKvCacheManager::new(total_blocks, block_size, enable_caching, log_stats)
            .map_err(to_pyerr)?;
        Ok(Self { core })
    }

    /// Build a manager from a shared `BlockManager<G1>` handle (so the
    /// G1 registry is shared with the connector leader).
    #[classmethod]
    #[pyo3(name = "from_g1_handle")]
    pub fn from_g1_handle(
        _cls: &Bound<'_, PyType>,
        handle: &PyG1BlockManagerHandle,
        enable_caching: bool,
        log_stats: bool,
    ) -> PyResult<Self> {
        let core = RustKvCacheManager::from_manager(handle.inner.clone(), enable_caching, log_stats)
            .map_err(to_pyerr)?;
        Ok(Self { core })
    }

    // -- Properties / simple getters -----------------------------------

    pub fn usage(&self) -> f32 {
        self.core.usage()
    }

    pub fn log_stats_enabled(&self) -> bool {
        self.core.log_stats_enabled()
    }

    pub fn take_prefix_cache_stats(&self) -> (u64, u64) {
        self.core.take_prefix_cache_stats()
    }

    pub fn total_blocks(&self) -> usize {
        self.core.total_blocks()
    }

    pub fn block_size(&self) -> usize {
        self.core.block_size()
    }

    pub fn has_slot(&self, request_id: &str) -> bool {
        self.core.has_slot(request_id)
    }

    // -- Slot lifecycle ------------------------------------------------

    pub fn create_slot(
        &self,
        request_id: String,
        tokens: Vec<u32>,
        salt_hash: u64,
        max_output_tokens: usize,
    ) -> PyResult<()> {
        self.core
            .create_slot(request_id, tokens, salt_hash, max_output_tokens)
            .map_err(to_pyerr)
    }

    pub fn get_computed_blocks(
        &self,
        request_id: &str,
    ) -> PyResult<(Vec<BlockId>, usize)> {
        self.core.get_computed_blocks(request_id).map_err(to_pyerr)
    }

    #[pyo3(signature = (
        request_id,
        new_token_ids,
        num_new_tokens,
        num_new_computed_tokens = 0,
        delay_cache_blocks = false,
    ))]
    pub fn allocate_slots(
        &self,
        request_id: &str,
        new_token_ids: Vec<u32>,
        num_new_tokens: usize,
        num_new_computed_tokens: usize,
        delay_cache_blocks: bool,
    ) -> PyResult<Option<Vec<BlockId>>> {
        if num_new_tokens == 0 {
            return Err(PyValueError::new_err("num_new_tokens must be > 0"));
        }
        self.core
            .allocate_slots(
                request_id,
                new_token_ids,
                num_new_tokens,
                num_new_computed_tokens,
                delay_cache_blocks,
            )
            .map_err(to_pyerr)
    }

    pub fn cache_blocks(&self, request_id: &str, num_computed_tokens: usize) -> PyResult<()> {
        self.core
            .cache_blocks(request_id, num_computed_tokens)
            .map_err(to_pyerr)
    }

    pub fn free(&self, request_id: &str) -> PyResult<()> {
        self.core.free(request_id);
        Ok(())
    }

    pub fn get_block_ids(&self, request_id: &str) -> Vec<BlockId> {
        self.core.get_block_ids(request_id)
    }

    pub fn reset_prefix_cache(&self) -> bool {
        self.core.reset_prefix_cache()
    }

    pub fn get_num_common_prefix_blocks(&self, running_request_id: &str) -> usize {
        self.core.get_num_common_prefix_blocks(running_request_id)
    }

    /// Take any queued KV cache events. The Rust core does not currently
    /// emit events, so this always returns an empty list. Kept on the
    /// class for signature parity with vLLM's ``take_events``.
    pub fn take_events<'py>(&self, py: Python<'py>) -> Bound<'py, PyList> {
        PyList::empty(py)
    }
}
