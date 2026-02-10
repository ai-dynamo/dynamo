// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for scheduler configuration.

use dynamo_kvbm::v2::integrations::scheduler::SchedulerConfig;
use pyo3::prelude::*;

/// Python wrapper for SchedulerConfig.
///
/// This wraps the real `SchedulerConfig` from `dynamo_kvbm::v2::integrations::scheduler`
/// and adds a `total_blocks` field for KVCacheManager creation.
///
/// # Required Arguments (from vLLM framework)
/// - `max_seq_len`: Maximum sequence length supported by the model
/// - `max_num_batched_tokens`: Maximum tokens per iteration
/// - `max_num_seqs`: Maximum sequences per iteration
/// - `block_size`: Block size in tokens
/// - `enable_prefix_caching`: Whether to enable prefix caching
/// - `enable_chunked_prefill`: Whether to enable chunked prefill
/// - `max_prefill_chunk_size`: Max prefill chunk size (can be None)
///
/// # Optional Arguments (have defaults)
/// - `enable_projection`: Enable projection-based scheduling (default: True)
/// - `projection_lookahead`: Iterations to look ahead (default: None -> 2*block_size)
/// - `min_guaranteed_blocks`: Minimum blocks before eviction eligible (default: None -> 3)
/// - `total_blocks`: Total KV cache blocks available (default: None, auto-calculated)
///
/// Example:
///     config = SchedulerConfig(
///         max_seq_len=8192,
///         max_num_batched_tokens=8192,
///         max_num_seqs=256,
///         block_size=16,
///         enable_prefix_caching=True,
///         enable_chunked_prefill=False,
///         max_prefill_chunk_size=None,
///         total_blocks=10132
///     )
#[pyclass(name = "SchedulerConfig")]
#[derive(Clone)]
pub struct PySchedulerConfig {
    /// The real scheduler config from kvbm.
    pub(crate) inner: SchedulerConfig,

    /// Total number of KV cache blocks available.
    /// This is stored separately because the real SchedulerConfig doesn't have it
    /// (it's determined by KVCacheManager).
    pub(crate) total_blocks: Option<usize>,
}

#[pymethods]
impl PySchedulerConfig {
    /// Create a new SchedulerConfig.
    ///
    /// # Required Args (from vLLM framework):
    ///     max_seq_len: Maximum sequence length supported by the model
    ///     max_num_batched_tokens: Maximum tokens per iteration
    ///     max_num_seqs: Maximum sequences per iteration
    ///     block_size: Block size in tokens
    ///     enable_prefix_caching: Enable prefix caching
    ///     enable_chunked_prefill: Enable chunked prefill
    ///     max_prefill_chunk_size: Max prefill chunk size (can be None)
    ///
    /// # Optional Args (have defaults):
    ///     enable_projection: Enable projection-based proactive scheduling (default: None -> True)
    ///     projection_lookahead: Iterations to look ahead for choke points (default: None -> 2*block_size)
    ///     min_guaranteed_blocks: Minimum guaranteed blocks before eviction-eligible (default: None -> 3)
    ///     total_blocks: Total KV cache blocks available (default: None, auto-calculated)
    #[new]
    #[pyo3(signature = (*,
        max_seq_len,
        max_num_batched_tokens,
        max_num_seqs,
        block_size,
        enable_prefix_caching,
        enable_chunked_prefill,
        max_prefill_chunk_size,
        enable_projection = None,
        projection_lookahead = None,
        min_guaranteed_blocks = None,
        total_blocks = None
    ))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_seq_len: usize,
        max_num_batched_tokens: usize,
        max_num_seqs: usize,
        block_size: usize,
        enable_prefix_caching: bool,
        enable_chunked_prefill: bool,
        max_prefill_chunk_size: Option<usize>,
        enable_projection: Option<bool>,
        projection_lookahead: Option<usize>,
        min_guaranteed_blocks: Option<usize>,
        total_blocks: Option<usize>,
    ) -> Self {
        // Build the config using the builder pattern
        let mut builder = SchedulerConfig::builder()
            .max_seq_len(max_seq_len)
            .max_num_batched_tokens(max_num_batched_tokens)
            .max_num_seqs(max_num_seqs)
            .block_size(block_size)
            .enable_prefix_caching(enable_prefix_caching)
            .enable_chunked_prefill(enable_chunked_prefill)
            .max_prefill_chunk_size(max_prefill_chunk_size);

        // Apply optional fields only if provided
        if let Some(v) = enable_projection {
            builder = builder.enable_projection(v);
        }
        if let Some(v) = projection_lookahead {
            builder = builder.projection_lookahead(v);
        }
        if let Some(v) = min_guaranteed_blocks {
            builder = builder.min_guaranteed_blocks(v);
        }

        let inner = builder.build().expect("All required fields provided");

        Self {
            inner,
            total_blocks,
        }
    }

    /// Get max_num_batched_tokens.
    #[getter]
    pub fn max_num_batched_tokens(&self) -> usize {
        self.inner.max_num_batched_tokens
    }

    /// Get max_num_seqs.
    #[getter]
    pub fn max_num_seqs(&self) -> usize {
        self.inner.max_num_seqs
    }

    /// Get block_size.
    #[getter]
    pub fn block_size(&self) -> usize {
        self.inner.block_size
    }

    /// Get enable_prefix_caching.
    #[getter]
    pub fn enable_prefix_caching(&self) -> bool {
        self.inner.enable_prefix_caching
    }

    /// Get enable_chunked_prefill.
    #[getter]
    pub fn enable_chunked_prefill(&self) -> bool {
        self.inner.enable_chunked_prefill
    }

    /// Get max_prefill_chunk_size.
    #[getter]
    pub fn max_prefill_chunk_size(&self) -> Option<usize> {
        self.inner.max_prefill_chunk_size
    }

    /// Get max_seq_len.
    #[getter]
    pub fn max_seq_len(&self) -> usize {
        self.inner.max_seq_len
    }

    /// Get enable_projection.
    #[getter]
    pub fn enable_projection(&self) -> bool {
        self.inner.enable_projection
    }

    /// Get projection_lookahead.
    #[getter]
    pub fn projection_lookahead(&self) -> usize {
        self.inner.projection_lookahead
    }

    /// Get min_guaranteed_blocks.
    #[getter]
    pub fn min_guaranteed_blocks(&self) -> usize {
        self.inner.min_guaranteed_blocks
    }

    /// Get total_blocks.
    #[getter]
    pub fn total_blocks(&self) -> Option<usize> {
        self.total_blocks
    }

    fn __repr__(&self) -> String {
        format!(
            "SchedulerConfig(max_seq_len={}, max_num_batched_tokens={}, max_num_seqs={}, \
             block_size={}, enable_prefix_caching={}, enable_chunked_prefill={}, \
             max_prefill_chunk_size={:?}, enable_projection={}, projection_lookahead={}, \
             min_guaranteed_blocks={}, total_blocks={:?})",
            self.inner.max_seq_len,
            self.inner.max_num_batched_tokens,
            self.inner.max_num_seqs,
            self.inner.block_size,
            self.inner.enable_prefix_caching,
            self.inner.enable_chunked_prefill,
            self.inner.max_prefill_chunk_size,
            self.inner.enable_projection,
            self.inner.projection_lookahead,
            self.inner.min_guaranteed_blocks,
            self.total_blocks
        )
    }
}
