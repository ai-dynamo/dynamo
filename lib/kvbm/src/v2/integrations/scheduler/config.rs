// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler configuration.

use derive_builder::Builder;

/// Configuration for the scheduler.
#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned", build_fn(error = "SchedulerConfigBuilderError"))]
pub struct SchedulerConfig {
    /// Maximum number of tokens that can be scheduled in a single iteration.
    #[builder(default = "8192")]
    pub max_num_batched_tokens: usize,

    /// Maximum number of sequences that can be scheduled in a single iteration.
    #[builder(default = "256")]
    pub max_num_seqs: usize,

    /// Block size in tokens.
    #[builder(default = "16")]
    pub block_size: usize,

    /// Whether to enable prefix caching (reuse blocks across requests).
    #[builder(default = "false")]
    pub enable_prefix_caching: bool,

    /// Whether to enable chunked prefill (split long prefills across iterations).
    #[builder(default = "false")]
    pub enable_chunked_prefill: bool,

    /// Maximum number of tokens to prefill in a single chunk (when chunked prefill is enabled).
    #[builder(default, setter(strip_option))]
    pub max_prefill_chunk_size: Option<usize>,
}

/// Error type for SchedulerConfigBuilder.
#[derive(Debug, Clone, thiserror::Error)]
pub enum SchedulerConfigBuilderError {
    #[error("Uninitialized field: {0}")]
    UninitializedField(&'static str),
    #[error("Validation error: {0}")]
    ValidationError(String),
}

impl From<derive_builder::UninitializedFieldError> for SchedulerConfigBuilderError {
    fn from(e: derive_builder::UninitializedFieldError) -> Self {
        Self::UninitializedField(e.field_name())
    }
}

impl From<String> for SchedulerConfigBuilderError {
    fn from(s: String) -> Self {
        Self::ValidationError(s)
    }
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_num_batched_tokens: 8192,
            max_num_seqs: 256,
            block_size: 16,
            enable_prefix_caching: false,
            enable_chunked_prefill: false,
            max_prefill_chunk_size: None,
        }
    }
}

impl SchedulerConfig {
    /// Create a new builder for SchedulerConfig.
    pub fn builder() -> SchedulerConfigBuilder {
        SchedulerConfigBuilder::default()
    }

    /// Create a new scheduler config with the given parameters.
    pub fn new(max_num_batched_tokens: usize, max_num_seqs: usize, block_size: usize) -> Self {
        Self {
            max_num_batched_tokens,
            max_num_seqs,
            block_size,
            ..Default::default()
        }
    }
}
