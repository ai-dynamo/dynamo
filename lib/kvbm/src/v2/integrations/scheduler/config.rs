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

    // =========================================================================
    // Projection System Configuration
    // =========================================================================
    /// Maximum sequence length supported by the model.
    ///
    /// Used by the projection system to estimate worst-case block requirements
    /// for requests without explicit `max_tokens` limits.
    #[builder(default = "8192")]
    pub max_seq_len: usize,

    /// Number of iterations to look ahead when detecting choke points.
    ///
    /// Higher values detect choke points earlier but may increase false positives.
    /// Lower values are more reactive but may miss opportunities for proactive
    /// pause/eviction.
    ///
    /// A value of 0 means the lookahead will be computed as `2 * block_size`,
    /// which provides coverage for worst-case block consumption scenarios.
    ///
    /// Use [`effective_lookahead()`](Self::effective_lookahead) to get the actual
    /// lookahead value accounting for this default behavior.
    #[builder(default = "0")]
    pub projection_lookahead: usize,

    /// Whether to enable the projection-based proactive scheduling system.
    ///
    /// When enabled, the scheduler:
    /// - Predicts future block demand based on min/max token constraints
    /// - Detects choke points where demand exceeds supply
    /// - Proactively pauses eligible requests before memory pressure
    /// - Supports progressive block release from paused requests
    #[builder(default = "false")]
    pub enable_projection: bool,
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
            max_seq_len: 8192,
            projection_lookahead: 0, // 0 means use 2 * block_size
            enable_projection: false,
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

    /// Get the effective lookahead iterations for projection.
    ///
    /// If `projection_lookahead` is 0, returns `2 * block_size` to provide
    /// adequate coverage for worst-case block consumption during chunked prefill.
    /// Otherwise returns the configured value.
    pub fn effective_lookahead(&self) -> usize {
        if self.projection_lookahead == 0 {
            2 * self.block_size
        } else {
            self.projection_lookahead
        }
    }

    /// Get the effective prefill chunk size.
    ///
    /// Returns `max_prefill_chunk_size` if set, otherwise `max_num_batched_tokens`.
    pub fn effective_prefill_chunk_size(&self) -> usize {
        self.max_prefill_chunk_size
            .unwrap_or(self.max_num_batched_tokens)
    }
}
