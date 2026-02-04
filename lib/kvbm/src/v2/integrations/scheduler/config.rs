// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Scheduler configuration.

use derive_builder::Builder;

/// Configuration for the scheduler.
///
/// Use [`SchedulerConfig::builder()`] to construct. Required fields must be set
/// explicitly; optional fields have sensible defaults.
///
/// # Required Fields (from vLLM framework)
/// - `max_seq_len` - Maximum sequence length supported by the model
/// - `max_num_batched_tokens` - Maximum tokens per iteration
/// - `max_num_seqs` - Maximum sequences per iteration
/// - `block_size` - Block size in tokens
/// - `enable_prefix_caching` - Whether to enable prefix caching
/// - `enable_chunked_prefill` - Whether to enable chunked prefill
/// - `max_prefill_chunk_size` - Maximum prefill chunk size (None = use max_num_batched_tokens)
///
/// # Optional Fields (have defaults)
/// - `enable_projection` - Enable projection-based scheduling (default: true)
/// - `projection_lookahead` - Iterations to look ahead (default: 0 = 2*block_size)
/// - `min_guaranteed_blocks` - Minimum blocks before eviction eligible (default: 3)
#[derive(Debug, Clone, Builder)]
#[builder(pattern = "owned", build_fn(error = "SchedulerConfigBuilderError"))]
pub struct SchedulerConfig {
    /// Private marker to prevent direct struct construction.
    /// Use `SchedulerConfig::builder()` instead.
    #[builder(setter(skip), default = "()")]
    _private: (),

    // =========================================================================
    // Required Fields - Must be set explicitly (vLLM framework alignment)
    // =========================================================================
    /// Maximum sequence length supported by the model.
    ///
    /// Used by the projection system to estimate worst-case block requirements
    /// for requests without explicit `max_tokens` limits.
    pub max_seq_len: usize,

    /// Maximum number of tokens that can be scheduled in a single iteration.
    pub max_num_batched_tokens: usize,

    /// Maximum number of sequences that can be scheduled in a single iteration.
    pub max_num_seqs: usize,

    /// Block size in tokens.
    pub block_size: usize,

    /// Whether to enable prefix caching (reuse blocks across requests).
    pub enable_prefix_caching: bool,

    /// Whether to enable chunked prefill (split long prefills across iterations).
    pub enable_chunked_prefill: bool,

    /// Maximum number of tokens to prefill in a single chunk (when chunked prefill is enabled).
    /// None means use `max_num_batched_tokens`.
    pub max_prefill_chunk_size: Option<usize>,

    // =========================================================================
    // Optional Fields - Have defaults
    // =========================================================================
    /// Whether to enable the projection-based proactive scheduling system.
    ///
    /// When enabled, the scheduler:
    /// - Predicts future block demand based on min/max token constraints
    /// - Detects choke points where demand exceeds supply
    /// - Proactively pauses eligible requests before memory pressure
    /// - Supports progressive block release from paused requests
    #[builder(default = "true")]
    pub enable_projection: bool,

    /// Number of iterations to look ahead when detecting choke points.
    ///
    /// **NOTE**: This is being replaced by dynamic horizon based on request completion.
    /// The projection system now automatically tracks the furthest iteration needed
    /// based on active requests' completion iterations. Set to 0 to use the new
    /// dynamic behavior (recommended).
    ///
    /// Legacy behavior (when non-zero):
    /// - Higher values detect choke points earlier but may increase false positives.
    /// - Lower values are more reactive but may miss opportunities for proactive
    ///   pause/eviction.
    ///
    /// A value of 0 means the lookahead will be computed dynamically as
    /// `max(2 * block_size, furthest_request_completion)`.
    ///
    /// Use [`effective_lookahead()`](Self::effective_lookahead) to get the fixed
    /// lookahead value for the legacy dense VecDeque system.
    #[builder(default = "0")]
    pub projection_lookahead: usize,

    /// Minimum guaranteed blocks before a request becomes eviction-eligible.
    ///
    /// This controls the guaranteed minimum compute window, if the value is N, then:
    /// - **New requests**: Finish partial block + N-1 more full blocks (up to this value)
    /// - **Restored requests**: Full `min_guaranteed_blocks` blocks (no partial deduction)
    ///
    /// The guarantee ensures requests make progress before being evicted, and
    /// provides time for offload preparation in case of subsequent eviction.
    #[builder(default = "3")]
    pub min_guaranteed_blocks: usize,
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

impl SchedulerConfig {
    /// Create a new builder for SchedulerConfig.
    pub fn builder() -> SchedulerConfigBuilder {
        SchedulerConfigBuilder::default()
    }

    /// Get the effective lookahead iterations for the legacy dense VecDeque system.
    ///
    /// If `projection_lookahead` is 0, returns `2 * block_size` to provide
    /// adequate coverage for worst-case block consumption during chunked prefill.
    /// Otherwise returns the configured value.
    ///
    /// **NOTE**: For the new sparse aggregate demand system, use
    /// `GlobalProjectionState::effective_horizon()` which provides a dynamic
    /// lookahead based on actual request completion iterations.
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

#[cfg(test)]
impl SchedulerConfig {
    /// Test-only convenience method with sensible defaults for all required fields.
    ///
    /// Creates a config with projections **disabled** for simpler test setup.
    /// Tests that specifically need projections should explicitly enable them.
    ///
    /// Creates a config with:
    /// - `max_seq_len`: 8192
    /// - `max_num_batched_tokens`: 8192
    /// - `max_num_seqs`: 256
    /// - `block_size`: 16
    /// - `enable_prefix_caching`: false
    /// - `enable_chunked_prefill`: false
    /// - `max_prefill_chunk_size`: None
    /// - `enable_projection`: false (explicit for test determinism)
    pub fn test_default() -> Self {
        Self::builder()
            .max_seq_len(8192)
            .max_num_batched_tokens(8192)
            .max_num_seqs(256)
            .block_size(16)
            .enable_prefix_caching(false)
            .enable_chunked_prefill(false)
            .max_prefill_chunk_size(None)
            .enable_projection(false) // Explicit for test determinism
            .build()
            .expect("test_default should always succeed")
    }
}
