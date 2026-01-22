// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Projection analysis for proactive block budgeting and planned eviction.
//!
//! This module provides infrastructure for predicting future block usage and
//! making proactive scheduling decisions based on those predictions.
//!
//! # Overview
//!
//! The projection system models the future state of all running requests to:
//! - Predict when block shortages (choke points) will occur
//! - Identify requests eligible for eviction (met minimum progress guarantee)
//! - Enable proactive pause/eviction before allocation failures
//! - Support progressive block release from paused requests
//!
//! # Key Concepts
//!
//! ## Guaranteed Minimum Progress
//!
//! Every request is guaranteed to make some minimum progress before becoming
//! eligible for eviction. This is computed as:
//!
//! ```text
//! guaranteed_min = min(
//!     user_min_tokens,
//!     min(tokens_to_boundary + 2 * block_size, 3 * block_size)
//! )
//! ```
//!
//! Where `tokens_to_boundary` is the number of tokens needed to complete the
//! current partial block after prefill.
//!
//! ## Choke Points
//!
//! A choke point is a predicted future iteration where block demand exceeds
//! supply. The projector looks ahead N iterations and identifies these points
//! so the scheduler can act proactively.
//!
//! ## Eviction Eligibility
//!
//! A request becomes eviction-eligible when:
//! 1. It has generated at least `guaranteed_min_tokens` output tokens
//! 2. The connector reports no inflight offloads (`can_evict() == true`)

use super::request::SchedulerRequest;
use crate::v2::BlockId;

use std::collections::HashMap;

/// Per-request projection of future block usage.
///
/// This struct models the expected resource consumption trajectory
/// for a request, enabling proactive scheduling decisions.
#[derive(Debug, Clone)]
pub struct ProjectionState {
    /// Current number of blocks allocated to this request.
    pub current_blocks: usize,

    /// Current total tokens (prompt + output).
    pub current_tokens: usize,

    /// Prompt/ISL length - tokens that were in the initial request.
    pub prompt_tokens: usize,

    /// Best-case blocks needed (based on min_tokens or guaranteed minimum).
    pub min_projected_blocks: usize,

    /// Worst-case blocks needed (based on max_tokens or max_seq_len - ISL).
    pub max_projected_blocks: usize,

    /// Earliest iteration this request could complete (if min_tokens reached).
    pub earliest_completion_iteration: Option<usize>,

    /// Latest iteration this request could complete (if max_tokens reached).
    pub latest_completion_iteration: Option<usize>,

    /// Whether this request has met its minimum progress guarantee.
    pub eviction_eligible: bool,

    /// Number of tokens generated so far (excludes prompt).
    pub output_tokens_generated: usize,

    /// The guaranteed minimum tokens before eviction (computed from min_tokens + block alignment).
    pub guaranteed_min_tokens: usize,

    /// G2 coverage ratio (0.0-1.0) for blocks already offloaded.
    /// Updated by querying the connector.
    pub g2_coverage: f32,

    /// Whether this request is at a block boundary.
    pub at_block_boundary: bool,

    // =========================================================================
    // Chunked Prefill State
    // =========================================================================
    /// Whether this request is currently prefilling (hasn't finished initial prompt processing).
    pub is_prefilling: bool,

    /// Remaining tokens to prefill (0 if prefill is complete).
    pub remaining_prefill_tokens: usize,

    /// The base iteration when this projection was created.
    /// Used for calculating relative iterations in blocks_at_iteration().
    pub base_iteration: usize,

    // =========================================================================
    // Priority and Eviction Fields
    // =========================================================================
    /// User-defined priority for eviction ordering.
    /// Higher values = higher priority (less likely to be evicted).
    /// None is treated as lowest priority (evicted first).
    pub user_priority: Option<usize>,

    /// Tokens until the next block boundary.
    /// Used for optimal pause point selection.
    pub tokens_to_boundary: usize,

    /// Maximum output tokens this request can generate.
    /// Used for completion estimation.
    pub max_output_tokens: usize,

    /// Number of blocks that would actually be freed on eviction.
    ///
    /// This accounts for block reference counting - shared blocks (via prefix
    /// caching) won't return resources when released. Only blocks with
    /// `use_count() == 1` are counted as freeable.
    pub freeable_blocks: usize,
}

impl ProjectionState {
    /// Create a new projection for a request.
    ///
    /// # Arguments
    /// * `request` - The scheduler request to project
    /// * `block_size` - Block size in tokens
    /// * `max_seq_len` - Maximum sequence length from model config
    /// * `current_iteration` - Current scheduler iteration
    ///
    /// # Note on Resumed Requests
    ///
    /// For resumed requests (after preemption), `prompt_tokens` is set to
    /// `total_known_tokens()` rather than `original_prompt_len()`. This is
    /// because the resumed request needs to recompute its entire sequence,
    /// making the full sequence effectively the "prompt" for projection purposes.
    pub fn new(
        request: &SchedulerRequest,
        block_size: usize,
        max_seq_len: usize,
        current_iteration: usize,
    ) -> Self {
        // Use total_known_tokens() for prompt_tokens - this handles both fresh
        // and resumed requests correctly. For fresh requests, this equals
        // original_prompt_len(). For resumed requests, this includes all tokens
        // up to the eviction point that need to be recomputed.
        let prompt_tokens = request.total_known_tokens();
        let current_tokens = request.total_known_tokens();
        let output_tokens_generated = request.num_output_tokens;

        // Calculate guaranteed minimum tokens before eviction eligibility
        let guaranteed_min_tokens = Self::compute_guaranteed_min(request, block_size);

        // Check eviction eligibility
        let eviction_eligible = output_tokens_generated >= guaranteed_min_tokens;

        // Calculate block projections
        let current_blocks = request.block_state.total_blocks();
        let min_projected_blocks =
            Self::compute_min_blocks(prompt_tokens, guaranteed_min_tokens, block_size);
        let max_projected_blocks = Self::compute_max_blocks(request, block_size, max_seq_len);

        // Estimate completion iterations
        let (earliest, latest) = Self::estimate_completion_iterations(
            output_tokens_generated,
            guaranteed_min_tokens,
            request.request.max_tokens,
            max_seq_len,
            prompt_tokens,
            current_iteration,
        );

        // Check block boundary alignment
        let at_block_boundary = current_tokens > 0 && (current_tokens % block_size) == 0;

        // Determine prefill state
        let is_prefilling = request.num_computed_tokens < prompt_tokens;
        let remaining_prefill_tokens = prompt_tokens.saturating_sub(request.num_computed_tokens);

        // Tokens until next block boundary
        let partial_tokens = current_tokens % block_size;
        let tokens_to_boundary = if partial_tokens == 0 {
            0
        } else {
            block_size - partial_tokens
        };

        // Max output tokens
        let max_output_tokens = request
            .request
            .max_tokens
            .unwrap_or(max_seq_len.saturating_sub(prompt_tokens));

        // Freeable blocks (accounts for prefix cache sharing)
        let freeable_blocks = request.block_state.freeable_blocks();

        Self {
            current_blocks,
            current_tokens,
            prompt_tokens,
            min_projected_blocks,
            max_projected_blocks,
            earliest_completion_iteration: earliest,
            latest_completion_iteration: latest,
            eviction_eligible,
            output_tokens_generated,
            guaranteed_min_tokens,
            g2_coverage: 0.0, // Updated by connector query
            at_block_boundary,
            is_prefilling,
            remaining_prefill_tokens,
            base_iteration: current_iteration,
            user_priority: request.request.priority,
            tokens_to_boundary,
            max_output_tokens,
            freeable_blocks,
        }
    }

    /// Update projection state incrementally after tokens are generated.
    ///
    /// This is an incremental update that avoids full recomputation.
    /// Call this after processing model output when new tokens are generated.
    ///
    /// # Arguments
    /// * `num_new_tokens` - Number of new output tokens generated
    /// * `block_size` - Block size in tokens
    /// * `current_iteration` - Current scheduler iteration for completion estimates
    pub fn update_for_tokens_generated(
        &mut self,
        num_new_tokens: usize,
        block_size: usize,
        current_iteration: usize,
    ) {
        // Update token counts
        self.output_tokens_generated += num_new_tokens;
        self.current_tokens += num_new_tokens;

        // Update block counts
        self.current_blocks = self.current_tokens.div_ceil(block_size);

        // Update eviction eligibility
        self.eviction_eligible = self.output_tokens_generated >= self.guaranteed_min_tokens;

        // Update block boundary detection
        self.at_block_boundary = self.current_tokens > 0 && (self.current_tokens % block_size) == 0;

        // Update prefill state - if we're generating tokens, prefill is done
        self.is_prefilling = false;
        self.remaining_prefill_tokens = 0;

        // Update tokens to next boundary
        let partial_tokens = self.current_tokens % block_size;
        self.tokens_to_boundary = if partial_tokens == 0 {
            0
        } else {
            block_size - partial_tokens
        };

        // Update completion iteration estimates
        let tokens_to_min = self
            .guaranteed_min_tokens
            .saturating_sub(self.output_tokens_generated);
        self.earliest_completion_iteration = Some(current_iteration + tokens_to_min);

        let tokens_to_max = self
            .max_output_tokens
            .saturating_sub(self.output_tokens_generated);
        self.latest_completion_iteration = Some(current_iteration + tokens_to_max);
    }

    /// Compute guaranteed minimum tokens before eviction eligibility.
    ///
    /// Default = min(min_tokens, up to 3 full blocks worth)
    /// If ISL not on block boundary: remaining tokens in partial block + 2 more blocks
    ///
    /// # Note on Resumed Requests
    ///
    /// Uses `total_known_tokens()` rather than `original_prompt_len()` because
    /// resumed requests need to recompute their entire sequence. The block
    /// boundary calculation should be based on the full sequence length.
    pub fn compute_guaranteed_min(request: &SchedulerRequest, block_size: usize) -> usize {
        // Use total_known_tokens() to handle both fresh and resumed requests.
        // For resumed requests, the "effective prompt" is the full sequence.
        let effective_prompt_len = request.total_known_tokens();
        let partial_block_tokens = effective_prompt_len % block_size;

        // Tokens to complete the partial block (if any)
        let tokens_to_boundary = if partial_block_tokens > 0 {
            block_size - partial_block_tokens
        } else {
            0
        };

        // Default: complete partial block + 2 more full blocks
        let default_guaranteed = tokens_to_boundary + (2 * block_size);

        // If min_tokens provided, use the smaller of min_tokens and default
        let user_min = request.request.min_tokens.unwrap_or(usize::MAX);

        // Also cap at 3 full blocks worth
        let max_guaranteed = 3 * block_size;

        default_guaranteed.min(user_min).min(max_guaranteed)
    }

    fn compute_min_blocks(
        prompt_tokens: usize,
        guaranteed_min_tokens: usize,
        block_size: usize,
    ) -> usize {
        let min_total_tokens = prompt_tokens + guaranteed_min_tokens;
        (min_total_tokens + block_size - 1) / block_size
    }

    fn compute_max_blocks(
        request: &SchedulerRequest,
        block_size: usize,
        max_seq_len: usize,
    ) -> usize {
        // Use original_prompt_len() + max_tokens for the absolute maximum.
        // This is correct even for resumed requests because max_tokens limits
        // output from the original prompt, not from the eviction point.
        let original_prompt = request.original_prompt_len();
        let max_output = request
            .request
            .max_tokens
            .unwrap_or(max_seq_len.saturating_sub(original_prompt));
        let max_total = original_prompt + max_output;
        (max_total + block_size - 1) / block_size
    }

    fn estimate_completion_iterations(
        output_tokens_generated: usize,
        guaranteed_min_tokens: usize,
        max_tokens: Option<usize>,
        max_seq_len: usize,
        prompt_len: usize,
        current_iteration: usize,
    ) -> (Option<usize>, Option<usize>) {
        // Earliest: when min tokens would be reached
        let tokens_to_min = guaranteed_min_tokens.saturating_sub(output_tokens_generated);
        let earliest = current_iteration + tokens_to_min;

        // Latest: when max_tokens would be reached
        let max_output = max_tokens.unwrap_or(max_seq_len.saturating_sub(prompt_len));
        let tokens_to_max = max_output.saturating_sub(output_tokens_generated);
        let latest = current_iteration + tokens_to_max;

        (Some(earliest), Some(latest))
    }

    /// Returns projected blocks needed at `iterations_ahead` iterations in the future.
    ///
    /// Returns (min_blocks, max_blocks) based on best/worst case scenarios.
    ///
    /// # Chunked Prefill Awareness
    ///
    /// During prefill, a request may consume multiple blocks per iteration (up to
    /// `max_prefill_chunk_size` tokens). This method accounts for this by:
    /// - Calculating how many iterations until prefill completes
    /// - During prefill iterations: projecting chunk_size tokens per iteration
    /// - After prefill: projecting 1 token per iteration (decode phase)
    ///
    /// # Arguments
    /// * `iterations_ahead` - Number of iterations from now (0 = after next iteration)
    /// * `block_size` - Block size in tokens
    /// * `max_prefill_chunk_size` - Maximum tokens per prefill chunk (None = unlimited)
    pub fn blocks_at_iteration(
        &self,
        iterations_ahead: usize,
        block_size: usize,
        max_prefill_chunk_size: Option<usize>,
    ) -> (usize, usize) {
        // Compute tokens that will have been processed by iteration N
        // (tokens_computed = tokens for which we need KV cache blocks)
        let tokens_computed = if self.is_prefilling {
            // Currently prefilling: we haven't computed all prompt tokens yet
            let chunk_size = max_prefill_chunk_size.unwrap_or(self.remaining_prefill_tokens);
            let prefill_iterations = self.remaining_prefill_tokens.div_ceil(chunk_size.max(1));

            // How many tokens were computed at base (before this projection)?
            let computed_at_base = self
                .prompt_tokens
                .saturating_sub(self.remaining_prefill_tokens);

            if iterations_ahead < prefill_iterations {
                // Still in prefill phase
                // After (iterations_ahead + 1) prefill passes, we've computed that many chunks
                let tokens_prefilled =
                    ((iterations_ahead + 1) * chunk_size).min(self.remaining_prefill_tokens);
                computed_at_base + tokens_prefilled
            } else {
                // Prefill is complete, now in decode phase
                // We've computed all prompt tokens + some decode tokens
                let decode_iterations = iterations_ahead - prefill_iterations;
                self.prompt_tokens + decode_iterations + 1 // +1 for first decode token after prefill
            }
        } else {
            // Already in decode phase: 1 token per iteration
            // iterations_ahead = 0 means after next pass, which adds 1 token
            self.current_tokens + iterations_ahead + 1
        };

        // Min: capped at what we need for guaranteed minimum
        let min_total = self.prompt_tokens + self.guaranteed_min_tokens;
        let projected_min =
            (tokens_computed.div_ceil(block_size)).min(min_total.div_ceil(block_size));

        // Max: capped at max projected blocks
        let projected_max = (tokens_computed.div_ceil(block_size)).min(self.max_projected_blocks);

        (projected_min, projected_max)
    }

    /// Returns the remaining tokens until completion.
    ///
    /// Returns the number of output tokens remaining before this request reaches
    /// its max_tokens limit. Used for eviction priority ordering.
    pub fn remaining_tokens(&self) -> usize {
        self.max_output_tokens
            .saturating_sub(self.output_tokens_generated)
    }

    /// Check if this request will likely complete within the given iterations.
    pub fn will_complete_within(&self, iterations: usize, current_iteration: usize) -> bool {
        if let Some(latest) = self.latest_completion_iteration {
            latest <= current_iteration + iterations
        } else {
            false
        }
    }
}

/// A predicted point where block demand exceeds supply.
#[derive(Debug, Clone)]
pub struct ChokePoint {
    /// Iteration at which the choke point occurs.
    pub iteration: usize,

    /// Minimum predicted block demand at this iteration.
    pub min_demand: usize,

    /// Maximum predicted block demand at this iteration.
    pub max_demand: usize,

    /// Available blocks at this iteration (assuming no changes).
    pub supply: usize,

    /// Block deficit (demand - supply) if positive.
    pub deficit: isize,

    /// Requests contributing most to the demand (top 3).
    pub major_contributors: Vec<String>,
}

/// Aggregates projections across all requests to predict future block pressure.
pub struct BlockBudgetProjector {
    /// Block size in tokens.
    block_size: usize,

    /// Maximum sequence length (from model config).
    max_seq_len: usize,

    /// Total blocks available in G1.
    total_blocks: usize,

    /// How many iterations to look ahead.
    lookahead_iterations: usize,

    /// Maximum prefill chunk size (for chunked prefill awareness).
    max_prefill_chunk_size: Option<usize>,

    /// Per-request projections (keyed by request_id).
    pub projections: HashMap<String, ProjectionState>,

    /// Predicted choke points in the lookahead window.
    choke_points: Vec<ChokePoint>,
}

impl BlockBudgetProjector {
    /// Create a new block budget projector.
    pub fn new(
        block_size: usize,
        max_seq_len: usize,
        total_blocks: usize,
        lookahead_iterations: usize,
    ) -> Self {
        Self::with_prefill_chunk_size(
            block_size,
            max_seq_len,
            total_blocks,
            lookahead_iterations,
            None,
        )
    }

    /// Create a new block budget projector with prefill chunk size configuration.
    pub fn with_prefill_chunk_size(
        block_size: usize,
        max_seq_len: usize,
        total_blocks: usize,
        lookahead_iterations: usize,
        max_prefill_chunk_size: Option<usize>,
    ) -> Self {
        Self {
            block_size,
            max_seq_len,
            total_blocks,
            lookahead_iterations,
            max_prefill_chunk_size,
            projections: HashMap::new(),
            choke_points: Vec::new(),
        }
    }

    /// Set the maximum prefill chunk size.
    pub fn set_max_prefill_chunk_size(&mut self, size: Option<usize>) {
        self.max_prefill_chunk_size = size;
    }

    /// Update projections for all requests.
    ///
    /// This should be called at the start of each scheduling iteration.
    pub fn update_projections<'a>(
        &mut self,
        requests: impl Iterator<Item = (&'a String, &'a SchedulerRequest)>,
        current_iteration: usize,
    ) {
        self.projections.clear();

        for (request_id, request) in requests {
            let projection = ProjectionState::new(
                request,
                self.block_size,
                self.max_seq_len,
                current_iteration,
            );
            self.projections.insert(request_id.clone(), projection);
        }
    }

    /// Compute choke points in the lookahead window.
    pub fn compute_choke_points(&mut self, current_iteration: usize) {
        self.choke_points.clear();

        for delta in 1..=self.lookahead_iterations {
            let iteration = current_iteration + delta;
            let (min_demand, max_demand, contributors) = self.compute_demand_at_iteration(delta);

            if max_demand > self.total_blocks {
                self.choke_points.push(ChokePoint {
                    iteration,
                    min_demand,
                    max_demand,
                    supply: self.total_blocks,
                    deficit: (max_demand as isize) - (self.total_blocks as isize),
                    major_contributors: contributors,
                });
            }
        }
    }

    fn compute_demand_at_iteration(&self, iterations_ahead: usize) -> (usize, usize, Vec<String>) {
        let mut total_min = 0;
        let mut total_max = 0;
        let mut contributors: Vec<(String, usize)> = Vec::new();

        for (request_id, projection) in &self.projections {
            let (min_blocks, max_blocks) = projection.blocks_at_iteration(
                iterations_ahead,
                self.block_size,
                self.max_prefill_chunk_size,
            );
            total_min += min_blocks;
            total_max += max_blocks;
            contributors.push((request_id.clone(), max_blocks));
        }

        // Sort by contribution (descending) and take top 3
        contributors.sort_by(|a, b| b.1.cmp(&a.1));
        let top_contributors: Vec<String> =
            contributors.into_iter().take(3).map(|(id, _)| id).collect();

        (total_min, total_max, top_contributors)
    }

    /// Get requests that are eviction-eligible, sorted by eviction preference.
    ///
    /// Eviction priority order (best candidates for eviction first):
    /// 1. Must be eviction-eligible (achieved compute_guaranteed_min)
    /// 2. Lowest user priority (None = lowest, evicted first)
    /// 3. Furthest from completion (most remaining tokens)
    /// 4. Closest to block boundary (less waste when pausing)
    ///
    /// This ordering ensures:
    /// - Only requests that have made guaranteed minimum progress are considered
    /// - User-specified priorities are respected
    /// - Near-completion requests are preserved (they'll finish soon)
    /// - Block-aligned pauses minimize wasted partial blocks
    pub fn get_eviction_candidates(&self) -> Vec<(&str, &ProjectionState)> {
        let mut candidates: Vec<_> = self
            .projections
            .iter()
            .filter(|(_, p)| p.eviction_eligible)
            .map(|(id, p)| (id.as_str(), p))
            .collect();

        // Sort by eviction priority (best candidates for eviction first):
        // 1. Lower user priority = evict first (None = 0 = lowest priority)
        // 2. Furthest from completion (most remaining tokens)
        // 3. Higher G2 coverage (faster resume from offloaded blocks)
        //
        // Note: tokens_to_boundary is NOT used here - it tells us WHEN to pause
        // (at block boundary for zero waste), not WHO to evict. We can always
        // pause sooner and accept recompute cost for partial block tokens.
        candidates.sort_by(|a, b| {
            let priority_a = a.1.user_priority.unwrap_or(0);
            let priority_b = b.1.user_priority.unwrap_or(0);

            priority_a
                .cmp(&priority_b)
                .then_with(|| {
                    // More remaining tokens = evict first (furthest from completion)
                    b.1.remaining_tokens().cmp(&a.1.remaining_tokens())
                })
                .then_with(|| {
                    // Higher G2 coverage = evict first (faster resume)
                    b.1.g2_coverage
                        .partial_cmp(&a.1.g2_coverage)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        candidates
    }

    /// Recommend pause candidates based on blocks needed.
    ///
    /// Returns request IDs that should be paused to free up the requested blocks.
    /// Uses `freeable_blocks` which accounts for block reference counting - shared
    /// blocks (via prefix caching) won't actually return capacity when released.
    pub fn recommend_pause_candidates(&self, blocks_to_free: usize) -> Vec<&str> {
        let candidates = self.get_eviction_candidates();
        let mut recommended = Vec::new();
        let mut freed = 0;

        for (request_id, projection) in candidates {
            if freed >= blocks_to_free {
                break;
            }
            recommended.push(request_id);
            // Use freeable_blocks, not current_blocks - shared blocks don't free capacity
            freed += projection.freeable_blocks;
        }

        recommended
    }

    /// Get projection for a specific request.
    pub fn get_projection(&self, request_id: &str) -> Option<&ProjectionState> {
        self.projections.get(request_id)
    }

    /// Get mutable projection for a specific request.
    pub fn get_projection_mut(&mut self, request_id: &str) -> Option<&mut ProjectionState> {
        self.projections.get_mut(request_id)
    }

    /// Remove a projection for a finished request.
    ///
    /// Call this when a request finishes to clean up its projection.
    pub fn remove_projection(&mut self, request_id: &str) -> Option<ProjectionState> {
        self.projections.remove(request_id)
    }

    /// Update a single projection incrementally after token generation.
    ///
    /// This avoids the full recomputation of all projections.
    ///
    /// # Arguments
    /// * `request_id` - The request to update
    /// * `num_new_tokens` - Number of new output tokens generated
    /// * `current_iteration` - Current scheduler iteration
    pub fn update_single_projection(
        &mut self,
        request_id: &str,
        num_new_tokens: usize,
        current_iteration: usize,
    ) {
        if let Some(projection) = self.projections.get_mut(request_id) {
            projection.update_for_tokens_generated(
                num_new_tokens,
                self.block_size,
                current_iteration,
            );
        }
    }

    /// Check if any choke points exist.
    pub fn has_choke_points(&self) -> bool {
        !self.choke_points.is_empty()
    }

    /// Get the nearest choke point.
    pub fn nearest_choke_point(&self) -> Option<&ChokePoint> {
        self.choke_points.first()
    }

    /// Get all choke points.
    pub fn choke_points(&self) -> &[ChokePoint] {
        &self.choke_points
    }

    /// Get the total block demand at the current iteration.
    pub fn current_block_demand(&self) -> usize {
        self.projections.values().map(|p| p.current_blocks).sum()
    }

    /// Get available headroom (free blocks).
    pub fn available_headroom(&self) -> usize {
        self.total_blocks
            .saturating_sub(self.current_block_demand())
    }
}

impl std::fmt::Debug for BlockBudgetProjector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BlockBudgetProjector")
            .field("block_size", &self.block_size)
            .field("max_seq_len", &self.max_seq_len)
            .field("total_blocks", &self.total_blocks)
            .field("lookahead_iterations", &self.lookahead_iterations)
            .field("num_projections", &self.projections.len())
            .field("num_choke_points", &self.choke_points.len())
            .finish()
    }
}

// ============================================================================
// Planned Eviction Tracking
// ============================================================================

/// A request planned for eviction with priority G2 offload.
#[derive(Debug, Clone)]
pub struct PlannedEviction {
    /// Request ID.
    pub request_id: String,

    /// Iteration when eviction should happen.
    pub target_iteration: usize,

    /// Blocks that need priority offload.
    pub blocks_to_offload: Vec<BlockId>,

    /// Blocks already offloaded.
    pub blocks_offloaded: Vec<BlockId>,

    /// Whether priority offload has been requested from connector.
    pub offload_requested: bool,
}

/// Tracks requests that are planned for eviction with priority offload.
#[derive(Debug, Default)]
pub struct PlannedEvictionTracker {
    /// Requests planned for eviction, with their target iteration.
    planned: HashMap<String, PlannedEviction>,
}

impl PlannedEvictionTracker {
    /// Create a new empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Plan a request for eviction.
    pub fn plan_eviction(
        &mut self,
        request_id: String,
        target_iteration: usize,
        blocks: Vec<BlockId>,
    ) {
        self.planned.insert(
            request_id.clone(),
            PlannedEviction {
                request_id,
                target_iteration,
                blocks_to_offload: blocks,
                blocks_offloaded: Vec::new(),
                offload_requested: false,
            },
        );
    }

    /// Check if a request is planned for eviction.
    pub fn is_planned(&self, request_id: &str) -> bool {
        self.planned.contains_key(request_id)
    }

    /// Get requests ready for eviction (offload complete or target reached).
    pub fn get_ready_for_eviction(&self, current_iteration: usize) -> Vec<&str> {
        self.planned
            .iter()
            .filter(|(_, p)| {
                p.blocks_to_offload.is_empty() || current_iteration >= p.target_iteration
            })
            .map(|(id, _)| id.as_str())
            .collect()
    }

    /// Mark blocks as offloaded for a planned eviction.
    pub fn mark_offloaded(&mut self, request_id: &str, block_ids: &[BlockId]) {
        if let Some(planned) = self.planned.get_mut(request_id) {
            for block_id in block_ids {
                if let Some(pos) = planned.blocks_to_offload.iter().position(|b| b == block_id) {
                    let block = planned.blocks_to_offload.remove(pos);
                    planned.blocks_offloaded.push(block);
                }
            }
        }
    }

    /// Remove a request from planned eviction (cancelled or completed).
    pub fn remove(&mut self, request_id: &str) -> Option<PlannedEviction> {
        self.planned.remove(request_id)
    }

    /// Get all planned evictions that need offload requests sent.
    pub fn get_pending_offload_requests(&mut self) -> Vec<&mut PlannedEviction> {
        self.planned
            .values_mut()
            .filter(|p| !p.offload_requested && !p.blocks_to_offload.is_empty())
            .collect()
    }

    /// Get the number of planned evictions.
    pub fn len(&self) -> usize {
        self.planned.len()
    }

    /// Check if there are no planned evictions.
    pub fn is_empty(&self) -> bool {
        self.planned.is_empty()
    }

    /// Iterate over planned evictions.
    pub fn iter(&self) -> impl Iterator<Item = (&String, &PlannedEviction)> {
        self.planned.iter()
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::v2::integrations::common::Request;

    fn create_test_scheduler_request(
        request_id: &str,
        prompt_len: usize,
        output_tokens: usize,
        min_tokens: Option<usize>,
        max_tokens: Option<usize>,
        block_size: usize,
    ) -> SchedulerRequest {
        let tokens: Vec<u32> = (0..prompt_len as u32).collect();
        let request = Request::with_token_limits(
            request_id, tokens, None, // lora_name
            None, // salt
            min_tokens, max_tokens, None, // metadata
        );
        let mut sched_req = SchedulerRequest::new(request, block_size);
        sched_req.num_output_tokens = output_tokens;
        sched_req
    }

    #[test]
    fn test_compute_guaranteed_min_at_boundary() {
        // Prompt exactly on block boundary (64 tokens, block_size=16)
        let request = create_test_scheduler_request("r1", 64, 0, None, None, 16);

        // At boundary: tokens_to_boundary = 0, so default = 0 + 2*16 = 32
        let guaranteed = ProjectionState::compute_guaranteed_min(&request, 16);
        assert_eq!(guaranteed, 32);
    }

    #[test]
    fn test_compute_guaranteed_min_partial_block() {
        // Prompt not on boundary (70 tokens, block_size=16)
        // 70 % 16 = 6 tokens in partial block
        // tokens_to_boundary = 16 - 6 = 10
        // default = 10 + 32 = 42
        let request = create_test_scheduler_request("r1", 70, 0, None, None, 16);

        let guaranteed = ProjectionState::compute_guaranteed_min(&request, 16);
        assert_eq!(guaranteed, 42);
    }

    #[test]
    fn test_compute_guaranteed_min_with_user_min() {
        // User specifies min_tokens=20, which is less than default (32)
        let request = create_test_scheduler_request("r1", 64, 0, Some(20), None, 16);

        let guaranteed = ProjectionState::compute_guaranteed_min(&request, 16);
        assert_eq!(guaranteed, 20);
    }

    #[test]
    fn test_compute_guaranteed_min_capped_at_3_blocks() {
        // Large partial block scenario that would exceed 3 blocks
        // block_size=16, 3*16=48 is the cap
        let request = create_test_scheduler_request("r1", 65, 0, Some(100), None, 16);

        let guaranteed = ProjectionState::compute_guaranteed_min(&request, 16);
        // tokens_to_boundary = 16 - 1 = 15, default = 15 + 32 = 47
        // user_min = 100, max = 48
        // min(47, 100, 48) = 47
        assert_eq!(guaranteed, 47);
    }

    #[test]
    fn test_eviction_eligible_not_met() {
        let request = create_test_scheduler_request("r1", 64, 10, None, None, 16);
        let projection = ProjectionState::new(&request, 16, 4096, 0);

        // guaranteed_min = 32, output = 10, so not eligible
        assert!(!projection.eviction_eligible);
        assert_eq!(projection.guaranteed_min_tokens, 32);
    }

    #[test]
    fn test_eviction_eligible_met() {
        let request = create_test_scheduler_request("r1", 64, 50, None, None, 16);
        let projection = ProjectionState::new(&request, 16, 4096, 0);

        // guaranteed_min = 32, output = 50, so eligible
        assert!(projection.eviction_eligible);
    }

    #[test]
    fn test_block_boundary_detection() {
        // On boundary: 64 tokens (64 / 16 = 4 blocks, no remainder)
        let request = create_test_scheduler_request("r1", 64, 0, None, None, 16);
        let projection = ProjectionState::new(&request, 16, 4096, 0);
        assert!(projection.at_block_boundary);

        // Not on boundary: 70 tokens
        let request2 = create_test_scheduler_request("r2", 70, 0, None, None, 16);
        let projection2 = ProjectionState::new(&request2, 16, 4096, 0);
        assert!(!projection2.at_block_boundary);
    }

    #[test]
    fn test_choke_point_detection() {
        // total_blocks=10, lookahead=20
        // Each request starts with 64 tokens = 4 blocks
        // At iteration +17: 64+17=81 tokens = 6 blocks each
        // 3 requests * 6 blocks = 18 blocks > 10 → choke point
        let mut projector = BlockBudgetProjector::new(16, 4096, 10, 20);

        // Create 3 requests that will exceed 10 blocks
        let r1 = create_test_scheduler_request("r1", 64, 0, None, Some(200), 16);
        let r2 = create_test_scheduler_request("r2", 64, 0, None, Some(200), 16);
        let r3 = create_test_scheduler_request("r3", 64, 0, None, Some(200), 16);

        let requests: Vec<(String, SchedulerRequest)> = vec![
            ("r1".to_string(), r1),
            ("r2".to_string(), r2),
            ("r3".to_string(), r3),
        ];

        let request_refs: Vec<(&String, &SchedulerRequest)> =
            requests.iter().map(|(k, v)| (k, v)).collect();

        projector.update_projections(request_refs.into_iter(), 0);
        projector.compute_choke_points(0);

        // With 3 requests growing toward 200+ tokens, we should see choke points
        // At iteration +17: 3 * 6 = 18 blocks > 10
        assert!(projector.has_choke_points());
        assert!(projector.choke_points()[0].deficit > 0);
    }

    #[test]
    fn test_eviction_candidate_ranking() {
        let mut projector = BlockBudgetProjector::new(16, 4096, 100, 5);

        // Request 1: eligible, no priority, max_tokens=100, generated=32
        // remaining = 100 - 32 = 68
        let mut r1 = create_test_scheduler_request("r1", 64, 32, None, Some(100), 16);
        r1.num_output_tokens = 32; // eligible (>= 32)

        // Request 2: eligible, no priority, max_tokens=200, generated=50
        // remaining = 200 - 50 = 150 (more remaining = evict first)
        let mut r2 = create_test_scheduler_request("r2", 70, 50, None, Some(200), 16);
        r2.num_output_tokens = 50; // eligible (>= 42)

        let requests: Vec<(String, SchedulerRequest)> =
            vec![("r1".to_string(), r1), ("r2".to_string(), r2)];

        let request_refs: Vec<(&String, &SchedulerRequest)> =
            requests.iter().map(|(k, v)| (k, v)).collect();

        projector.update_projections(request_refs.into_iter(), 0);

        let candidates = projector.get_eviction_candidates();

        // r2 should be first (more remaining tokens = evict first)
        // Both have same priority (None), so we compare remaining_tokens
        // r2 has 150 remaining vs r1 has 68 remaining
        assert_eq!(candidates[0].0, "r2");
        assert_eq!(candidates[1].0, "r1");
    }

    #[test]
    fn test_eviction_candidate_priority_ordering() {
        let mut projector = BlockBudgetProjector::new(16, 4096, 100, 5);

        // Request 1: eligible, priority=10 (higher = less likely to evict)
        let tokens1: Vec<u32> = (0..64).collect();
        let request1 = Request::with_priority(
            "r1",
            tokens1,
            None,
            None,
            None,
            Some(100),
            Some(10), // Higher priority
            None,
        );
        let mut r1 = SchedulerRequest::new(request1, 16);
        r1.num_output_tokens = 50;

        // Request 2: eligible, no priority (None = 0 = lowest)
        let tokens2: Vec<u32> = (0..64).collect();
        let request2 = Request::with_priority(
            "r2",
            tokens2,
            None,
            None,
            None,
            Some(100),
            None, // No priority = lowest
            None,
        );
        let mut r2 = SchedulerRequest::new(request2, 16);
        r2.num_output_tokens = 50;

        let requests: Vec<(String, SchedulerRequest)> =
            vec![("r1".to_string(), r1), ("r2".to_string(), r2)];

        let request_refs: Vec<(&String, &SchedulerRequest)> =
            requests.iter().map(|(k, v)| (k, v)).collect();

        projector.update_projections(request_refs.into_iter(), 0);

        let candidates = projector.get_eviction_candidates();

        // r2 should be first (lower priority = evict first)
        assert_eq!(candidates[0].0, "r2");
        assert_eq!(candidates[1].0, "r1");
    }

    #[test]
    fn test_blocks_at_iteration_chunked_prefill() {
        // Create a request that is prefilling
        let tokens: Vec<u32> = (0..256).collect(); // 256 prompt tokens
        let request = Request::with_token_limits("r1", tokens, None, None, None, Some(100), None);
        let mut sched_req = SchedulerRequest::new(request, 16);
        // Set num_computed_tokens < prompt_len to indicate prefilling
        sched_req.num_computed_tokens = 0;
        sched_req.num_output_tokens = 0;

        let projection = ProjectionState::new(&sched_req, 16, 4096, 0);

        // Should be marked as prefilling
        assert!(projection.is_prefilling);
        assert_eq!(projection.remaining_prefill_tokens, 256);

        // With chunk_size=128, prefill takes 2 iterations
        // Iteration 0: 128 tokens → 8 blocks
        // Iteration 1: 256 tokens → 16 blocks
        // Iteration 2: 256 + 1 (decode) = 257 tokens → 17 blocks
        let (_, max_blocks_0) = projection.blocks_at_iteration(0, 16, Some(128));
        let (_, max_blocks_1) = projection.blocks_at_iteration(1, 16, Some(128));
        let (_, max_blocks_2) = projection.blocks_at_iteration(2, 16, Some(128));

        assert_eq!(max_blocks_0, 8); // 128 / 16 = 8
        assert_eq!(max_blocks_1, 16); // 256 / 16 = 16
        assert_eq!(max_blocks_2, 17); // (256 + 1) / 16 rounded up = 17
    }

    #[test]
    fn test_planned_eviction_tracker() {
        let mut tracker = PlannedEvictionTracker::new();

        tracker.plan_eviction("r1".to_string(), 10, vec![1, 2, 3]);
        assert!(tracker.is_planned("r1"));
        assert!(!tracker.is_planned("r2"));

        // Not ready yet (target=10, current=5)
        let ready = tracker.get_ready_for_eviction(5);
        assert!(ready.is_empty());

        // Mark some blocks offloaded
        tracker.mark_offloaded("r1", &[1, 2]);

        // Still not ready (one block remaining)
        let ready = tracker.get_ready_for_eviction(5);
        assert!(ready.is_empty());

        // Mark last block offloaded
        tracker.mark_offloaded("r1", &[3]);

        // Now ready (all blocks offloaded)
        let ready = tracker.get_ready_for_eviction(5);
        assert_eq!(ready, vec!["r1"]);
    }

    #[test]
    fn test_eviction_ordering_ignores_boundary() {
        // Verify that tokens_to_boundary does NOT affect eviction ordering.
        // Block boundary distance tells us WHEN to pause, not WHO to evict.

        let mut projector = BlockBudgetProjector::new(16, 4096, 100, 5);

        // Request 1: eligible, 5 tokens to boundary
        let mut r1 = create_test_scheduler_request("r1", 64 + 11, 50, None, Some(200), 16);
        // 64 + 11 = 75 tokens, 75 % 16 = 11, so 5 tokens to boundary
        r1.num_output_tokens = 50;

        // Request 2: eligible, 0 tokens to boundary (at boundary)
        let mut r2 = create_test_scheduler_request("r2", 64, 50, None, Some(200), 16);
        // 64 tokens, 64 % 16 = 0, at boundary
        r2.num_output_tokens = 50;

        let requests: Vec<(String, SchedulerRequest)> =
            vec![("r1".to_string(), r1), ("r2".to_string(), r2)];

        let request_refs: Vec<(&String, &SchedulerRequest)> =
            requests.iter().map(|(k, v)| (k, v)).collect();

        projector.update_projections(request_refs.into_iter(), 0);

        let candidates = projector.get_eviction_candidates();

        // Both should be eligible
        assert_eq!(candidates.len(), 2);

        // With old logic, r2 (at boundary) would be first.
        // With new logic, tokens_to_boundary is ignored.
        // Both have same priority (None), so order is by remaining_tokens.
        // r1: max=200, output=50 -> remaining=150
        // r2: max=200, output=50 -> remaining=150
        // Same remaining, so order may be arbitrary.
        // The key assertion is that the ORDER is NOT determined by tokens_to_boundary.

        // Verify projections have different tokens_to_boundary
        let p1 = projector.get_projection("r1").unwrap();
        let p2 = projector.get_projection("r2").unwrap();
        assert_eq!(p2.tokens_to_boundary, 0); // r2 at boundary
        assert!(p1.tokens_to_boundary > 0); // r1 not at boundary

        // Both have same remaining tokens, so they could be in either order
        // This test just verifies the logic doesn't crash and both are included
        assert!(candidates.iter().any(|(id, _)| *id == "r1"));
        assert!(candidates.iter().any(|(id, _)| *id == "r2"));
    }

    #[test]
    fn test_projection_uses_total_known_tokens() {
        // Test that ProjectionState correctly uses total_known_tokens()
        // which handles both fresh and resumed requests.

        let tokens: Vec<u32> = (0..64).collect();
        let request = Request::with_token_limits("r1", tokens, None, None, None, Some(100), None);
        let mut sched_req = SchedulerRequest::new(request, 16);

        // Simulate the request generating some output tokens
        sched_req.num_output_tokens = 20;
        // Also extend the token sequence to match
        let output_tokens: Vec<u32> = (64..84).collect();
        sched_req.extend_tokens(&output_tokens).unwrap();

        // Create projection
        let projection = ProjectionState::new(&sched_req, 16, 4096, 0);

        // prompt_tokens should be total_known_tokens() = 84 (64 + 20)
        // This is important for resumed requests where we need to recompute
        // the full sequence, not just the original prompt
        assert_eq!(projection.prompt_tokens, 84);
        assert_eq!(projection.current_tokens, 84);
    }

    #[test]
    fn test_recommend_pause_uses_freeable_blocks() {
        // Test that recommend_pause_candidates uses freeable_blocks
        // not current_blocks (which wouldn't account for shared blocks)

        let mut projector = BlockBudgetProjector::new(16, 4096, 100, 5);

        // Request with enough output to be eviction eligible
        let mut r1 = create_test_scheduler_request("r1", 64, 50, None, Some(200), 16);
        r1.num_output_tokens = 50;

        let requests: Vec<(String, SchedulerRequest)> = vec![("r1".to_string(), r1)];

        let request_refs: Vec<(&String, &SchedulerRequest)> =
            requests.iter().map(|(k, v)| (k, v)).collect();

        projector.update_projections(request_refs.into_iter(), 0);

        // Get projection and check freeable_blocks field exists
        let projection = projector.get_projection("r1").unwrap();
        // For a request with no allocated blocks, freeable_blocks should be 0
        assert_eq!(projection.freeable_blocks, 0);

        // recommend_pause_candidates should work (even if it returns empty
        // because no blocks can actually be freed)
        let candidates = projector.recommend_pause_candidates(5);
        // It should still recommend the request (even though freeable is 0)
        // because we iterate through candidates adding their freeable counts
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0], "r1");
    }
}
