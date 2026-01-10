// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock engine core for CPU-only scheduler testing.

use super::model::MockModelRunner;
use crate::v2::integrations::common::{Request, SchedulerOutput};
use crate::v2::integrations::scheduler::{
    GlobalProjectionState, KVCacheManager, Scheduler, SchedulerConfig,
};
use crate::v2::logical::manager::BlockManager;
use crate::v2::testing::managers::create_test_registry;
use crate::v2::G1;

use std::collections::{HashMap, HashSet};

/// Simple test request for the mock engine.
#[derive(Debug, Clone)]
pub struct TestRequest {
    /// Unique request ID.
    pub request_id: String,
    /// Prompt tokens.
    pub prompt_tokens: Vec<u32>,
    /// Maximum output tokens to generate.
    pub max_tokens: usize,
}

/// Configuration for MockEngineCore.
#[derive(Debug, Clone)]
pub struct MockEngineCoreConfig {
    /// Maximum sequence length supported.
    pub max_seq_len: usize,
    /// Maximum tokens per iteration.
    pub max_num_batched_tokens: usize,
    /// Maximum sequences per iteration.
    pub max_num_seqs: usize,
    /// Block size in tokens.
    pub block_size: usize,
    /// Total blocks available.
    pub total_blocks: usize,
    /// Random seed for deterministic output.
    pub seed: u64,
    /// Vocabulary size for token generation.
    pub vocab_size: u32,
    /// Whether to enable projection-based scheduling.
    pub enable_projection: bool,
}

impl Default for MockEngineCoreConfig {
    fn default() -> Self {
        Self {
            max_seq_len: 8192,
            max_num_batched_tokens: 4096,
            max_num_seqs: 128,
            block_size: 16,
            total_blocks: 512,
            seed: 42,
            vocab_size: 50257,
            enable_projection: true,
        }
    }
}

/// Output from a single scheduler step.
#[derive(Debug)]
pub struct StepOutput {
    /// The scheduler output from this step.
    pub schedule_output: SchedulerOutput,
    /// Generated tokens per request.
    pub model_output: HashMap<String, Vec<u32>>,
    /// Requests that finished this step.
    pub finished: Vec<String>,
    /// Current iteration number.
    pub iteration: usize,
}

/// Mock engine core for CPU-only scheduler testing.
///
/// This drives the real Scheduler without GPU by generating deterministic
/// "model outputs" using seeded random tokens.
pub struct MockEngineCore {
    /// The real scheduler being tested.
    scheduler: Scheduler,
    /// Mock model runner for token generation.
    model_runner: MockModelRunner,
    /// Current iteration number.
    iteration: usize,
    /// Configuration.
    config: MockEngineCoreConfig,

    // Request tracking
    /// All requests added to the engine.
    pub requests: HashMap<String, TestRequest>,
    /// Generated output tokens per request.
    pub output_tokens: HashMap<String, Vec<u32>>,
    /// Requests that have finished.
    pub finished: HashSet<String>,
}

impl MockEngineCore {
    /// Create a new mock engine core.
    pub fn new(config: MockEngineCoreConfig) -> anyhow::Result<Self> {
        // Create a test block manager
        let registry = create_test_registry();
        let block_manager = BlockManager::<G1>::builder()
            .block_count(config.total_blocks)
            .block_size(config.block_size)
            .registry(registry)
            .with_lru_backend()
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build BlockManager: {}", e))?;

        // Create KV cache manager
        let kv_cache = KVCacheManager::new(block_manager, config.block_size)?;

        // Create scheduler config
        let scheduler_config = SchedulerConfig::builder()
            .max_seq_len(config.max_seq_len)
            .max_num_batched_tokens(config.max_num_batched_tokens)
            .max_num_seqs(config.max_num_seqs)
            .block_size(config.block_size)
            .enable_prefix_caching(false)
            .enable_chunked_prefill(false)
            .max_prefill_chunk_size(None)
            .enable_projection(config.enable_projection)
            .build()
            .map_err(|e| anyhow::anyhow!("Failed to build SchedulerConfig: {}", e))?;

        // Create scheduler
        let scheduler = Scheduler::new(scheduler_config, kv_cache);

        // Create mock model runner
        let model_runner = MockModelRunner::new(config.seed, config.vocab_size);

        Ok(Self {
            scheduler,
            model_runner,
            iteration: 0,
            config,
            requests: HashMap::new(),
            output_tokens: HashMap::new(),
            finished: HashSet::new(),
        })
    }

    /// Add a test request to the engine.
    pub fn add_request(&mut self, request: TestRequest) {
        // Create the scheduler Request
        let scheduler_request = Request::new(
            &request.request_id,
            request.prompt_tokens.clone(),
            None, // lora_name
            None, // salt
            Some(request.max_tokens),
        );

        // Track in our state
        self.output_tokens
            .insert(request.request_id.clone(), Vec::new());
        self.requests
            .insert(request.request_id.clone(), request);

        // Add to scheduler
        self.scheduler.add_request(scheduler_request);
    }

    /// Check if there are pending requests to process.
    pub fn has_pending_requests(&self) -> bool {
        self.scheduler.num_waiting() > 0 || self.scheduler.num_running() > 0
    }

    /// Execute one scheduler step.
    ///
    /// Returns `None` if there are no pending requests.
    pub fn step(&mut self) -> Option<StepOutput> {
        if !self.has_pending_requests() {
            return None;
        }

        // 1. Schedule
        let schedule_output = self.scheduler.schedule();
        self.iteration = schedule_output.iteration;

        // 2. Collect scheduled request IDs
        let scheduled_ids: Vec<String> = schedule_output
            .scheduled_new_reqs
            .iter()
            .map(|r| r.req_id.clone())
            .chain(
                schedule_output
                    .scheduled_cached_reqs
                    .iter()
                    .map(|r| r.req_id.clone()),
            )
            .collect();

        // 3. Generate mock tokens for scheduled requests
        let model_output = self.model_runner.generate(&scheduled_ids);

        // 4. Detect finished requests and update state
        let finished = self.update_request_state(&model_output);

        // 5. Update scheduler with generated tokens
        self.scheduler.update_from_output(&finished, &model_output);

        Some(StepOutput {
            schedule_output,
            model_output,
            finished,
            iteration: self.iteration,
        })
    }

    /// Update request state after model output.
    ///
    /// Returns list of request IDs that finished this step.
    fn update_request_state(&mut self, model_output: &HashMap<String, Vec<u32>>) -> Vec<String> {
        let mut finished = Vec::new();

        for (req_id, tokens) in model_output {
            // Skip if request already finished or not tracked
            if self.finished.contains(req_id) {
                continue;
            }
            let Some(request) = self.requests.get(req_id) else {
                continue;
            };

            // Add generated tokens to our tracking
            if let Some(output) = self.output_tokens.get_mut(req_id) {
                output.extend(tokens);

                // Check if finished (hit max_tokens)
                if output.len() >= request.max_tokens {
                    finished.push(req_id.clone());
                    self.finished.insert(req_id.clone());
                }
            }
        }

        finished
    }

    /// Run until all requests complete or max iterations reached.
    pub fn run_to_completion(&mut self, max_iterations: usize) -> Vec<StepOutput> {
        let mut outputs = Vec::new();

        for _ in 0..max_iterations {
            match self.step() {
                Some(output) => outputs.push(output),
                None => break,
            }
        }

        outputs
    }

    // === Projection Accessors ===

    /// Get the global projection state for validation.
    pub fn projection_state(&self) -> Option<&GlobalProjectionState> {
        self.scheduler.projection_state()
    }

    /// Check if any choke points are detected.
    pub fn has_choke_points(&self) -> bool {
        self.projection_state()
            .map(|p| p.has_choke_points())
            .unwrap_or(false)
    }

    /// Get the current iteration number.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    /// Get the scheduler's KV cache usage.
    pub fn cache_usage(&self) -> f32 {
        self.scheduler.cache_usage()
    }

    /// Get the number of waiting requests.
    pub fn num_waiting(&self) -> usize {
        self.scheduler.num_waiting()
    }

    /// Get the number of running requests.
    pub fn num_running(&self) -> usize {
        self.scheduler.num_running()
    }

    /// Get the configuration.
    pub fn config(&self) -> &MockEngineCoreConfig {
        &self.config
    }

    // === Request Management ===

    /// Abort a request by ID.
    ///
    /// This removes the request from the scheduler and cleans up internal state.
    /// Delegates to the underlying scheduler's `abort_request()` method.
    ///
    /// # Arguments
    /// * `request_id` - The ID of the request to abort
    pub fn abort_request(&mut self, request_id: &str) {
        // Abort in the scheduler (frees blocks, removes from queues)
        self.scheduler.abort_request(request_id);

        // Clean up our internal tracking
        self.requests.remove(request_id);
        self.output_tokens.remove(request_id);
        // Note: We don't add to `finished` set since abort is not a normal completion
    }

    /// Get access to the underlying scheduler for advanced operations.
    ///
    /// Use with caution - direct scheduler manipulation may bypass mock engine tracking.
    pub fn scheduler(&self) -> &Scheduler {
        &self.scheduler
    }

    /// Get mutable access to the underlying scheduler.
    ///
    /// Use with caution - direct scheduler manipulation may bypass mock engine tracking.
    pub fn scheduler_mut(&mut self) -> &mut Scheduler {
        &mut self.scheduler
    }
}
