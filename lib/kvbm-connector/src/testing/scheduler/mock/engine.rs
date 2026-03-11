// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// TODO: Disabled — depends on v2::integrations::scheduler::{KVCacheManager, Scheduler, SchedulerConfig,
// GlobalProjectionState} which has no workspace equivalent in this phase.
// Re-enable when integrations/scheduler is ported.
#[cfg(TODO)]
mod engine_disabled {
    //! Mock engine core for CPU-only scheduler testing.

    use super::model::MockModelRunner;
    use crate::G1;
    use crate::common::{Request, SchedulerOutput};
    use kvbm_logical::manager::BlockManager;
    use kvbm_engine::testing::managers::TestRegistryBuilder;

    // TODO: These types come from v2::integrations::scheduler — no workspace equivalent
    // use crate::v2::integrations::scheduler::{
    //     GlobalProjectionState, KVCacheManager, Scheduler, SchedulerConfig,
    // };

    use crate::testing::connector::{ConnectorTestConfig, TestConnectorInstance};

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
        /// Whether to enable connector integration for E2E testing.
        pub enable_connector: bool,
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
                enable_connector: false,
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

        /// Optional connector instance for E2E testing.
        connector_instance: Option<TestConnectorInstance>,
    }

    impl MockEngineCore {
        /// Create a new mock engine core.
        pub fn new(config: MockEngineCoreConfig) -> anyhow::Result<Self> {
            let registry = TestRegistryBuilder::new().build();
            let block_manager = BlockManager::<G1>::builder()
                .block_count(config.total_blocks)
                .block_size(config.block_size)
                .registry(registry)
                .with_lru_backend()
                .build()
                .map_err(|e| anyhow::anyhow!("Failed to build BlockManager: {}", e))?;

            let kv_cache = KVCacheManager::new(block_manager, config.block_size)?;

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

            let (scheduler, connector_instance) = if config.enable_connector {
                let connector_config = ConnectorTestConfig::new()
                    .leader_cache_blocks(64)
                    .leader_disk_blocks(32);

                let instance = TestConnectorInstance::create_with_config(connector_config, 1)
                    .map_err(|e| anyhow::anyhow!("Failed to create TestConnectorInstance: {}", e))?;

                let scheduler = Scheduler::builder()
                    .config(scheduler_config)
                    .kv_cache(kv_cache)
                    .connector(instance.leader.clone())
                    .build()
                    .map_err(|e| anyhow::anyhow!("Failed to build Scheduler with connector: {}", e))?;

                (scheduler, Some(instance))
            } else {
                let scheduler = Scheduler::new(scheduler_config, kv_cache);
                (scheduler, None)
            };

            let model_runner = MockModelRunner::new(config.seed, config.vocab_size);

            Ok(Self {
                scheduler,
                model_runner,
                iteration: 0,
                config,
                requests: HashMap::new(),
                output_tokens: HashMap::new(),
                finished: HashSet::new(),
                connector_instance,
            })
        }

        /// Add a test request to the engine.
        pub fn add_request(&mut self, request: TestRequest) {
            let scheduler_request = Request::new(
                &request.request_id,
                request.prompt_tokens.clone(),
                None,
                None,
                Some(request.max_tokens),
            );

            self.output_tokens
                .insert(request.request_id.clone(), Vec::new());
            self.requests.insert(request.request_id.clone(), request);

            self.scheduler.add_request(scheduler_request);
        }

        /// Check if there are pending requests to process.
        pub fn has_pending_requests(&self) -> bool {
            self.scheduler.num_waiting() > 0 || self.scheduler.num_running() > 0
        }

        /// Execute one scheduler step.
        pub fn step(&mut self) -> Option<StepOutput> {
            if !self.has_pending_requests() {
                return None;
            }

            let schedule_output = self.scheduler.schedule();
            self.iteration = schedule_output.iteration;

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

            let model_output = self.model_runner.generate(&scheduled_ids);
            let finished = self.update_request_state(&model_output);
            self.scheduler.update_from_output(&finished, &model_output);

            Some(StepOutput {
                schedule_output,
                model_output,
                finished,
                iteration: self.iteration,
            })
        }

        fn update_request_state(&mut self, model_output: &HashMap<String, Vec<u32>>) -> Vec<String> {
            let mut finished = Vec::new();

            for (req_id, tokens) in model_output {
                if self.finished.contains(req_id) {
                    continue;
                }
                let Some(request) = self.requests.get(req_id) else {
                    continue;
                };

                if let Some(output) = self.output_tokens.get_mut(req_id) {
                    output.extend(tokens);

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

        pub fn projection_state(&self) -> Option<&GlobalProjectionState> {
            self.scheduler.projection_state()
        }

        pub fn has_choke_points(&self) -> bool {
            self.projection_state()
                .map(|p| p.has_choke_points())
                .unwrap_or(false)
        }

        pub fn iteration(&self) -> usize {
            self.iteration
        }

        pub fn cache_usage(&self) -> f32 {
            self.scheduler.cache_usage()
        }

        pub fn num_waiting(&self) -> usize {
            self.scheduler.num_waiting()
        }

        pub fn num_running(&self) -> usize {
            self.scheduler.num_running()
        }

        pub fn config(&self) -> &MockEngineCoreConfig {
            &self.config
        }

        pub fn abort_request(&mut self, request_id: &str) {
            self.scheduler.abort_request(request_id);
            self.requests.remove(request_id);
            self.output_tokens.remove(request_id);
        }

        pub fn scheduler(&self) -> &Scheduler {
            &self.scheduler
        }

        pub fn scheduler_mut(&mut self) -> &mut Scheduler {
            &mut self.scheduler
        }

        pub fn has_connector(&self) -> bool {
            self.connector_instance.is_some()
        }

        pub fn connector_instance(&self) -> Option<&TestConnectorInstance> {
            self.connector_instance.as_ref()
        }

        pub fn has_connector_slot(&self, request_id: &str) -> bool {
            self.scheduler
                .connector_shim()
                .map(|shim| shim.has_slot(request_id))
                .unwrap_or(false)
        }
    }
}
