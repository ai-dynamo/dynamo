// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Core scheduler types: SchedulerConfig, RequestStatus, KVCacheManager, Scheduler.
//!
//! These implement the scheduling contract expected by PyScheduler in mod.rs.
//! KVCacheManager wraps BlockManager<G1> from kvbm-logical, ensuring that block
//! IDs returned to vLLM match the KVBM block pool — fixing the ISL≥24 crash
//! caused by block-ID mismatch between vLLM's scheduler and the KVBM connector.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use anyhow::Result;
use kvbm_connector::common::{Request, SchedulerOutput};
use kvbm_common::BlockId;
use kvbm_connector::connector::leader::ConnectorLeader;
use kvbm_logical::{BlockManager, ImmutableBlock, SequenceHash};

/// GPU VRAM tier — marker type for BlockManager (mirrors kvbm_engine::G1).
/// Satisfies BlockMetadata via its blanket impl (Clone + Send + Sync + 'static).
#[derive(Clone, Copy, Debug)]
pub struct G1;

/// Finish/eviction status for a request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStatus {
    Waiting,
    Running,
    Preempted,
    FinishedStopped,
    FinishedAborted,
    FinishedLengthCapped,
}

impl RequestStatus {
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            Self::FinishedStopped | Self::FinishedAborted | Self::FinishedLengthCapped
        )
    }
}

/// Scheduler configuration mirroring vLLM's scheduler settings.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    pub max_seq_len: usize,
    pub max_num_batched_tokens: usize,
    pub max_num_seqs: usize,
    pub block_size: usize,
    pub enable_prefix_caching: bool,
    pub enable_chunked_prefill: bool,
    pub max_prefill_chunk_size: Option<usize>,
    pub enable_projection: bool,
    pub projection_lookahead: usize,
    pub min_guaranteed_blocks: usize,
}

/// Builder for SchedulerConfig.
#[derive(Default)]
pub struct SchedulerConfigBuilder {
    max_seq_len: Option<usize>,
    max_num_batched_tokens: Option<usize>,
    max_num_seqs: Option<usize>,
    block_size: Option<usize>,
    enable_prefix_caching: Option<bool>,
    enable_chunked_prefill: Option<bool>,
    max_prefill_chunk_size: Option<Option<usize>>,
    enable_projection: Option<bool>,
    projection_lookahead: Option<usize>,
    min_guaranteed_blocks: Option<usize>,
}

impl SchedulerConfigBuilder {
    pub fn max_seq_len(mut self, v: usize) -> Self { self.max_seq_len = Some(v); self }
    pub fn max_num_batched_tokens(mut self, v: usize) -> Self { self.max_num_batched_tokens = Some(v); self }
    pub fn max_num_seqs(mut self, v: usize) -> Self { self.max_num_seqs = Some(v); self }
    pub fn block_size(mut self, v: usize) -> Self { self.block_size = Some(v); self }
    pub fn enable_prefix_caching(mut self, v: bool) -> Self { self.enable_prefix_caching = Some(v); self }
    pub fn enable_chunked_prefill(mut self, v: bool) -> Self { self.enable_chunked_prefill = Some(v); self }
    pub fn max_prefill_chunk_size(mut self, v: Option<usize>) -> Self { self.max_prefill_chunk_size = Some(v); self }
    pub fn enable_projection(mut self, v: bool) -> Self { self.enable_projection = Some(v); self }
    pub fn projection_lookahead(mut self, v: usize) -> Self { self.projection_lookahead = Some(v); self }
    pub fn min_guaranteed_blocks(mut self, v: usize) -> Self { self.min_guaranteed_blocks = Some(v); self }

    pub fn build(self) -> Result<SchedulerConfig> {
        let block_size = self.block_size.ok_or_else(|| anyhow::anyhow!("block_size required"))?;
        Ok(SchedulerConfig {
            max_seq_len: self.max_seq_len.ok_or_else(|| anyhow::anyhow!("max_seq_len required"))?,
            max_num_batched_tokens: self.max_num_batched_tokens.ok_or_else(|| anyhow::anyhow!("max_num_batched_tokens required"))?,
            max_num_seqs: self.max_num_seqs.ok_or_else(|| anyhow::anyhow!("max_num_seqs required"))?,
            block_size,
            enable_prefix_caching: self.enable_prefix_caching.ok_or_else(|| anyhow::anyhow!("enable_prefix_caching required"))?,
            enable_chunked_prefill: self.enable_chunked_prefill.ok_or_else(|| anyhow::anyhow!("enable_chunked_prefill required"))?,
            max_prefill_chunk_size: self.max_prefill_chunk_size.unwrap_or(None),
            enable_projection: self.enable_projection.unwrap_or(true),
            projection_lookahead: self.projection_lookahead.unwrap_or(2 * block_size),
            min_guaranteed_blocks: self.min_guaranteed_blocks.unwrap_or(3),
        })
    }
}

impl SchedulerConfig {
    pub fn builder() -> SchedulerConfigBuilder {
        SchedulerConfigBuilder::default()
    }
}

/// KV cache block manager wrapping BlockManager<G1> for the scheduler.
///
/// Ensures that block IDs returned to vLLM match KVBM's G1 block pool,
/// which is required for the ConnectorLeader to correctly map block IDs
/// to NIXL transfer descriptors for disaggregated prefill.
pub struct KVCacheManager {
    pub block_manager: Arc<BlockManager<G1>>,
    block_size: usize,
}

impl KVCacheManager {
    pub fn with_prefix_caching(
        block_manager: BlockManager<G1>,
        block_size: usize,
        _enable_prefix_caching: bool,
    ) -> Result<Self> {
        Ok(Self {
            block_manager: Arc::new(block_manager),
            block_size,
        })
    }

    fn allocate_and_register(&self, n_blocks: usize) -> Option<Vec<ImmutableBlock<G1>>> {
        if n_blocks == 0 {
            return Some(vec![]);
        }
        let mut_blocks = self.block_manager.allocate_blocks(n_blocks)?;
        let complete_blocks: Option<Vec<_>> = mut_blocks
            .into_iter()
            .map(|b| {
                // Use default hash since prefix caching is disabled (--no-enable-prefix-caching)
                let dummy_hash = kvbm_logical::SequenceHash::default();
                b.stage(dummy_hash, self.block_size).ok()
            })
            .collect();
        Some(self.block_manager.register_blocks(complete_blocks?))
    }

    pub fn total_blocks(&self) -> usize {
        self.block_manager.total_blocks()
    }

    pub fn available_blocks(&self) -> usize {
        self.block_manager.available_blocks()
    }

    pub fn cache_usage(&self) -> f32 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        let used = total - self.available_blocks();
        used as f32 / total as f32
    }
}

struct RunningRequest {
    token_ids: Vec<u32>,
    block_ids: Vec<BlockId>,
    _blocks: Vec<ImmutableBlock<G1>>,
    output_token_count: usize,
}

/// Rust scheduler backed by BlockManager<G1>.
///
/// When a ConnectorLeader is provided, `schedule()` calls
/// `process_scheduler_output` to generate `KvConnectorMetadata` which
/// contains the NIXL transfer descriptors workers need for disaggregated
/// prefill. Without this, decode workers never receive the onboarding
/// signal for requests requiring >1 KV block.
pub struct Scheduler {
    config: SchedulerConfig,
    kv_cache: KVCacheManager,
    connector: Option<Arc<ConnectorLeader>>,
    waiting: VecDeque<(String, Vec<u32>)>,
    running: HashMap<String, RunningRequest>,
    iteration: usize,
}

/// Builder for Scheduler.
#[derive(Default)]
pub struct SchedulerBuilder {
    config: Option<SchedulerConfig>,
    kv_cache: Option<KVCacheManager>,
    connector: Option<Arc<ConnectorLeader>>,
}

impl SchedulerBuilder {
    pub fn config(mut self, config: SchedulerConfig) -> Self {
        self.config = Some(config);
        self
    }

    pub fn kv_cache(mut self, kv_cache: KVCacheManager) -> Self {
        self.kv_cache = Some(kv_cache);
        self
    }

    pub fn connector(mut self, connector: Arc<ConnectorLeader>) -> Self {
        self.connector = Some(connector);
        self
    }

    pub fn build(self) -> Result<Scheduler> {
        Ok(Scheduler {
            config: self.config.ok_or_else(|| anyhow::anyhow!("config required"))?,
            kv_cache: self.kv_cache.ok_or_else(|| anyhow::anyhow!("kv_cache required"))?,
            connector: self.connector,
            waiting: VecDeque::new(),
            running: HashMap::new(),
            iteration: 0,
        })
    }
}

impl Scheduler {
    pub fn builder() -> SchedulerBuilder {
        SchedulerBuilder::default()
    }

    pub fn add_request(&mut self, request: Request) {
        let token_ids: Vec<u32> = Vec::from(request.tokens);
        self.waiting.push_back((request.request_id, token_ids));
    }

    pub fn schedule(&mut self) -> SchedulerOutput {
        self.iteration += 1;
        let mut output = SchedulerOutput::new(self.iteration);

        let max_new = self.config.max_num_seqs.saturating_sub(self.running.len());
        let max_tokens = self.config.max_num_batched_tokens;
        let mut tokens_scheduled = 0usize;
        let mut to_run: Vec<(String, Vec<u32>, Vec<BlockId>, Vec<ImmutableBlock<G1>>)> = vec![];

        while to_run.len() < max_new {
            let Some((req_id, token_ids)) = self.waiting.front() else { break };
            let n_tokens = token_ids.len();
            if tokens_scheduled + n_tokens > max_tokens && tokens_scheduled > 0 {
                break;
            }
            let n_blocks = (n_tokens + self.config.block_size - 1) / self.config.block_size;
            if let Some(blocks) = self.kv_cache.allocate_and_register(n_blocks) {
                let block_ids: Vec<BlockId> = blocks.iter().map(|b| b.block_id()).collect();
                tokens_scheduled += n_tokens;
                to_run.push((req_id.clone(), token_ids.clone(), block_ids, blocks));
                self.waiting.pop_front();
            } else {
                break;
            }
        }

        for (req_id, token_ids, block_ids, blocks) in to_run {
            let n = token_ids.len();
            output.add_new_request(
                req_id.clone(),
                token_ids.clone(),
                block_ids.clone(),
                0,
            );
            output.num_scheduled_tokens.insert(req_id.clone(), n);
            output.total_num_scheduled_tokens += n;
            self.running.insert(
                req_id,
                RunningRequest {
                    token_ids,
                    block_ids,
                    _blocks: blocks,
                    output_token_count: 0,
                },
            );
        }

        // Generate NIXL transfer metadata via the ConnectorLeader.
        // This fires the onboarding signal so decode workers know which
        // G1 block IDs to pull from prefill — without it, ISL≥24 hangs.
        if let Some(ref connector) = self.connector {
            if let Ok(metadata) = connector.process_scheduler_output(output.clone()) {
                output.kv_connector_metadata = Some(metadata);
            }
        }

        output
    }

    pub fn abort_request(&mut self, request_id: &str) {
        self.waiting.retain(|(id, _)| id != request_id);
        self.running.remove(request_id);
    }

    pub fn finish_requests(&mut self, request_ids: &[String], _status: RequestStatus) {
        for id in request_ids {
            self.running.remove(id);
        }
    }

    pub fn update_from_output(
        &mut self,
        finished_ids: &[String],
        output_tokens: &HashMap<String, Vec<u32>>,
    ) {
        for id in finished_ids {
            self.running.remove(id);
        }
        for (id, tokens) in output_tokens {
            if let Some(req) = self.running.get_mut(id) {
                req.output_token_count += tokens.len();
            }
        }
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn cache_usage(&self) -> f32 {
        self.kv_cache.cache_usage()
    }
}
