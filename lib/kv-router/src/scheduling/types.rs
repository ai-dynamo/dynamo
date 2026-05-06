// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::RouterConfigOverride;
use crate::protocols::{DpRank, SharedCacheHits, WorkerConfigLike, WorkerId, WorkerWithDpRank};
use crate::sequences::PrefillTokenDeltas;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TierOverlapBlocks {
    pub host_pinned: HashMap<WorkerWithDpRank, usize>,
    pub disk: HashMap<WorkerWithDpRank, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialLoad {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub potential_prefill_tokens: usize,
    pub potential_decode_blocks: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum KvSchedulerError {
    #[error("no endpoints available to route work")]
    NoEndpoints,

    #[error("pinned worker {worker_id} is not in allowed worker set")]
    PinnedWorkerNotAllowed { worker_id: WorkerId },

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub effective_overlap_blocks: f64,
    pub cached_tokens: usize,
}

pub struct SchedulingRequest {
    // Request identity and payload.
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub lora_name: Option<String>,
    pub expected_output_tokens: Option<u32>,

    // Routing constraints and request-level config.
    pub pinned_worker: Option<WorkerWithDpRank>,
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub router_config_override: Option<RouterConfigOverride>,
    pub track_prefill_tokens: bool,
    pub priority_jump: f64,

    // Overlap and cache signals.
    pub tier_overlap_blocks: TierOverlapBlocks,
    pub effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
    pub shared_cache_hits: Option<SharedCacheHits>,

    // Load state computed during admission.
    pub decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: FxHashMap<WorkerWithDpRank, usize>,

    // Scheduling side effects and lifecycle controls.
    pub update_states: bool,
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

impl SchedulingRequest {
    pub(crate) fn prefill_token_deltas(&self) -> PrefillTokenDeltas {
        if !self.track_prefill_tokens {
            return PrefillTokenDeltas::none();
        }

        let by_worker = self
            .effective_cached_tokens
            .iter()
            .map(|(worker, cached_tokens)| {
                let delta = self
                    .isl_tokens
                    .checked_sub(*cached_tokens)
                    .unwrap_or_else(|| {
                        tracing::error!(
                            "prefill_tokens < 0 with ISL {} < cached_tokens {}, returning 0",
                            self.isl_tokens,
                            cached_tokens
                        );
                        0
                    });
                (*worker, delta)
            })
            .collect();

        PrefillTokenDeltas::new(self.isl_tokens, by_worker)
    }

    pub(crate) fn best_effective_prefill_tokens(&self) -> usize {
        let cached_tokens = match self.pinned_worker {
            Some(worker) => self.effective_cached_tokens_for(worker),
            None => self
                .effective_cached_tokens
                .iter()
                .filter(|(worker, _)| self.is_worker_allowed(worker.worker_id))
                .map(|(_, cached_tokens)| *cached_tokens)
                .max()
                .unwrap_or(0),
        };

        self.isl_tokens.saturating_sub(cached_tokens)
    }

    pub(crate) fn effective_cached_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        self.effective_cached_tokens
            .get(&worker)
            .copied()
            .unwrap_or(0)
    }

    pub(crate) fn effective_overlap_blocks_for(&self, worker: WorkerWithDpRank) -> f64 {
        self.effective_overlap_blocks
            .get(&worker)
            .copied()
            .unwrap_or(0.0)
    }

    pub(crate) fn is_worker_allowed(&self, worker_id: WorkerId) -> bool {
        self.allowed_worker_ids
            .as_ref()
            .is_none_or(|ids| ids.contains(&worker_id))
    }

    pub(crate) fn prefill_tokens_for(&self, worker: WorkerWithDpRank) -> usize {
        let default_prefill_tokens = if self.track_prefill_tokens {
            self.isl_tokens
        } else {
            0
        };
        self.prefill_tokens
            .get(&worker)
            .copied()
            .unwrap_or(default_prefill_tokens)
    }

    pub(crate) fn request_blocks(&self, block_size: u32) -> u64 {
        self.isl_tokens.div_ceil(block_size as usize) as u64
    }

    pub fn validate_worker_constraints(&self) -> Result<(), KvSchedulerError> {
        let Some(pinned_worker) = self.pinned_worker else {
            return Ok(());
        };
        let Some(allowed_worker_ids) = self.allowed_worker_ids.as_ref() else {
            return Ok(());
        };
        if allowed_worker_ids.contains(&pinned_worker.worker_id) {
            return Ok(());
        }

        Err(KvSchedulerError::PinnedWorkerNotAllowed {
            worker_id: pinned_worker.worker_id,
        })
    }

    pub fn bypass_capacity_check(&self) -> bool {
        self.pinned_worker.is_none() && self.allowed_worker_ids.is_some()
    }

    pub fn respond(&mut self, result: Result<SchedulingResponse, KvSchedulerError>) {
        let Some(tx) = self.resp_tx.take() else {
            tracing::error!("respond called multiple times on same request");
            return;
        };
        if tx.send(result).is_err() {
            tracing::error!("failed to send response to requestor");
        }
    }
}

pub fn pinned_worker_config<C: WorkerConfigLike>(
    workers: &HashMap<WorkerId, C>,
    worker: WorkerWithDpRank,
) -> Result<&C, KvSchedulerError> {
    let Some(config) = workers.get(&worker.worker_id) else {
        return Err(KvSchedulerError::NoEndpoints);
    };
    let dp_start_rank = config.data_parallel_start_rank();
    let dp_end_rank = dp_start_rank + config.data_parallel_size();
    if !(dp_start_rank..dp_end_rank).contains(&worker.dp_rank) {
        return Err(KvSchedulerError::NoEndpoints);
    }

    Ok(config)
}
