// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_tokens::SequenceHash;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use super::config::RouterConfigOverride;
use crate::protocols::{DpRank, SharedCacheHits, WorkerConfigLike, WorkerId, WorkerWithDpRank};

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
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub tier_overlap_blocks: TierOverlapBlocks,
    pub effective_overlap_blocks: HashMap<WorkerWithDpRank, f64>,
    pub effective_cached_tokens: HashMap<WorkerWithDpRank, usize>,
    /// Per-worker tree size, used only for tie-breaking when multiple workers
    /// produce the same logit at temperature=0.
    pub tree_sizes: HashMap<WorkerWithDpRank, usize>,
    pub decode_blocks: FxHashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: FxHashMap<WorkerWithDpRank, usize>,
    pub track_prefill_tokens: bool,
    pub router_config_override: Option<RouterConfigOverride>,
    pub update_states: bool,
    pub lora_name: Option<String>,
    /// Priority jump in seconds; decreases effective arrival time in the queue.
    pub priority_jump: f64,
    /// Expected output tokens from agent_hints.osl, forwarded to the slot tracker
    /// for output block decay estimation.
    pub expected_output_tokens: Option<u32>,
    /// Exact worker/rank pin used by scheduler queueing, WSPT, and selection.
    pub pinned_worker: Option<WorkerWithDpRank>,
    /// Optional set of allowed worker IDs to restrict routing decisions (EPP).
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    /// Shared cache hit information from an external shared KV cache pool.
    /// When present, the selector adjusts prefill cost by weighting shared hits
    /// beyond each worker's device prefix.
    pub shared_cache_hits: Option<SharedCacheHits>,
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

impl SchedulingRequest {
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
