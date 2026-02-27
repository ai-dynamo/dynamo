// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};

use crate::config::RouterConfigOverride;
use crate::protocols::{DpRank, OverlapScores, WorkerId, WorkerWithDpRank};

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

    #[error("endpoint subscriber shutdown")]
    SubscriberShutdown,

    #[error("failed to initialize event publisher: {0}")]
    InitFailed(String),
}

#[derive(Debug)]
pub struct SchedulingResponse {
    pub best_worker: WorkerWithDpRank,
    pub overlap_blocks: u32,
}

pub struct SchedulingRequest {
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    pub router_config_override: Option<RouterConfigOverride>,
    pub update_states: bool,
    pub lora_name: Option<String>,
    /// Priority jump in seconds; decreases effective arrival time in the queue.
    pub priority_jump: f64,
    /// Optional set of allowed worker IDs to restrict routing decisions (EPP).
    pub allowed_worker_ids: Option<HashSet<WorkerId>>,
    pub resp_tx: Option<tokio::sync::oneshot::Sender<Result<SchedulingResponse, KvSchedulerError>>>,
}

impl SchedulingRequest {
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
