// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replication wire between slot trackers: active-sequence, active-block, and prefill events
//! with the batch limits both ends share. Both ends of this wire are slot trackers; the
//! mesh that carries it is a `dynamo-kv-router` adapter.

use std::time::Duration;

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};

use dynamo_router_protocol::worker::{WorkerId, WorkerWithDpRank};

const fn default_track_prefill_tokens() -> bool {
    true
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct PrefillEvent {
    pub request_id: String,
    pub worker_id: WorkerId,
    pub data: PrefillEventData,
    pub router_id: u64,
}

/// Represents the different stages of prefilling tokens for a request.
///
/// Each variant contains a `usize` representing the number of tokens
/// that are pending prefill in the request.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum PrefillEventData {
    NewPrefill(usize),
    UpdatePrefill(usize),
    CompletePrefill,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActiveSequenceEvent {
    pub request_id: String,
    pub worker: WorkerWithDpRank,
    pub data: ActiveSequenceEventData,
    /// Source DRT identity, used to suppress a publisher's own echo. Router events use the router
    /// ID; worker-origin completion marks use the worker ID.
    pub router_id: u64,
    pub lora_name: Option<String>,
}

/// Active-sequence lifecycle events carried in publisher-queue arrival order.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActiveSequenceEventBatch {
    pub events: Vec<ActiveSequenceEvent>,
}

/// Shared cooperative batch limits for active-sequence replica sync.
pub const MAX_REPLICA_BATCH_EVENTS: usize = 256;
pub const MAX_REPLICA_BATCH_DURATION: Duration = Duration::from_millis(1);

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
pub struct PrefillLoadHint {
    pub initial_effective_prefill_tokens: usize,
    pub expected_prefill_duration: Option<Duration>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActiveSequenceEventData {
    AddRequest {
        token_sequence: Option<Vec<SequenceHash>>,
        #[serde(default = "default_track_prefill_tokens")]
        track_prefill_tokens: bool,
        expected_output_tokens: Option<u32>,
        prefill_load_hint: Option<PrefillLoadHint>,
    },
    // NOTE: Output-block growth is intentionally not a replica-sync event. It can occur
    // at high frequency, and broadcasting it would consume disproportionate network bandwidth.
    Free,
    MarkPrefillCompleted,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ActiveBlockEvent {
    pub request_id: String,
    pub data: ActiveBlockEventData,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum ActiveBlockEventData {
    NewBlock(Vec<SequenceHash>),
    FreeBlock,
}
