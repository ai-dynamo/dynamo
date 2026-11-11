// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replication wire between slot trackers: active-sequence, active-block, and prefill events
//! with the batch limits both ends share, and the scheduler-load report one tracker publishes
//! about a rank's booked load. Both ends of this wire are slot trackers; the mesh that
//! carries it is a `dynamo-kv-router` adapter.

use std::time::Duration;

use dynamo_tokens::SequenceHash;
use serde::{Deserialize, Serialize};

use dynamo_router_protocol::worker::{DpRank, WorkerId, WorkerWithDpRank};

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

/// One router scheduler's view of a worker rank's in-flight load.
///
/// Routers publish this on the scheduler-load subject; workers never do.
/// Fields added after the first release must be `#[serde(default)]`, and `Option`
/// when absent does not mean zero.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchedulerLoad {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
    pub active_decode_blocks: u64,
    pub active_prefill_tokens: u64,
    pub group: SchedulerGroup,
}

/// Which schedulers' views describe the same requests.
///
/// Views within one group overlap, so consumers take their maximum; views from
/// different groups are disjoint, so consumers sum them.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SchedulerGroup {
    /// A scheduler that only knows the requests it routed itself.
    Standalone { scheduler_id: u64 },
    /// Schedulers that replicate request state, so each reports the group's full load.
    ReplicaGroup { group_id: String },
    /// Future aggregation kinds are decoded but cannot contribute to load totals.
    #[serde(other)]
    Unknown,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_load_accepts_unknown_fields_but_requires_identity_and_measurements() {
        let frame = serde_json::json!({
            "worker_id": 7,
            "dp_rank": 2,
            "active_decode_blocks": 10,
            "active_prefill_tokens": 20,
            "group": {"kind": "standalone", "scheduler_id": 42, "future_group_field": true},
            "future_measurement": 99
        });
        let batch: Vec<SchedulerLoad> =
            rmp_serde::from_slice(&rmp_serde::to_vec_named(&vec![frame.clone()]).unwrap()).unwrap();
        assert_eq!(
            batch,
            [SchedulerLoad {
                worker_id: 7,
                dp_rank: 2,
                active_decode_blocks: 10,
                active_prefill_tokens: 20,
                group: SchedulerGroup::Standalone { scheduler_id: 42 },
            }]
        );
        for required in [
            "worker_id",
            "dp_rank",
            "group",
            "active_decode_blocks",
            "active_prefill_tokens",
        ] {
            let mut incomplete = frame.clone();
            incomplete.as_object_mut().unwrap().remove(required);
            assert!(
                rmp_serde::from_slice::<Vec<SchedulerLoad>>(
                    &rmp_serde::to_vec_named(&vec![incomplete]).unwrap(),
                )
                .is_err(),
                "missing {required} must not become a valid zero-load report"
            );
        }
    }
}
