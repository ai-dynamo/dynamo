// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker load reports on the wire: scheduler views and worker-published active load.

use serde::{Deserialize, Serialize};

use crate::worker::{DpRank, WorkerId};

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

/// Active load metrics for a worker, used for overload detection.
///
/// Published by workers (with `kv_used_blocks`) and by the scheduler (with
/// `active_decode_blocks` and `active_prefill_tokens`).
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
pub struct ActiveLoad {
    pub worker_id: WorkerId,
    #[serde(default)]
    pub dp_rank: DpRank,
    /// Scheduler-reported decode block load.
    pub active_decode_blocks: Option<u64>,
    /// Number of active prefill tokens (from scheduler's view).
    pub active_prefill_tokens: Option<u64>,
    /// Total KV blocks currently in use on the worker.
    ///
    /// This is published by workers only and is the authoritative signal for
    /// backend KV occupancy used by overload detection.
    pub kv_used_blocks: Option<u64>,
}

/// Prefill budget assumed for a rank that does not report `max_num_batched_tokens`. Large
/// enough that a fractional prefill threshold never fires on an unreported budget.
pub const DEFAULT_MAX_BATCHED_TOKENS: u64 = 10_000_000;

/// Advertised capacity of one worker rank, from its runtime config. It changes only when
/// the worker registers again, so hosts serve it from cached discovery state, not per request.
/// The values cross the process boundary in the worker's runtime config, and the slot
/// tracker, worker selection, and admission all name the type, so it lives in the wire crate.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WorkerCapacity {
    max_num_batched_tokens: Option<u64>,
    total_kv_blocks: Option<u64>,
}

impl WorkerCapacity {
    /// A reported zero means unreported. vLLM registers `total_kv_blocks = 0` as a placeholder
    /// under the Ray data-parallel backend, and a zero budget would mark every rank full.
    pub fn new(max_num_batched_tokens: Option<u64>, total_kv_blocks: Option<u64>) -> Self {
        Self {
            max_num_batched_tokens: max_num_batched_tokens.filter(|&value| value > 0),
            total_kv_blocks: total_kv_blocks.filter(|&value| value > 0),
        }
    }

    /// Prefill tokens the rank batches per engine step, when reported.
    pub fn max_num_batched_tokens(self) -> Option<u64> {
        self.max_num_batched_tokens
    }

    /// Total KV blocks the worker advertises for this rank, when reported: capacity, not free
    /// blocks. Some adapters derive it from aggregate data (vLLM data parallelism publishes
    /// the process total divided by the rank count), so treat it as approximate.
    pub fn total_kv_blocks(self) -> Option<u64> {
        self.total_kv_blocks
    }

    /// The budget fractional prefill thresholds are taken of: `max_num_batched_tokens`, or
    /// [`DEFAULT_MAX_BATCHED_TOKENS`] when the rank does not report it.
    pub fn prefill_budget(self) -> u64 {
        self.max_num_batched_tokens
            .unwrap_or(DEFAULT_MAX_BATCHED_TOKENS)
    }
}

/// A rank's booked load as a fullness check reads it. The slot tracker's projection
/// implements it, and so does worker selection's load column built from that projection,
/// so one check serves both without either stage naming the other.
pub trait RankLoad {
    /// Tokens active in the rank's prefill stage.
    fn active_prefill_tokens(&self) -> usize;
    /// KV blocks the rank's active requests hold.
    fn active_decode_blocks(&self) -> usize;
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
