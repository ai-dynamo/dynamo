// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Selection fixtures for the ported policies' unit tests.

use std::collections::HashMap;

use dynamo_kv_router::plugins::worker_selection::WorkerSelectionPolicy;
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use dynamo_kv_router::{WorkerLoadProjection, WorkerSelectionInput, WorkerSelector};

pub(crate) const BLOCK_SIZE: u32 = 16;

pub(crate) struct TestWorker {
    total_kv_blocks: Option<u64>,
}

impl WorkerConfigLike for TestWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        1
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        self.total_kv_blocks
    }
}

/// One worker's observable state for a selection.
#[derive(Clone, Copy, Default)]
pub(crate) struct Worker {
    pub(crate) id: u64,
    pub(crate) device_blocks: usize,
    pub(crate) active_requests: usize,
    pub(crate) active_prefill_tokens: usize,
    pub(crate) decode_blocks: usize,
    pub(crate) total_kv_blocks: Option<u64>,
}

impl Worker {
    pub(crate) fn new(id: u64) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }
    pub(crate) fn cached(mut self, device_blocks: usize) -> Self {
        self.device_blocks = device_blocks;
        self
    }
    pub(crate) fn requests(mut self, active_requests: usize) -> Self {
        self.active_requests = active_requests;
        self
    }
    pub(crate) fn prefill(mut self, active_prefill_tokens: usize) -> Self {
        self.active_prefill_tokens = active_prefill_tokens;
        self
    }
    pub(crate) fn decode(mut self, decode_blocks: usize, total_kv_blocks: u64) -> Self {
        self.decode_blocks = decode_blocks;
        self.total_kv_blocks = Some(total_kv_blocks);
        self
    }
}

/// A request over `prompt_blocks` full blocks. `prefix` seeds its chained prefix hashes, so
/// requests built with the same seed share every prefix.
pub(crate) fn request(prompt_blocks: usize, prefix: u64) -> SchedulingRequest {
    SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: Some(
            (0..prompt_blocks as u64)
                .map(|i| prefix * 1_000 + i)
                .collect(),
        ),
        isl_tokens: prompt_blocks * BLOCK_SIZE as usize,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: None,
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: None,
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    }
}

/// Select one worker for `request` given each worker's state.
pub(crate) fn select(
    policy: &WorkerSelectionPolicy,
    mut request: SchedulingRequest,
    workers: &[Worker],
) -> u64 {
    for worker in workers {
        let rank = WorkerWithDpRank::from_worker_id(worker.id);
        request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(rank, worker.device_blocks);
        request.worker_loads.insert(
            rank,
            WorkerLoadProjection {
                active_requests: worker.active_requests,
                active_prefill_tokens: worker.active_prefill_tokens,
                active_decode_blocks: worker.decode_blocks,
                ..Default::default()
            },
        );
    }
    let configs: HashMap<u64, TestWorker> = workers
        .iter()
        .map(|worker| {
            (
                worker.id,
                TestWorker {
                    total_kv_blocks: worker.total_kv_blocks,
                },
            )
        })
        .collect();
    policy
        .select_worker(WorkerSelectionInput::configured(
            &configs,
            &request,
            request.eligibility(),
            BLOCK_SIZE,
        ))
        .unwrap()
        .worker
        .worker_id
}
