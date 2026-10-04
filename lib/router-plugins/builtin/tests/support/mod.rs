// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{OverlapSignals, ScheduleMode, SchedulingRequest};
use dynamo_kv_router::{WorkerLoadProjection, WorkerSelectionInput};
use std::collections::HashMap;

#[derive(Clone, Copy)]
pub struct TestWorker;
impl WorkerConfigLike for TestWorker {
    fn data_parallel_start_rank(&self) -> u32 {
        0
    }
    fn data_parallel_size(&self) -> u32 {
        2
    }
    fn max_num_batched_tokens(&self) -> Option<u64> {
        None
    }
    fn total_kv_blocks(&self) -> Option<u64> {
        Some(16384)
    }
}

pub fn fixture(
    count: usize,
    prompt_tokens: usize,
) -> (HashMap<u64, TestWorker>, SchedulingRequest) {
    let mut request = SchedulingRequest {
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: prompt_tokens,
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
    };
    let workers = (0..count as u64).map(|id| (id, TestWorker)).collect();
    for id in 0..count as u64 {
        for dp_rank in 0..2 {
            let w = WorkerWithDpRank::new(id, dp_rank);
            let overlap = (id as usize * 7 + dp_rank as usize) % 9;
            request
                .overlap
                .tier_overlap_blocks
                .device
                .insert(w, overlap);
            request.overlap.tier_overlap_blocks.host_pinned.insert(w, 2);
            request.overlap.tier_overlap_blocks.disk.insert(w, 1);
            request
                .overlap
                .effective_overlap_blocks
                .insert(w, overlap as f64 + 0.5);
            request
                .overlap
                .effective_cached_tokens
                .insert(w, overlap * 16 + 8);
            request.worker_loads.insert(
                w,
                WorkerLoadProjection {
                    active_requests: id as usize % 5,
                    active_prefill_tokens: (id as usize % 7) * 19,
                    active_decode_blocks: id as usize % 11,
                    additional_active_blocks: 3,
                },
            );
        }
    }
    (workers, request)
}

pub fn selection_input<'a>(
    workers: &'a HashMap<u64, TestWorker>,
    request: &'a SchedulingRequest,
    block_size: u32,
) -> WorkerSelectionInput<'a, TestWorker> {
    WorkerSelectionInput::configured(workers, request, request.eligibility(), block_size)
}
