// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{collections::HashMap, path::Path};

use dynamo_kv_router::plugins::prelude::*;
use dynamo_scheduler::host::{WorkerSelectionInput, WorkerSelector};
use dynamo_slot_tracker::host::{
    ActiveSequencesMultiWorker, DecodeProgress, DecodeProgressUpdate, NoopSequencePublisher,
    SchedulerBookingDescriptor, SequenceRequest,
};
use tokio::time::Instant;

pub const BLOCK_SIZE: u32 = 64;
pub const SOURCE: WorkerWithDpRank = WorkerWithDpRank {
    worker_id: 29,
    dp_rank: 0,
};
pub const ALTERNATIVE: WorkerWithDpRank = WorkerWithDpRank {
    worker_id: 41,
    dp_rank: 0,
};
pub type NativeSlots = ActiveSequencesMultiWorker<NoopSequencePublisher>;

pub struct TestWorker;
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
        Some(1_000_000)
    }
}

pub fn workers() -> HashMap<u64, TestWorker> {
    HashMap::from([
        (SOURCE.worker_id, TestWorker),
        (ALTERNATIVE.worker_id, TestWorker),
    ])
}

pub fn request() -> SchedulingRequest {
    let mut request = SchedulingRequest {
        accounting_attribution: Default::default(),
        mode: ScheduleMode::QueryOnly { request_id: None },
        token_seq: None,
        isl_tokens: 16_385,
        lora_name: None,
        expected_output_tokens: None,
        affinity_target: Some(SOURCE.into()),
        pinned_worker: None,
        allowed_worker_ids: None,
        routing_constraints: RoutingConstraints::default(),
        router_config_override: None,
        track_prefill_tokens: true,
        priority_jump: 0.0,
        strict_priority: 0,
        policy_class: None,
        session_context: Some(SessionContext::new(
            "child".into(),
            Some("parent".into()),
            None,
            None,
        )),
        overlap: OverlapSignals::default(),
        kv_transfer_candidates: None,
        retain_kv_transfer_chain: false,
        shared_cache_hits: None,
        worker_loads: Default::default(),
        resp_tx: None,
    };
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(SOURCE, 256);
    request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(ALTERNATIVE, 0);
    request
}

pub fn backgrounds(limit: u32, generated: usize) -> (NativeSlots, Vec<SchedulerBookingDescriptor>) {
    let slots = NativeSlots::new_without_expiry(
        NoopSequencePublisher,
        BLOCK_SIZE as usize,
        HashMap::from([(SOURCE.worker_id, (0, 1)), (ALTERNATIVE.worker_id, (0, 1))]),
        false,
        1,
        "aggregated",
    );
    let owners = (0..8)
        .map(|index| {
            let request_id = format!("background-{index}");
            let now = Instant::now();
            let attempt_id = slots
                .add_request_if_registered_admitted(
                    SequenceRequest {
                        request_id: request_id.clone(),
                        token_sequence: None,
                        expected_output_tokens: None,
                        track_prefill_tokens: false,
                        prefill_load_hint: None,
                        worker: SOURCE,
                        lora_name: None,
                    },
                    now,
                )
                .unwrap();
            slots
                .mark_prefill_completed_if_booking(&request_id, SOURCE, attempt_id, now)
                .unwrap();
            slots
                .update_output_if_booking(
                    &request_id,
                    SOURCE,
                    attempt_id,
                    0,
                    None,
                    Some(DecodeProgressUpdate::observed(limit, generated)),
                )
                .unwrap();
            SchedulerBookingDescriptor {
                request_id,
                worker: SOURCE,
                attempt_id,
            }
        })
        .collect();
    (slots, owners)
}

pub fn policy() -> WorkerSelectionPolicy {
    policy_for_role(WorkerType::Aggregated)
}

pub fn policy_for_role(role: WorkerType) -> WorkerSelectionPolicy {
    policy_from_path(
        Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/worker-selection.yaml"
        )),
        role,
    )
}

pub fn policy_with_age(max_age_ms: u64) -> WorkerSelectionPolicy {
    let yaml = include_str!("../../worker-selection.yaml").replace(
        "max_progress_age_ms: 100",
        &format!("max_progress_age_ms: {max_age_ms}"),
    );
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), yaml).unwrap();
    policy_from_path(file.path(), WorkerType::Aggregated)
}

fn policy_from_path(path: &Path, role: WorkerType) -> WorkerSelectionPolicy {
    let config = KvRouterConfig {
        router_policy_config: Some(path.to_string_lossy().into_owned()),
        ..Default::default()
    };
    let mut registry = RouterPluginRegistry::default();
    dynamo_custom_policy_example_hotspot_progress::register(&mut registry).unwrap();
    let factory = registry
        .resolve_for_worker_type(&config, WorkerType::Aggregated)
        .unwrap()
        .unwrap();
    factory(
        &SelectionConfig::from(&config),
        role,
        RoutingPartitionRef::new("test", "default"),
    )
}

pub fn select(
    policy: &WorkerSelectionPolicy,
    slots: &NativeSlots,
    request: &mut SchedulingRequest,
) -> WorkerWithDpRank {
    request.worker_loads = slots.project_worker_loads(None, Instant::now());
    let provider = |worker| slots.decode_progress_for(worker);
    select_with_provider(policy, &workers(), request, Some(&provider))
}

pub fn select_with_provider(
    policy: &WorkerSelectionPolicy,
    workers: &HashMap<u64, TestWorker>,
    request: &SchedulingRequest,
    provider: Option<&dyn Fn(WorkerWithDpRank) -> Option<DecodeProgress>>,
) -> WorkerWithDpRank {
    policy
        .select_worker(
            WorkerSelectionInput::configured(workers, request, request.eligibility(), BLOCK_SIZE)
                .with_decode_progress(provider),
        )
        .unwrap()
        .worker
}
