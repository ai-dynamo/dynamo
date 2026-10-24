// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::progress_group::ProgressGroupParameters;
use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
use dynamo_kv_router::scheduling::{
    OverlapSignals, ScheduleMode, SessionContext, decode_progress::DecodeProgressUpdate,
    queue::SchedulerBookingDescriptor,
};
use dynamo_kv_router::{
    ActiveSequencesMultiWorker, NoopSequencePublisher, SchedulingRequest, SequenceRequest,
    WorkerLoadProjection, WorkerSelectionInput, WorkerSelector,
};
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use std::time::Duration;

static NEXT_WORKER: AtomicU64 = AtomicU64::new((1 << 62) + 100);

struct Worker;
impl WorkerConfigLike for Worker {
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

fn parameters() -> ProgressGroupParameters {
    ProgressGroupParameters {
        source_active_requests: 8,
        min_prefix_tokens: 16_384,
        min_remaining_tokens: 256,
        max_progress_age_ms: 100,
    }
}

type NativeSlots = ActiveSequencesMultiWorker<NoopSequencePublisher>;

struct NativeOwner {
    slots: Arc<NativeSlots>,
    booking: SchedulerBookingDescriptor,
    budget: Option<u32>,
}
impl NativeOwner {
    fn advance(&self, tokens: usize) {
        if let Some(budget) = self.budget {
            self.slots
                .update_decode_progress_for_benchmark(
                    &self.booking,
                    DecodeProgressUpdate::observed(budget, tokens),
                )
                .unwrap();
        }
    }
}
impl Drop for NativeOwner {
    fn drop(&mut self) {
        self.slots
            .free(&self.booking.request_id, tokio::time::Instant::now())
            .unwrap();
    }
}

struct Frame {
    workers: HashMap<u64, Worker>,
    request: SchedulingRequest,
    source: WorkerWithDpRank,
    other: WorkerWithDpRank,
    guards: Vec<NativeOwner>,
    slots: Arc<NativeSlots>,
    block_size: u32,
}

impl Frame {
    fn new(budget: Option<u32>, ignore_eos: bool) -> Self {
        let id = NEXT_WORKER.fetch_add(10, Ordering::Relaxed);
        let source = WorkerWithDpRank::from_worker_id(id);
        let other = WorkerWithDpRank::from_worker_id(id + 1);
        let mut request = SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16_385,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: Some(source.into()),
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
                Some("arbitrary-parent".into()),
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
            .insert(source, 256);
        request.overlap.tier_overlap_blocks.device.insert(other, 0);
        request.worker_loads.insert(
            source,
            WorkerLoadProjection {
                active_requests: 8,
                ..Default::default()
            },
        );
        request
            .worker_loads
            .insert(other, WorkerLoadProjection::default());
        let slots = Arc::new(NativeSlots::new_without_expiry(
            NoopSequencePublisher,
            64,
            HashMap::from([(id, (0, 1)), (id + 1, (0, 1))]),
            false,
            1,
            "aggregated",
        ));
        let guards = (0..8)
            .map(|i| {
                let request_id = format!("progress-test-{id}-{i}");
                let now = tokio::time::Instant::now();
                let attempt_id = slots
                    .add_request_if_registered_admitted(
                        SequenceRequest {
                            request_id: request_id.clone(),
                            token_sequence: Some(Vec::new()),
                            expected_output_tokens: None,
                            track_prefill_tokens: false,
                            prefill_load_hint: None,
                            worker: source,
                            lora_name: None,
                        },
                        now,
                    )
                    .unwrap();
                slots.mark_prefill_completed(&request_id, now).unwrap();
                let owner = NativeOwner {
                    slots: Arc::clone(&slots),
                    booking: SchedulerBookingDescriptor {
                        request_id,
                        worker: source,
                        attempt_id,
                    },
                    budget: budget.filter(|_| ignore_eos),
                };
                owner.advance(8);
                owner
            })
            .collect();
        Self {
            workers: HashMap::from([(id, Worker), (id + 1, Worker)]),
            request,
            source,
            other,
            guards,
            slots,
            block_size: 64,
        }
    }

    fn select(&self, parameters: Option<ProgressGroupParameters>) -> WorkerWithDpRank {
        self.select_with_slots(parameters, &self.slots)
    }

    fn select_with_slots(
        &self,
        parameters: Option<ProgressGroupParameters>,
        slots: &NativeSlots,
    ) -> WorkerWithDpRank {
        let policy = WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "prefill",
            Vec::new(),
            Box::new(TwoTierCostFnPicker {
                parameters: Parameters {
                    progress_group: parameters,
                    ..Default::default()
                },
            }),
        );
        let provider = |worker| slots.decode_progress_for_benchmark(worker);
        policy
            .select_worker_with_decode_progress(
                WorkerSelectionInput::configured(
                    &self.workers,
                    &self.request,
                    self.request.eligibility(),
                    self.block_size,
                ),
                None,
                Some(&provider),
            )
            .unwrap()
            .0
            .worker
    }
}

#[test]
fn progress_group_disabled_preserves_original_cache_choice() {
    let frame = Frame::new(Some(128), true);
    assert_eq!(frame.select(None), frame.source);
    assert_eq!(frame.select(Some(parameters())), frame.other);
}

#[test]
fn progress_group_short_decode_keeps_warm_source() {
    for budget in [16, 24, 32] {
        let frame = Frame::new(Some(budget), true);
        assert_eq!(frame.select(Some(parameters())), frame.source);
    }
}

#[test]
fn progress_group_budget_equality_is_not_a_move() {
    let frame = Frame::new(Some(40), true); // 8 * (40 - 8) == 256
    assert_eq!(frame.select(Some(parameters())), frame.source);
}

#[test]
fn progress_group_missing_partial_unknown_or_eos_progress_falls_back() {
    for (budget, eos) in [(None, true), (Some(128), false)] {
        let frame = Frame::new(budget, eos);
        assert_eq!(frame.select(Some(parameters())), frame.source);
    }
    let mut frame = Frame::new(Some(128), true);
    frame.guards.pop();
    assert_eq!(frame.select(Some(parameters())), frame.source);
    frame.guards.clear();
    assert_eq!(frame.select(Some(parameters())), frame.source);
}

#[test]
fn progress_group_stale_progress_falls_back() {
    let frame = Frame::new(Some(128), true);
    std::thread::sleep(Duration::from_millis(3));
    let mut config = parameters();
    config.max_progress_age_ms = 1;
    assert_eq!(frame.select(Some(config)), frame.source);
}

#[test]
fn progress_group_invalid_observation_is_isolated_to_eligible_workers() {
    let broken = Frame::new(Some(128), true);
    broken.guards[0].advance(7);
    assert_eq!(broken.select(Some(parameters())), broken.source);
    let healthy = Frame::new(Some(128), true);
    assert_eq!(healthy.select(Some(parameters())), healthy.other);
}

#[test]
fn progress_group_explicit_pin_and_allowlist_remain_host_owned() {
    let mut frame = Frame::new(Some(128), true);
    frame.request.pinned_worker = Some(frame.source);
    assert_eq!(frame.select(Some(parameters())), frame.source);
    frame.request.pinned_worker = Some(frame.other);
    assert_eq!(frame.select(Some(parameters())), frame.other);
    frame.request.pinned_worker = None;
    frame.request.allowed_worker_ids = Some([frame.source.worker_id].into_iter().collect());
    assert_eq!(frame.select(Some(parameters())), frame.source);
}

#[test]
fn progress_group_non_sibling_or_busy_alternative_falls_back() {
    let mut frame = Frame::new(Some(128), true);
    frame.request.session_context = None;
    assert_eq!(frame.select(Some(parameters())), frame.source);
    frame.request.session_context = Some(SessionContext::new(
        "c".into(),
        Some("p".into()),
        None,
        None,
    ));
    frame
        .request
        .worker_loads
        .get_mut(&frame.other)
        .unwrap()
        .active_requests = 2;
    assert_eq!(frame.select(Some(parameters())), frame.source);
}

#[test]
fn progress_group_prefix_units_do_not_depend_on_257_blocks() {
    for block_size in [32, 64, 128] {
        let mut frame = Frame::new(Some(128), true);
        frame.block_size = block_size;
        frame.request.isl_tokens = 32_769;
        frame
            .request
            .overlap
            .tier_overlap_blocks
            .device
            .insert(frame.source, 32_768 / block_size as usize);
        assert_eq!(frame.select(Some(parameters())), frame.other);
    }
}

#[test]
fn progress_group_existing_cold_target_stays_during_first_sibling_prefill() {
    let mut frame = Frame::new(Some(128), true);
    assert_eq!(frame.select(Some(parameters())), frame.other);
    frame.request.affinity_target = Some(frame.other.into());
    let load = frame.request.worker_loads.get_mut(&frame.other).unwrap();
    load.active_requests = 1;
    load.active_prefill_tokens = 16_385;
    assert_eq!(frame.select(Some(parameters())), frame.other);
    frame
        .request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(frame.other, 256);
    frame
        .request
        .worker_loads
        .get_mut(&frame.other)
        .unwrap()
        .active_prefill_tokens = 0;
    assert_eq!(frame.select(Some(parameters())), frame.other);
}

#[test]
fn progress_group_first_warm_sibling_does_not_split_the_next_child() {
    let mut frame = Frame::new(Some(32), true);
    let load = frame.request.worker_loads.get_mut(&frame.source).unwrap();
    load.active_requests = 9;
    load.active_prefill_tokens = 1;
    assert_eq!(frame.select(Some(parameters())), frame.source);
}

#[test]
fn progress_group_cold_target_does_not_bypass_missing_source_progress() {
    let mut frame = Frame::new(Some(128), true);
    frame.request.affinity_target = Some(frame.other.into());
    let load = frame.request.worker_loads.get_mut(&frame.other).unwrap();
    load.active_requests = 1;
    load.active_prefill_tokens = 16_385;
    frame.guards.clear();
    assert_eq!(frame.select(Some(parameters())), frame.source);
}

#[test]
fn progress_group_both_warm_does_not_pin_a_busy_source() {
    let mut frame = Frame::new(Some(128), true);
    frame
        .request
        .overlap
        .tier_overlap_blocks
        .device
        .insert(frame.other, 256);
    assert_eq!(frame.select(Some(parameters())), frame.other);
}

#[test]
fn progress_group_uses_owning_native_scheduler_not_same_ids_in_another_scope() {
    let frame = Frame::new(Some(128), true);
    let other_scope = NativeSlots::new_without_expiry(
        NoopSequencePublisher,
        64,
        HashMap::from([
            (frame.source.worker_id, (0, 1)),
            (frame.other.worker_id, (0, 1)),
        ]),
        false,
        2,
        "aggregated",
    );
    assert_eq!(frame.select(Some(parameters())), frame.other);
    assert_eq!(
        frame.select_with_slots(Some(parameters()), &other_scope),
        frame.source
    );
}

#[test]
fn progress_group_factory_uses_legacy_selection_for_disaggregated_roles() {
    let frame = Frame::new(Some(128), true);
    let yaml = include_str!("../../../../examples/router/progress-group.yaml").replace(
        "  aggregated: progress-group",
        "  aggregated: progress-group\n  prefill: progress-group\n  decode: progress-group",
    );
    let file = tempfile::NamedTempFile::new().unwrap();
    std::fs::write(file.path(), yaml).unwrap();
    let config = KvRouterConfig {
        router_policy_config: Some(file.path().display().to_string()),
        router_track_output_blocks: true,
        ..Default::default()
    };
    let mut registry = crate::default_registry();
    crate::register(&mut registry).unwrap();
    let factory = registry.resolve(&config).unwrap().unwrap();
    for (role, expected) in [
        (dynamo_kv_router::WorkerType::Aggregated, frame.other),
        (dynamo_kv_router::WorkerType::Prefill, frame.source),
        (dynamo_kv_router::WorkerType::Decode, frame.source),
    ] {
        let policy = factory(
            &config,
            role,
            dynamo_kv_router::RoutingPartitionRef::new("model", "default"),
        );
        let provider = |worker| frame.slots.decode_progress_for_benchmark(worker);
        assert_eq!(
            policy
                .select_worker_with_decode_progress(
                    WorkerSelectionInput::configured(
                        &frame.workers,
                        &frame.request,
                        frame.request.eligibility(),
                        64
                    ),
                    None,
                    Some(&provider)
                )
                .unwrap()
                .0
                .worker,
            expected
        );
    }
}

struct Probe {
    declare: bool,
    seen: Arc<Mutex<Option<usize>>>,
}
impl WorkerPicker for Probe {
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.declare {
            WorkerInputs::DECODE_PROGRESS
        } else {
            WorkerInputs::NONE
        }
    }
    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        *self.seen.lock().unwrap() = context.decode_progress().map(|s| s.entries().len());
        Ok(0)
    }
}

#[test]
fn progress_group_public_picker_contract_requires_explicit_input_group() {
    let frame = Frame::new(Some(128), true);
    for declare in [false, true] {
        let seen = Arc::new(Mutex::new(None));
        let provider = |worker| frame.slots.decode_progress_for_benchmark(worker);
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "prefill",
            Vec::new(),
            Box::new(Probe {
                declare,
                seen: Arc::clone(&seen),
            }),
        )
        .select_worker_with_decode_progress(
            WorkerSelectionInput::configured(
                &frame.workers,
                &frame.request,
                frame.request.eligibility(),
                64,
            ),
            None,
            Some(&provider),
        )
        .unwrap();
        assert_eq!(*seen.lock().unwrap(), declare.then_some(1));
    }
}

#[test]
fn progress_group_configuration_is_explicit_and_validated() {
    assert!(parameters().validate().is_ok());
    for bad in [0, 4096] {
        let mut config = parameters();
        config.source_active_requests = bad;
        assert!(config.validate().is_err());
    }
}
