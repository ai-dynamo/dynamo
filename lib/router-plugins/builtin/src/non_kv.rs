// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Built-in adapters for the legacy non-KV worker-choice rules.
//!
//! These policies intentionally run inside the configured scheduler. Discovery, eligibility,
//! queueing, booking, dispatch, affinity, and metrics remain host-owned. Candidate rows are
//! grouped by worker before the shared runtime picker runs, preserving the legacy worker-level
//! unit; a second policy-family picker resolves a rank within the chosen worker.
//!
//! | policy | worker signal | mutable state | admission |
//! |---|---|---|---|
//! | round-robin | eligible IDs | partition-local cursor | scheduler lifecycle only |
//! | random | eligible IDs | RNG | scheduler lifecycle only |
//! | P2C | reservation occupancy | RNG | occupancy plus scheduler lifecycle |
//! | least-loaded | reservation occupancy | RNG tie-break | occupancy plus scheduler lifecycle |
//! | direct | exact target | none | scheduler lifecycle, no fallback |
//! | device-aware | device/cache hits and occupancy | RNG tie-break | no occupancy for full hits |
//!
//! The scheduler actor materializes occupancy and books the selected request before replying, so
//! selection and reservation are atomic at the same boundary. Its existing request guard owns
//! stream completion, cancellation, and retry cleanup. This is the configured-plane equivalent
//! of hosted occupancy without creating a second lifecycle owner.

use std::collections::BTreeMap;
use std::sync::Arc;

use dynamo_kv_router::KvRouterConfig;
use dynamo_kv_router::plugins::worker_selection::{
    WorkerDevice, WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionContext,
    WorkerSelectionPolicy, WorkerSelectionPolicyError, WorkerSelectionPolicyFactory,
};
use dynamo_kv_router::plugins::{
    RouterPluginRegistry, WorkerSelectionPolicyParameters, WorkerSelectionPolicyProviderError,
    WorkerSelectionPolicyRegistryError,
};
use dynamo_runtime::pipeline::{
    BuiltinRoutePicker, RouteCandidate, RouteContext, RouteDevice, RouteTarget,
};

pub const ROUND_ROBIN: &str = "dynamo-round-robin";
pub const RANDOM: &str = "dynamo-random";
pub const POWER_OF_TWO_CHOICES: &str = "dynamo-power-of-two-choices";
pub const LEAST_LOADED: &str = "dynamo-least-loaded";
pub const DIRECT: &str = "dynamo-direct";
pub const DEVICE_AWARE_WEIGHTED: &str = "dynamo-device-aware-weighted";

#[derive(Clone, Copy)]
enum Kind {
    RoundRobin,
    Random,
    PowerOfTwoChoices,
    LeastLoaded,
    Direct,
    DeviceAwareWeighted,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {}

struct NonKvPicker {
    kind: Kind,
    worker_picker: BuiltinRoutePicker,
    rank_pickers: BTreeMap<u64, BuiltinRoutePicker>,
    last_occupancy_admission: bool,
}

impl NonKvPicker {
    fn picker(kind: Kind) -> BuiltinRoutePicker {
        match kind {
            Kind::RoundRobin => BuiltinRoutePicker::round_robin(),
            Kind::Random => BuiltinRoutePicker::random(),
            Kind::PowerOfTwoChoices => BuiltinRoutePicker::power_of_two_choices(),
            Kind::LeastLoaded | Kind::DeviceAwareWeighted | Kind::Direct => {
                BuiltinRoutePicker::least_loaded()
            }
        }
    }

    fn new(kind: Kind) -> Self {
        let worker_picker = match kind {
            Kind::DeviceAwareWeighted => BuiltinRoutePicker::device_aware_weighted(),
            _ => Self::picker(kind),
        };
        Self {
            kind,
            worker_picker,
            rank_pickers: BTreeMap::new(),
            last_occupancy_admission: true,
        }
    }

    fn grouped_rows(input: WorkerInputView<'_>) -> BTreeMap<u64, Vec<usize>> {
        let mut grouped = BTreeMap::<u64, Vec<usize>>::new();
        for (row, candidate) in input.candidates().iter().enumerate() {
            grouped
                .entry(candidate.worker().worker_id)
                .or_default()
                .push(row);
        }
        for rows in grouped.values_mut() {
            rows.sort_unstable_by_key(|row| input.candidates()[*row].worker().dp_rank);
        }
        grouped
    }

    fn row_occupancy(input: WorkerInputView<'_>, row: usize) -> u64 {
        input
            .occupancy()
            .and_then(|occupancy| occupancy.get(row))
            .map(|occupancy| occupancy.active_requests())
            .unwrap_or_default()
    }

    fn resolve_rank(
        &mut self,
        input: WorkerInputView<'_>,
        worker_id: u64,
        rows: &[usize],
    ) -> Option<usize> {
        if rows.len() == 1 {
            return rows.first().copied();
        }
        let encoded = rows.iter().map(|row| *row as u64).collect::<Vec<_>>();
        self.rank_pickers
            .entry(worker_id)
            .or_insert_with(|| Self::picker(self.kind))
            .select_worker(&encoded, |row| Self::row_occupancy(input, row as usize))
            .map(|row| row as usize)
    }

    fn target_row(
        &mut self,
        input: WorkerInputView<'_>,
        grouped: &BTreeMap<u64, Vec<usize>>,
        target: dynamo_kv_router::protocols::WorkerAffinityTarget,
    ) -> Option<usize> {
        let rows = grouped.get(&target.worker_id)?;
        match target.dp_rank {
            Some(dp_rank) => rows
                .iter()
                .copied()
                .find(|row| input.candidates()[*row].worker().dp_rank == dp_rank),
            None if matches!(self.kind, Kind::Direct) => rows.first().copied(),
            None => self.resolve_rank(input, target.worker_id, rows),
        }
    }
}

impl WorkerPicker for NonKvPicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        match self.kind {
            Kind::PowerOfTwoChoices | Kind::LeastLoaded => WorkerInputs::OCCUPANCY,
            Kind::DeviceAwareWeighted => WorkerInputs::OCCUPANCY | WorkerInputs::DEVICE_AWARE,
            Kind::RoundRobin | Kind::Random | Kind::Direct => WorkerInputs::NONE,
        }
    }

    fn pick(
        &mut self,
        context: &WorkerSelectionContext<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let grouped = Self::grouped_rows(input);
        if grouped.is_empty() {
            return Err(WorkerSelectionPolicyError::failed("no eligible worker"));
        }

        if matches!(self.kind, Kind::Direct) {
            let target = context.exact_target().ok_or_else(|| {
                WorkerSelectionPolicyError::failed(
                    "Direct routing requires an exact affinity or request target",
                )
            })?;
            return self.target_row(input, &grouped, target).ok_or_else(|| {
                WorkerSelectionPolicyError::failed("direct target is not eligible")
            });
        }

        // Hosted non-KV modes route to a live explicit, hard-affinity, or soft-affinity target
        // before applying their picker. Keep that bypass here: it neither advances the
        // worker-level cursor nor consumes random samples. Device-aware hosted routing reserves a
        // target even on a complete cache hit, so this path deliberately retains occupancy.
        if let Some(target) = context.exact_target()
            && let Some(row) = self.target_row(input, &grouped, target)
        {
            self.last_occupancy_admission = true;
            return Ok(row);
        }

        let worker_ids = grouped.keys().copied().collect::<Vec<_>>();
        let worker_occupancy = |worker_id| {
            grouped[&worker_id]
                .iter()
                .map(|row| Self::row_occupancy(input, *row))
                .sum()
        };
        let worker_id = if matches!(self.kind, Kind::DeviceAwareWeighted) {
            let request = context.device_aware().ok_or_else(|| {
                WorkerSelectionPolicyError::failed("device-aware request input unavailable")
            })?;
            let device = input.device_aware().ok_or_else(|| {
                WorkerSelectionPolicyError::failed("device-aware worker input unavailable")
            })?;
            let candidates = worker_ids
                .iter()
                .map(|worker_id| {
                    let row = grouped[worker_id][0];
                    let worker = device[row];
                    let route_device = match worker.device() {
                        WorkerDevice::Cpu => RouteDevice::Cpu,
                        WorkerDevice::Accelerator => RouteDevice::Accelerator,
                    };
                    RouteCandidate::new(
                        RouteTarget::worker(*worker_id),
                        route_device,
                        worker.cache_hits(),
                    )
                })
                .collect::<Vec<_>>();
            let (target, admission) = self
                .worker_picker
                .select_device_aware(
                    &candidates,
                    RouteContext::new(
                        request.required_cache_hits(),
                        request.non_cpu_to_cpu_ratio(),
                    ),
                    worker_occupancy,
                )
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?;
            self.last_occupancy_admission = admission;
            target.worker_id
        } else {
            self.last_occupancy_admission = true;
            self.worker_picker
                .select_worker(&worker_ids, worker_occupancy)
                .ok_or_else(|| WorkerSelectionPolicyError::failed("no eligible worker"))?
        };

        self.resolve_rank(input, worker_id, &grouped[&worker_id])
            .ok_or_else(|| WorkerSelectionPolicyError::failed("selected worker has no rank"))
    }

    fn occupancy_admission(
        &mut self,
        _context: &WorkerSelectionContext<'_>,
        _input: WorkerInputView<'_>,
        _selected_row: usize,
    ) -> bool {
        self.last_occupancy_admission
    }

    fn uses_exclusive_affinity_target(&self) -> bool {
        matches!(self.kind, Kind::Direct)
    }

    fn requires_exact_target(&self) -> bool {
        matches!(self.kind, Kind::Direct)
    }

    fn resolves_worker_only_target(&self) -> bool {
        true
    }

    fn supports_lora(&self) -> bool {
        matches!(self.kind, Kind::RoundRobin | Kind::Random)
    }
}

fn provider(
    kind: Kind,
    parameters: &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError> {
    let _parameters: Parameters = parameters.deserialize()?;
    Ok(Arc::new(
        move |config: &KvRouterConfig, worker_type, _partition| {
            WorkerSelectionPolicy::new(
                config.clone(),
                worker_type.as_str(),
                Vec::new(),
                Box::new(NonKvPicker::new(kind)),
            )
        },
    ))
}

pub fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), WorkerSelectionPolicyRegistryError> {
    for (name, kind) in [
        (ROUND_ROBIN, Kind::RoundRobin),
        (RANDOM, Kind::Random),
        (POWER_OF_TWO_CHOICES, Kind::PowerOfTwoChoices),
        (LEAST_LOADED, Kind::LeastLoaded),
        (DIRECT, Kind::Direct),
        (DEVICE_AWARE_WEIGHTED, Kind::DeviceAwareWeighted),
    ] {
        registry.register_worker_selection(
            name,
            Arc::new(move |parameters| provider(kind, parameters)),
        )?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::{HashMap, HashSet};
    use std::sync::Arc;
    use std::time::Duration;

    use dynamo_kv_router::config::RouterQueuePolicy;
    use dynamo_kv_router::plugins::worker_selection::{
        DeviceAwareRequestInputs, WorkerDeviceAwareInput,
    };
    use dynamo_kv_router::protocols::{RoutingConstraints, WorkerConfigLike, WorkerWithDpRank};
    use dynamo_kv_router::scheduling::{
        LocalScheduler, NoopOverlapScoresRefresh, OverlapSignals, PolicyProfile,
        RoutingEligibility, ScheduleMode, ScheduleRequest, SchedulingRequest,
    };
    use dynamo_kv_router::{
        ActiveSequencesMultiWorker, NoopSequencePublisher, WorkerLoadProjection,
        WorkerSelectionInput, WorkerSelector,
    };
    use dynamo_runtime::CancellationToken;
    use tokio::sync::watch;

    use super::*;

    #[derive(Clone, PartialEq)]
    struct TestWorker {
        ranks: u32,
    }

    impl WorkerConfigLike for TestWorker {
        fn data_parallel_start_rank(&self) -> u32 {
            0
        }

        fn data_parallel_size(&self) -> u32 {
            self.ranks
        }

        fn max_num_batched_tokens(&self) -> Option<u64> {
            None
        }

        fn total_kv_blocks(&self) -> Option<u64> {
            Some(1024)
        }
    }

    fn request() -> SchedulingRequest {
        SchedulingRequest {
            mode: ScheduleMode::QueryOnly { request_id: None },
            token_seq: None,
            isl_tokens: 16,
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
            device_aware_inputs: None,
            worker_loads: Default::default(),
            resp_tx: None,
        }
    }

    fn policy(kind: Kind) -> WorkerSelectionPolicy {
        WorkerSelectionPolicy::new(
            KvRouterConfig::default(),
            "test",
            Vec::new(),
            Box::new(NonKvPicker::new(kind)),
        )
    }

    fn schedule_request(
        request_id: &str,
        pinned_worker: Option<WorkerWithDpRank>,
        device_aware_inputs: Option<DeviceAwareRequestInputs>,
    ) -> ScheduleRequest {
        ScheduleRequest {
            mode: ScheduleMode::Tracked {
                request_id: request_id.to_string(),
            },
            token_seq: None,
            block_hashes: None,
            isl_tokens: 16,
            lora_name: None,
            expected_output_tokens: None,
            affinity_target: None,
            pinned_worker,
            allowed_worker_ids: None,
            routing_constraints: RoutingConstraints::default(),
            router_config_override: None,
            priority_jump: 0.0,
            strict_priority: 0,
            policy_class: None,
            session_context: None,
            overlap: OverlapSignals::default(),
            kv_transfer_candidates: None,
            retain_kv_transfer_chain: false,
            shared_cache_hits: None,
            device_aware_inputs,
        }
    }

    fn scheduler(
        kind: Kind,
    ) -> (
        LocalScheduler<
            NoopSequencePublisher,
            TestWorker,
            WorkerSelectionPolicy,
            NoopOverlapScoresRefresh,
        >,
        CancellationToken,
    ) {
        let workers = HashMap::from([(10, TestWorker { ranks: 1 }), (20, TestWorker { ranks: 1 })]);
        let slots = Arc::new(ActiveSequencesMultiWorker::new_without_expiry(
            NoopSequencePublisher,
            16,
            HashMap::from([(10, (0, 1)), (20, (0, 1))]),
            false,
            0,
            "test",
        ));
        let (_workers_tx, workers_rx) = watch::channel(workers);
        let cancel = CancellationToken::new();
        let scheduler = LocalScheduler::new(
            slots,
            workers_rx,
            PolicyProfile::synthetic(None, RouterQueuePolicy::Fcfs),
            16,
            policy(kind),
            None,
            None,
            None,
            None,
            Duration::from_secs(60),
            false,
            cancel.clone(),
            "test",
            false,
        );
        (scheduler, cancel)
    }

    fn select(
        policy: &WorkerSelectionPolicy,
        workers: &HashMap<u64, TestWorker>,
        request: &SchedulingRequest,
    ) -> Result<
        dynamo_kv_router::protocols::WorkerSelectionResult,
        dynamo_kv_router::KvSchedulerError,
    > {
        policy.select_worker(WorkerSelectionInput::configured(
            workers,
            request,
            RoutingEligibility::new(
                request.allowed_worker_ids.as_ref(),
                None,
                request.pinned_worker,
                &request.routing_constraints,
            ),
            16,
        ))
    }

    #[test]
    fn round_robin_cycles_workers_before_ranks() {
        let workers = HashMap::from([(20, TestWorker { ranks: 2 }), (10, TestWorker { ranks: 2 })]);
        let request = request();
        let policy = policy(Kind::RoundRobin);
        let selected = (0..4)
            .map(|_| select(&policy, &workers, &request).unwrap().worker)
            .collect::<Vec<_>>();
        assert_eq!(
            selected,
            [
                WorkerWithDpRank::new(10, 0),
                WorkerWithDpRank::new(20, 0),
                WorkerWithDpRank::new(10, 1),
                WorkerWithDpRank::new(20, 1),
            ]
        );
    }

    #[test]
    fn least_loaded_uses_reservation_occupancy_aggregated_by_worker() {
        let workers = HashMap::from([(10, TestWorker { ranks: 2 }), (20, TestWorker { ranks: 1 })]);
        let mut request = request();
        request.worker_loads.insert(
            WorkerWithDpRank::new(10, 0),
            WorkerLoadProjection {
                routing_occupancy: 2,
                ..Default::default()
            },
        );
        request.worker_loads.insert(
            WorkerWithDpRank::new(10, 1),
            WorkerLoadProjection {
                routing_occupancy: 1,
                ..Default::default()
            },
        );
        request.worker_loads.insert(
            WorkerWithDpRank::new(20, 0),
            WorkerLoadProjection {
                routing_occupancy: 2,
                ..Default::default()
            },
        );
        assert_eq!(
            select(&policy(Kind::LeastLoaded), &workers, &request)
                .unwrap()
                .worker,
            WorkerWithDpRank::new(20, 0)
        );
    }

    #[test]
    fn direct_requires_and_preserves_an_exact_target() {
        let workers = HashMap::from([(10, TestWorker { ranks: 1 }), (20, TestWorker { ranks: 2 })]);
        let policy = policy(Kind::Direct);
        let mut targetless = request();
        let error = select(&policy, &workers, &targetless).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Direct routing requires an exact affinity or request target")
        );

        targetless.affinity_target = Some(dynamo_kv_router::protocols::WorkerAffinityTarget::new(
            20, None,
        ));
        assert_eq!(
            select(&policy, &workers, &targetless).unwrap().worker,
            WorkerWithDpRank::new(20, 0)
        );
    }

    #[test]
    fn non_direct_modes_prefer_an_eligible_soft_affinity_target() {
        let workers = HashMap::from([(10, TestWorker { ranks: 1 }), (20, TestWorker { ranks: 1 })]);
        for kind in [
            Kind::RoundRobin,
            Kind::Random,
            Kind::PowerOfTwoChoices,
            Kind::LeastLoaded,
            Kind::DeviceAwareWeighted,
        ] {
            let mut request = request();
            request.affinity_target = Some(dynamo_kv_router::protocols::WorkerAffinityTarget::new(
                20, None,
            ));
            if matches!(kind, Kind::DeviceAwareWeighted) {
                request.device_aware_inputs = Some(DeviceAwareRequestInputs::new(
                    [
                        (10, WorkerDeviceAwareInput::new(WorkerDevice::Cpu, 0)),
                        (
                            20,
                            WorkerDeviceAwareInput::new(WorkerDevice::Accelerator, 1),
                        ),
                    ],
                    1,
                    8,
                ));
            }
            let selected = select(&policy(kind), &workers, &request).unwrap();
            assert_eq!(selected.worker.worker_id, 20);
            assert!(selected.occupancy_admission);
        }
    }

    #[test]
    fn exact_target_does_not_advance_round_robin_worker_state() {
        let workers = HashMap::from([(10, TestWorker { ranks: 1 }), (20, TestWorker { ranks: 1 })]);
        let policy = policy(Kind::RoundRobin);
        let mut targeted = request();
        targeted.pinned_worker = Some(WorkerWithDpRank::new(20, 0));
        assert_eq!(
            select(&policy, &workers, &targeted).unwrap().worker,
            WorkerWithDpRank::new(20, 0)
        );
        assert_eq!(
            select(&policy, &workers, &request()).unwrap().worker,
            WorkerWithDpRank::new(10, 0)
        );
    }

    #[test]
    fn device_aware_full_cache_hit_wins_without_occupancy_admission() {
        let workers = HashMap::from([(10, TestWorker { ranks: 1 }), (20, TestWorker { ranks: 1 })]);
        let mut request = request();
        request.device_aware_inputs = Some(DeviceAwareRequestInputs::new(
            [
                (10, WorkerDeviceAwareInput::new(WorkerDevice::Cpu, 0)),
                (
                    20,
                    WorkerDeviceAwareInput::new(WorkerDevice::Accelerator, 2),
                ),
            ],
            2,
            8,
        ));
        let selected = select(&policy(Kind::DeviceAwareWeighted), &workers, &request).unwrap();
        assert_eq!(selected.worker, WorkerWithDpRank::new(20, 0));
        assert!(!selected.occupancy_admission);
    }

    #[test]
    fn random_and_p2c_never_escape_eligibility() {
        let workers = HashMap::from([
            (10, TestWorker { ranks: 1 }),
            (20, TestWorker { ranks: 1 }),
            (30, TestWorker { ranks: 1 }),
        ]);
        let mut request = request();
        request.allowed_worker_ids = Some(HashSet::from([10, 30]));
        for kind in [Kind::Random, Kind::PowerOfTwoChoices] {
            let policy = policy(kind);
            for _ in 0..128 {
                let worker = select(&policy, &workers, &request)
                    .unwrap()
                    .worker
                    .worker_id;
                assert!(worker == 10 || worker == 30);
            }
        }
    }

    #[test]
    fn lora_support_matches_legacy_non_kv_modes() {
        for (kind, expected) in [
            (Kind::RoundRobin, true),
            (Kind::Random, true),
            (Kind::PowerOfTwoChoices, false),
            (Kind::LeastLoaded, false),
            (Kind::Direct, false),
            (Kind::DeviceAwareWeighted, false),
        ] {
            assert_eq!(
                <WorkerSelectionPolicy as WorkerSelector<TestWorker>>::supports_lora(&policy(kind)),
                expected
            );
        }
    }

    #[tokio::test]
    async fn every_policy_uses_scheduler_booking_and_cleanup_lifecycle() {
        for kind in [
            Kind::RoundRobin,
            Kind::Random,
            Kind::PowerOfTwoChoices,
            Kind::LeastLoaded,
            Kind::Direct,
            Kind::DeviceAwareWeighted,
        ] {
            let (scheduler, cancel) = scheduler(kind);
            let pinned = matches!(kind, Kind::Direct).then(|| WorkerWithDpRank::new(20, 0));
            let device = matches!(kind, Kind::DeviceAwareWeighted).then(|| {
                DeviceAwareRequestInputs::new(
                    [
                        (10, WorkerDeviceAwareInput::new(WorkerDevice::Cpu, 0)),
                        (
                            20,
                            WorkerDeviceAwareInput::new(WorkerDevice::Accelerator, 1),
                        ),
                    ],
                    1,
                    8,
                )
            });
            let response = scheduler
                .schedule_request(schedule_request("lifecycle", pinned, device))
                .await
                .unwrap();
            assert!(matches!(response.best_worker.worker_id, 10 | 20));
            assert_eq!(
                scheduler
                    .get_potential_loads(None, 0, HashMap::new(), false)
                    .into_iter()
                    .map(|load| load.active_requests)
                    .sum::<usize>(),
                1
            );
            scheduler.free("lifecycle").await.unwrap();
            assert_eq!(
                scheduler
                    .get_potential_loads(None, 0, HashMap::new(), false)
                    .into_iter()
                    .map(|load| load.active_requests)
                    .sum::<usize>(),
                0
            );
            cancel.cancel();
        }
    }

    #[tokio::test]
    async fn occupancy_policies_observe_atomic_scheduler_reservations() {
        for kind in [Kind::PowerOfTwoChoices, Kind::LeastLoaded] {
            let (scheduler, cancel) = scheduler(kind);
            let first = scheduler
                .schedule_request(schedule_request("first", None, None))
                .await
                .unwrap();
            let second = scheduler
                .schedule_request(schedule_request("second", None, None))
                .await
                .unwrap();
            assert_ne!(first.best_worker.worker_id, second.best_worker.worker_id);
            scheduler.free("first").await.unwrap();
            scheduler.free("second").await.unwrap();
            cancel.cancel();
        }
    }

    #[tokio::test]
    async fn direct_targetless_request_is_rejected_before_booking() {
        let (scheduler, cancel) = scheduler(Kind::Direct);
        let error = scheduler
            .schedule_request(schedule_request("targetless", None, None))
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            dynamo_kv_router::KvSchedulerError::DirectTargetRequired
        ));
        assert_eq!(
            scheduler
                .get_potential_loads(None, 0, HashMap::new(), false)
                .into_iter()
                .map(|load| load.active_requests)
                .sum::<usize>(),
            0
        );
        cancel.cancel();
    }
}
